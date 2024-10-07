import random
import argparse
import os
import logging
from tqdm import tqdm
import torch
import torch.nn as nn

import perm_equivariant_seq2seq.utils as utils
from utils import set_seed, count_parameters
from perm_equivariant_seq2seq.models import BasicSeq2Seq
from perm_equivariant_seq2seq.multiequitune_models import MultiEquiSCANModel
from perm_equivariant_seq2seq.equitune_models import EquiSCANModel
from perm_equivariant_seq2seq.canonical_models import CanonicalModel
from perm_equivariant_seq2seq.data_utils import get_scan_split, get_invariant_scan_languages
from perm_equivariant_seq2seq.utils import tensors_from_pair
from perm_equivariant_seq2seq.language_utils import SCANMultiGroup, SCANGroup
import time
import json


time_start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1

parser = argparse.ArgumentParser()

# Model options
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--config', type=str, choices=["equi", "multi_equi", "canonical"])
parser.add_argument('--layer_type',
                    choices=['LSTM', 'GRU', 'RNN'],
                    default='RNN',
                    help='Type of rnn layers to be used for recurrent components')
parser.add_argument('--hidden_size',
                    type=int,
                    default=64,
                    help='Number of hidden units in encoder / decoder')
parser.add_argument('--semantic_size',
                    type=int,
                    default=64,
                    help='Dimensionality of semantic embedding')
parser.add_argument('--num_layers',
                    type=int,
                    default=1,
                    help='Number of hidden layers in encoder')
parser.add_argument('--use_attention',
                    dest='use_attention',
                    default=True,
                    action='store_true',
                    help="Boolean to use attention in the decoder")
parser.add_argument('--bidirectional',
                    dest='bidirectional',
                    default=True,
                    action='store_true',
                    help="Boolean to use bidirectional encoder")

parser.add_argument('--drop_rate',
                    type=float,
                    default=0.1,
                    help="Dropout drop rate (not keep rate)")

# Optimization and training hyper-parameters
parser.add_argument('--split',
                    default='add_jump',
                    choices=[None, 'simple', 'add_jump', 'around_right', 'jump', 'turn_left', 'turn_up', 'turn_up_jump_turn_left'],
                    help='Each possible split defines a different experiment as proposed by [1]')
parser.add_argument('--validation_size',
                    type=float,
                    default=0.2,
                    help='Validation proportion to use for early-stopping')
parser.add_argument('--n_iters',
                    type=int,
                    default=10000,
                    help='number of training iterations')
parser.add_argument('--learning_rate',
                    type=float,
                    default=5*1e-5,  # set to 2*1e-5 for around_right, 5*1e-5 for add_jump
                    help='init learning rate')
parser.add_argument('--teacher_forcing_ratio',
                    type=float,
                    default=0.5)
parser.add_argument('--save_dir',
                    type=str,
                    # default='./finetuned_models/',
                    help='Top-level directory for saving multiequituned experiment')
parser.add_argument('--print_freq',
                    type=int,
                    default=100,
                    help='Frequency with which to print training loss')
parser.add_argument('--save_freq',
                    type=int,
                    # default=100,
                    default=1000,
                    help='Frequency with which to save models during training')

# multiequituning options
parser.add_argument('--load_model_path',
                    type=str,
                    default='./models/add_jump/rnn_RNN_hidden_64_directions_2/seed_0/model0/',
                    help='path to model checkpoint')

parser.add_argument('--group_type',
                    choices=['cyclic', 'permutation', 'none'],
                    default='permutation',
                    help='group type for equivariance')

parser.add_argument('--equivariance',
                    # default='verb',
                    action='append',
                    choices=['verb', 'direction-rl', 'direction-ud', 'none'])

parser.add_argument('--feature_extracting',
                    dest='feature_extracting',
                    default=False,
                    action='store_true',
                    help="Boolean to freeze pretrained model")
parser.add_argument('--expanded',
                    action='store_true',
                    help="Boolean to expanded dataset")
parser.add_argument('--cuda',
                    type=int,
                    help="CUDA device")

args = parser.parse_args()

set_seed(args.seed)
args.save_path = os.path.join(args.save_dir,
                              # '%s' % args.split,
                              'rnn_%s_hidden_%s_directions_%s' % (
                                  args.layer_type,
                                  args.hidden_size,
                                  2 if args.bidirectional else 1
                              ),
                              'seed_%s' % args.seed)
# Create model directory
if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)


def finetune(input_tensor,
          target_tensor,
          model_to_train,
          enc_optimizer,
          dec_optimizer,
          loss_fn,
          teacher_forcing_ratio):
    """Perform one training iteration for the model

    Args:
        input_tensor: (torch.tensor) Tensor representation (1-hot) of sentence
        in input language
        target_tensor: (torch.tensor) Tensor representation (1-hot) of target
        sentence in output language
        model_to_train: (nn.Module: Seq2SeqModel) seq2seq model being trained
        in_G: input group
        out_G: output_group
        enc_optimizer: (torch.optimizer) Optimizer object for model encoder
        dec_optimizer: (torch.optimizer) Optimizer object for model decoder
        loss_fn: (torch.nn.Loss) Loss object used for training
        teacher_forcing_ratio: (float) Ratio with which true word is used as
        input to decoder
    Returns:
        (torch.scalar) Value of loss achieved by model at current iteration
    """
    model_to_train.train()
    # Forget gradients via optimizers
    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    model_output = model_to_train(input_tensor=input_tensor,
                                  target_tensor=target_tensor,
                                  use_teacher_forcing=use_teacher_forcing)
    train_loss = 0

    target_length = target_tensor.size(0)
    for di in range(target_length):
        decoder_output = model_output[di]
        train_loss += loss_fn(decoder_output[None, :], target_tensor[di])
        _, decoder_output_symbol = decoder_output.topk(1)
        if decoder_output_symbol.item() == EOS_token:
            break
    train_loss.backward()

    # Clip gradients by norm (5.) and take optimization step
    torch.nn.utils.clip_grad_norm_(model_to_train.pre_model.encoder.parameters(), 5.)
    torch.nn.utils.clip_grad_norm_(model_to_train.pre_model.decoder.parameters(), 5.)
    enc_optimizer.step()
    dec_optimizer.step()

    return train_loss.item() / target_length


def test_accuracy(model_to_test, pairs):
    """Test a model (metric: accuracy) on all pairs in _pairs_

    Args:
        model_to_test: (seq2seq) Model object to be tested
        pairs: (list::pairs) List of list of input/output language pairs
    Returns:
        (float) accuracy on test pairs
    """

    def sentence_correct(target, model_sentence):
        # First, extract sentence up to EOS
        _, sentence_ints = model_sentence.data.topk(1)
        # If there is no EOS token, take the complete list
        try:
            eos_location = (sentence_ints == EOS_token).nonzero()[0][0]
        except:
            eos_location = len(sentence_ints) - 2
        model_sentence = sentence_ints[:eos_location + 1]
        # Check length is correct
        if len(model_sentence) != len(target):
            return torch.tensor(0, device=device)
        else:
            correct = model_sentence == target
            return torch.prod(correct).to(device)

    accuracies = []
    model.eval()
    with torch.no_grad():
        for pair in tqdm(pairs):
            input_tensor, output_tensor = pair
            # fixme: would not not work with pretrained model
            model_output = model_to_test(input_tensor=input_tensor, target_tensor=output_tensor)
            accuracies.append(sentence_correct(output_tensor, model_output))
    return torch.stack(accuracies).type(torch.float).mean()


if __name__ == '__main__':

    # Load data
    train_pairs, test_pairs = get_scan_split(split=args.split, expanded=args.expanded)
    commands, actions = get_invariant_scan_languages(train_pairs, invariances=[])

    # get commands, actions, indices, group generators
    equivariant_commands, equivariant_actions = get_invariant_scan_languages(train_pairs, invariances=[])
    if args.config == 'multi_equi':
        group = SCANMultiGroup(args.equivariance, equivariant_commands, equivariant_actions)
    else:
        group = SCANGroup(args.equivariance, equivariant_commands, equivariant_actions, canonical=args.config=='canonical')

    # Initialize model
    model = BasicSeq2Seq(input_language=equivariant_commands,
                         encoder_hidden_size=args.hidden_size,
                         decoder_hidden_size=args.hidden_size,
                         output_language=equivariant_actions,
                         layer_type=args.layer_type,
                         use_attention=args.use_attention,
                         drop_rate=args.drop_rate,
                         bidirectional=args.bidirectional,
                         num_layers=args.num_layers)

    # Move model to device and load weights
    logging.info("Loading model from {}".format(args.load_model_path))
    load_model_path = os.path.join(args.load_model_path, "best_validation.pt")
    model_state_dicts = torch.load(load_model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(model_state_dicts)

    if args.config == 'multi_equi':
        # initialize multiequitune model
        model = MultiEquiSCANModel(pre_model=model, in_G=group.in_g, out_G=group.out_g,
                                   vocab_size=equivariant_actions.n_words,
                                   eq_word_indices=group.eq_word_indices,
                                   feature_extracting=args.feature_extracting,
                                   group_type=args.group_type)
    else:
        # initialize equitune model
        model = EquiSCANModel(pre_model=model, in_G=group.in_g, out_G=group.out_g,
                              vocab_size=equivariant_actions.n_words,
                              eq_word_indices=group.eq_word_indices,
                              feature_extracting=args.feature_extracting,
                              group_type=args.group_type)

    model_path = utils.create_exp_dir(args)
    print(os.path.join(model_path, f"{args.config}tune.log"))
    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(model_path, f"{args.config}tune.log"),  force=True, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

    logging.info("num of parameters: {} ".format(count_parameters(model)))
    # Split off validation set
    val_size = int(len(train_pairs) * args.validation_size)
    random.shuffle(train_pairs)
    train_pairs, val_pairs = train_pairs[val_size:], train_pairs[:val_size]

    # Convert data to torch tensors
    training_pairs = [tensors_from_pair(random.choice(train_pairs), commands, actions)
                      for i in range(args.n_iters)]
    training_eval = [tensors_from_pair(pair, commands, actions)
                     for pair in train_pairs]
    validation_pairs = [tensors_from_pair(pair, commands, actions)
                        for pair in val_pairs]
    testing_pairs = [tensors_from_pair(pair, commands, actions)
                     for pair in test_pairs]

    # Initialize criterion
    criterion = nn.NLLLoss().to(device)

    # Initialize optimizers
    encoder_optimizer = torch.optim.Adam(model.pre_model.encoder.parameters(),
                                         lr=args.learning_rate,
                                         weight_decay=0.0001)
    decoder_optimizer = torch.optim.Adam(model.pre_model.decoder.parameters(),
                                         lr=args.learning_rate,
                                         weight_decay=0.0001)

    # Initialize printing / plotting variables
    print_loss_total = 0

    # Enter training loop
    best_acc = 0.

    logging.info(" *** Dataset *** ")
    logging.info("   Train: {}".format(len(train_pairs)))
    logging.info("   Validation: {}".format(len(validation_pairs)))
    logging.info("   Test: {}".format(len(testing_pairs)))

    logging.info(" *** Device *** ")
    logging.info("   Device: {}, Index: {}".format(device.type, device.index))

    logging.info(" *** Model *** ")
    logging.info("   Params: {}".format(sum(p.numel() for p in model.parameters())))
    logging.info("   Trainable: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    logging.info(" *** ")

    val_accuracies = []
    test_accuracies = []

    for iteration in range(0, args.n_iters):
        if (iteration) % 100 == 0:
            logging.info(f"iteration: {iteration}")

        # Grab iteration translation triplet (input tensor, syntax tensor, output tensor)
        training_pair = training_pairs[iteration]
        iteration_input, iteration_output = training_pair

        # Compute loss (and take one gradient step)
        loss = finetune(
            input_tensor=iteration_input,
            target_tensor=iteration_output,
            model_to_train=model,
            enc_optimizer=encoder_optimizer,
            dec_optimizer=decoder_optimizer,
            loss_fn=criterion,
            teacher_forcing_ratio=args.teacher_forcing_ratio
        )

        print_loss_total += loss

        # Print, plot, etc'
        if iteration % args.print_freq == 0:
            print_loss_avg = print_loss_total / args.print_freq
            print_loss_total = 0
            logging.info('%s iterations: %s' % (iteration, print_loss_avg))

        if iteration % args.save_freq == 0:
            # save model if is better
            if args.validation_size > 0.:
                val_acc = test_accuracy(model, validation_pairs).item()
                val_accuracies.append(val_acc)
                if val_acc > best_acc:
                    best_acc = val_acc
                    save_path = os.path.join(model_path, 'best_validation.pt')
                    logging.info('Best validation accuracy at iteration %s: %s' % (iteration, val_acc))
                    torch.save(model.state_dict(), save_path)

    # Save fully trained model
    save_path = os.path.join(model_path, 'model_fully_trained.pt')
    torch.save(model.state_dict(), save_path)

    test_acc = test_accuracy(model, testing_pairs)
    test_accuracies.append(test_acc)
    logging.info("Model test accuracy: %s" % test_acc.item())

    time_elapsed = (time.time() - time_start)
