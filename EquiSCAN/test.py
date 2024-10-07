# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random, argparse, os, torch, json, logging
from tqdm import tqdm

import perm_equivariant_seq2seq.utils as utils
from utils import set_seed, count_parameters
from perm_equivariant_seq2seq.models import BasicSeq2Seq
from perm_equivariant_seq2seq.multiequitune_models import MultiEquiSCANModel
from perm_equivariant_seq2seq.equitune_models import EquiSCANModel
from perm_equivariant_seq2seq.data_utils import get_scan_split, get_invariant_scan_languages
from perm_equivariant_seq2seq.utils import tensors_from_pair, tensor_from_sentence
from perm_equivariant_seq2seq.language_utils import SCANMultiGroup
from itertools import product


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1

# Parse command-line arguments
parser = argparse.ArgumentParser()
# Experiment options
parser.add_argument('--load_model_path',
                    type=str, 
                    help='Path to experiment directory, should contain args and model')
parser.add_argument('--config', type=str, choices=["equi", "multi_equi", "pretrain"])
parser.add_argument('--split',
                    default='add_jump',
                    choices=[None, 'simple', 'add_jump', 'around_right', 'jump', 'turn_left', 'turn_up', 'turn_up_jump_turn_left'],
                    help='Each possible split defines a different experiment as proposed by [1]')
parser.add_argument('--use_attention',
                    dest='use_attention',
                    default=False,
                    action='store_true',
                    help="Boolean to use attention in the decoder")
parser.add_argument('--bidirectional',
                    dest='bidirectional',
                    default=False,
                    action='store_true',
                    help="Boolean to use bidirectional encoder")
parser.add_argument('--expanded',
                    default=False,
                    action='store_true',
                    help="Boolean to use expanded dataset")
parser.add_argument('--equivariance',
                    # default='none',
                    action='append',
                    choices=['verb', 'direction-rl', 'direction-ud'])
parser.add_argument('--layer_type',
                    choices=['LSTM', 'GRU', 'RNN'],
                    default='LSTM',
                    help='Type of rnn layers to be used for recurrent components')
parser.add_argument('--pretrained_model_dir',
                    type=str,
                    default=None,
                    help='path to pretrained model')
parser.add_argument('--cuda',
                    type=int,
                    help="CUDA device")
parser.add_argument('--feature_extracting',
                    dest='feature_extracting',
                    default=False,
                    action='store_true',
                    help="Boolean to freeze pretrained model")
args = parser.parse_args()
# Model options
parser.add_argument('--seed', default=0, type=int)
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
parser.add_argument('--drop_rate',
                    type=float, 
                    default=0.1, 
                    help="Dropout drop rate (not keep rate)")
# Optimization and training hyper-parameters
parser.add_argument('--validation_size',
                    type=float, 
                    default=0.01,
                    help='Validation proportion to use for early-stopping')
parser.add_argument('--n_iters', 
                    type=int, 
                    default=200000, 
                    help='number of training iterations')
parser.add_argument('--learning_rate', 
                    type=float, 
                    default=1e-4, 
                    help='init learning rate')
parser.add_argument('--teacher_forcing_ratio', 
                    type=float, 
                    default=0.5)
parser.add_argument('--save_dir', 
                    type=str, 
                    default='./models/', 
                    help='Top-level directory for saving experiment')
parser.add_argument('--print_freq', 
                    type=int, 
                    default=1000, 
                    help='Frequency with which to print training loss')
parser.add_argument('--plot_freq', 
                    type=int, 
                    default=20, 
                    help='Frequency with which to plot training loss')
parser.add_argument('--save_freq', 
                    type=int,
                    default=200000, 
                    help='Frequency with which to save models during training')
parser.add_argument('--group_type',
                    choices=['cyclic', 'permutation', 'none'],
                    default='permutation',
                    help='group type for equivariance')
def evaluate(model_to_eval,
             inp_lang,
             syntax_lang,
             out_lang,
             sentence,
             target):
    """Decode one sentence from input -> output language with the model

    Args:
        model_to_eval: (nn.Module: Seq2SeqModel) seq2seq model being evaluated
        inp_lang: (Lang) Language object for input language
        syntax_lang: (InvariantLang) Language object for input language
        out_lang: (Lang) Language object for output language
        sentence: (torch.tensor) Tensor representation (1-hot) of sentence in 
        input language
    Returns:
        (list) Words in output language as decoded by model
    """
    model.eval()
    with torch.no_grad():
        input_tensor = tensor_from_sentence(inp_lang, sentence)
        target_tensor = tensor_from_sentence(out_lang, target)
        model_output = model_to_eval(input_tensor=input_tensor, target_tensor=target_tensor)

        decoded_words = []
        for di in range(model_to_eval.pre_model.max_length):
            topv, topi = model_output[di].data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(out_lang.index2word[topi.item()])
        return decoded_words


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
    # Make sure all data is contained in the directory and load arguments
    args_path = os.path.join(args.load_model_path, "commandline_args.txt")
    model_path = os.path.join(args.load_model_path, "best_validation.pt")

    assert os.path.exists(args.load_model_path), "Experiment directory not found"
    assert os.path.exists(model_path), "Model number not found in directory"
    assert os.path.exists(args_path), "Argparser details directory not found in directory"

    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(args.load_model_path, "test.log"), filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    load_model_args = utils.load_args_from_txt(parser, args_path)

    # Load data
    train_pairs, _ = get_scan_split(split=load_model_args.split, expanded=load_model_args.expanded)
    commands, actions = get_invariant_scan_languages(train_pairs, invariances=[])

    # get commands, actions, indices, group generators
    equivariant_commands, equivariant_actions = get_invariant_scan_languages(train_pairs, invariances=[])

    group = SCANMultiGroup(load_model_args.equivariance, equivariant_commands, equivariant_actions,
                           product=load_model_args.config == 'equi')

    # Initialize model
    model = BasicSeq2Seq(input_language=equivariant_commands,
                         encoder_hidden_size=load_model_args.hidden_size,
                         decoder_hidden_size=load_model_args.semantic_size,
                         output_language=equivariant_actions,
                         layer_type=load_model_args.layer_type,
                         use_attention=load_model_args.use_attention,
                         drop_rate=load_model_args.drop_rate,
                         bidirectional=load_model_args.bidirectional,
                         num_layers=load_model_args.num_layers)

    if load_model_args.config == 'multi_equi':
        # initialize multiequitune model
        model = MultiEquiSCANModel(pre_model=model, in_G=group.in_g, out_G=group.out_g,
                                   vocab_size=equivariant_actions.n_words,
                                   eq_word_indices=group.eq_word_indices,
                                   feature_extracting=load_model_args.feature_extracting,
                                   group_type=load_model_args.group_type)
    elif load_model_args.config == 'equi':
        # initialize equitune model
        model = EquiSCANModel(pre_model=model, in_G=group.in_g, out_G=group.out_g,
                              vocab_size=equivariant_actions.n_words,
                              eq_word_indices=group.eq_word_indices,
                              feature_extracting=load_model_args.feature_extracting,
                              group_type=load_model_args.group_type)
    else:
        raise NotImplementedError

    # Move model to device and load weights
    logging.info("Loading model from {}".format(model_path))
    load_model_path = os.path.join(args.load_model_path, "best_validation.pt")
    model_state_dicts = torch.load(load_model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(model_state_dicts)
    logging.info("num of parameters: {} ".format(count_parameters(model)))

    logging.info("Loading Dataset with split: {} ".format(args.split))
    train_pairs, test_pairs = get_scan_split(split=args.split, expanded=args.expanded)

    # Convert data to torch tensors
    testing_pairs = [tensors_from_pair(pair, commands, actions) for pair in test_pairs]

    logging.info("Computing Test Accuracy")
    test_acc = test_accuracy(model, testing_pairs)
    logging.info("Model test accuracy: %s" % test_acc.item())

    fname = os.path.join(args.load_model_path, 'results.jsonl')
    logging.info("Saving test accuracy to: %s" % fname)
    with open(fname, 'a') as f:
        f.write(json.dumps({
            'config': load_model_args.config,
            'train_split': load_model_args.split,
            'train_seed': load_model_args.seed,
            'layer_type': load_model_args.layer_type,
            'test_split': args.split,
            'test_acc': test_acc.item()
        }) + '\n')

    pair = random.choice(test_pairs)
    print('>', pair[0])
    print('=', pair[1])
    output_words = evaluate(model, commands, None, actions, pair[0], pair[1])
    output_sentence = ' '.join(output_words)
    print('<', output_sentence)
    print('')
