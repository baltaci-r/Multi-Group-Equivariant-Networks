import random

import torch
import torch.nn as nn

from perm_equivariant_seq2seq.eq_utils import *
from perm_equivariant_seq2seq.equitune_modules import EquiSCANLinear
from torchvision.transforms.functional import hflip, vflip
from perm_equivariant_seq2seq.g_utils import g_transform_data, g_inverse_transform_data_prob


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class CanonicalModel(nn.Module):
    def __init__(self, pre_model, in_G, out_G, vocab_size=8, eq_word_indices=[2, 7], feature_extracting=True, group_type='cyclic'):
        super(CanonicalModel, self).__init__()
        self.in_G = in_G
        self.out_G = out_G
        self.vocab_size = vocab_size
        self.eq_word_indices = eq_word_indices

        set_parameter_requires_grad(pre_model, feature_extracting)
        self.pre_model = pre_model  # requires grad set to false already
        self.nonlin = nn.ReLU()

    def forward(self, input_tensor, target_tensor, use_teacher_forcing=False):

        transformed_input_tensor = g_transform_data(input_tensor, self.in_G.g, device)  # generate transformed inputs for equituning
        transformed_target_tensor = g_transform_data(target_tensor, self.out_G.g, device)  # generate transformed target_tensors only for teacher forcing

        # get transformed outputs
        transformed_outputs = []
        for i in range(len(transformed_input_tensor)):
            curr_input = transformed_input_tensor[i]
            curr_target_tensor = transformed_target_tensor[i]
            model_output = self.pre_model(input_tensor=curr_input,
                                          target_tensor=curr_target_tensor,
                                          use_teacher_forcing=use_teacher_forcing)
            transformed_outputs.append(model_output)

        # add inv_transforms here
        transformed_outputs = torch.stack(transformed_outputs)
        outputs = g_inverse_transform_data_prob(transformed_outputs, G=self.out_G.g)  # dims [group_size, max_sentence_len=50, vocab_size=8]
        model_output = torch.mean(outputs, dim=0, keepdim=False)  # dim [max_sentence_len=50, vocab_size=8]

        return model_output
