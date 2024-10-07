import torch.nn as nn
import torch

from perm_equivariant_seq2seq.eq_utils import *
from perm_equivariant_seq2seq.equitune_modules import EquiSCANLinear
from perm_equivariant_seq2seq.g_utils import g_transform_data, g_invariant_data, g_inv_transform_prob_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class MultiEquiSCANModel(nn.Module):
    def __init__(self, pre_model, in_G, out_G, vocab_size=8, eq_word_indices=[2, 7], feature_extracting=True, group_type='cyclic'):
        super(MultiEquiSCANModel, self).__init__()

        self.in_G = in_G
        self.out_G = out_G

        self.num_G = len(self.in_G)

        self.vocab_size = vocab_size
        self.eq_word_indices = eq_word_indices

        set_parameter_requires_grad(pre_model, feature_extracting)
        self.pre_model = pre_model  # requires grad set to false already
        self.nonlin = nn.ReLU()

    def forward(self, input_tensor, target_tensor, use_teacher_forcing=False):

        model_outputs = []
        for i in range(self.num_G):
            inds = list(range(self.num_G))
            inds.remove(i)

            in_G = [self.in_G[j].g for j in inds]
            in_G_inv = [self.in_G[j].inv for j in inds]

            out_G = [self.out_G[j].g for j in inds]
            out_G_inv = [self.out_G[j].inv for j in inds]

            invariant_input_tensor = g_invariant_data(input_tensor, in_G_inv, device)
            invariant_target_tensor = g_invariant_data(target_tensor, out_G_inv, device)

            transformed_input_tensor = g_transform_data(invariant_input_tensor, self.in_G[i].g, device)
            transformed_target_tensor = g_transform_data(invariant_target_tensor, self.out_G[i].g, device)

            # get transformed outputs
            transformed_outputs = []
            for j in range(len(transformed_input_tensor)):
                curr_input = transformed_input_tensor[j]
                curr_target_tensor = transformed_target_tensor[j]
                model_output = self.pre_model(input_tensor=curr_input,
                                              target_tensor=curr_target_tensor,
                                              use_teacher_forcing=use_teacher_forcing)
                transformed_outputs.append(model_output)

            # add inv_transforms here
            transformed_outputs = torch.stack(transformed_outputs)
            outputs = g_inv_transform_prob_data(transformed_outputs, G=self.out_G[i].g)

            model_output = torch.mean(outputs, dim=0, keepdim=False)[None, :, :]

            for G in out_G:
                model_output = torch.mean(g_inv_transform_prob_data(model_output.repeat(len(G), 1, 1), G=G), dim=0)

            model_outputs.append(model_output)

        return torch.mean(torch.stack(model_outputs), dim=0)
