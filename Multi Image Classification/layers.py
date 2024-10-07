from modules import SymmetricOperator

import torch
import torch.nn as nn


class SiameseLayer(nn.Module):
    def __init__(self, operator, layer_type="conv"):
        super().__init__()
        self.operator = operator
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "conv":
            x = self.operator(x)
        elif self.layer_type == "linear":
            x = self.operator(x)
        else:
            raise NotImplementedError
        return x


class DeepSetLayer(nn.Module):
    def __init__(self, operator1, operator2, k, layer_type="conv"):
        super().__init__()
        self.operator1 = operator1
        self.operator2 = operator2
        self.k = k
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "conv":
            x_shape = x.shape  # dim [B*k, c, H, W]

            # siamese output
            x_siam = self.operator1(x)

            # aggregation output
            x = x.reshape(-1, self.k, x_shape[1], x_shape[2], x_shape[3])
            x_sum = torch.sum(x, dim=1, keepdim=True)
            x_agg = x_sum - x  # dim [batch_size, k, c_in, H, W]
            x_agg_shape = x_agg.shape
            x_agg = x_agg.reshape(x_agg_shape[0] * x_agg_shape[1], x_agg_shape[2], x_agg_shape[3],
                                  x_agg_shape[4]).contiguous()
            x_agg = self.operator2(x_agg)

            x = x_siam + x_agg  # dim [B*k, c, H, W]

        elif self.layer_type == "linear":
            x_shape = x.shape  # dim [B*k, -1]

            # siamese output
            x_siam = self.operator1(x)
            x_siam_shape = x_siam.shape  # dim [B*k, -1]
            x_siam = x_siam.reshape(-1, self.k, x_siam_shape[1])

            # aggregation output
            x = x.reshape(-1, self.k, x_shape[1])  # dim [B, k, -1]
            x_sum = torch.sum(x, dim=1, keepdim=True)  # dim [batch_size, 1, -1]
            x_agg = x_sum - x  # dim [batch_size, k, -1]
            x_agg_shape = x_agg.shape
            x_agg = x_agg.reshape(x_agg_shape[0] * x_agg_shape[1], -1).contiguous()
            x_agg = self.operator2(x_agg).reshape(x_agg_shape[0], x_agg_shape[1], -1).contiguous()

            x = x_siam + x_agg  # dim [B, k, -1]
        else:
            raise NotImplementedError
        return x


class DSSLayer(nn.Module):
    def __init__(self, operator1, operator2, k, layer_type="conv"):
        super().__init__()
        self.operator1 = operator1
        self.operator2 = operator2
        self.k = k
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "conv":
            # siamese output
            x_siam = self.operator1(x)

            x_sum = torch.sum(x, dim=1, keepdim=True)
            x_agg = x_sum - x  # dim [batch_size, k, c_in, H, W]
            x_agg = self.operator2(x_agg)

            x = x_siam + x_agg  # dim [B, k, c, H, W]

        elif self.layer_type == "conv_block":
            # x dim [B, k, c, H, W]
            x_shape = x.shape
            #  siamese output
            x_siam = x.reshape(x_shape[0]*x_shape[1], x_shape[2], x_shape[3], x_shape[4])
            x_siam = self.operator1(x_siam)
            x_siam_shape = x_siam.shape  # dim [B*k, c1, H1, W1]
            x_siam = x_siam.reshape(x_shape[0], x_shape[1], x_siam_shape[1], x_siam_shape[2], x_siam_shape[3])

            x_sum = torch.sum(x, dim=1, keepdim=True)
            x_agg = x_sum - x  # dim [batch_size, k, c_in, H, W]
            x_agg = x_agg.reshape(x_shape[0] * x_shape[1], x_shape[2], x_shape[3], x_shape[4])
            x_agg = self.operator2(x_agg)
            x_agg_shape = x_agg.shape  # dim [B*k, c1, H1, W1]
            x_agg = x_agg.reshape(x_shape[0], x_shape[1], x_agg_shape[1], x_agg_shape[2], x_agg_shape[3])

            x = x_siam + x_agg  # dim [B, k, c, H, W]

        elif self.layer_type == "linear":
            x_shape = x.shape  # dim [B, k, -1]
            x = x.reshape(-1, x_shape[2])

            # siamese output
            x_siam = self.operator1(x)
            x_siam_shape = x_siam.shape  # dim [B, k, -1]
            x_siam = x_siam.reshape(-1, self.k, x_siam_shape[1])

            # aggregation output
            x = x.reshape(-1, x_shape[1], x_shape[2])  # dim [B, k, -1]
            x_sum = torch.sum(x, dim=1, keepdim=True)  # dim [batch_size, 1, -1]
            x_agg = x_sum - x  # dim [batch_size, k, -1]
            x_agg_shape = x_agg.shape
            x_agg = x_agg.reshape(x_agg_shape[0] * x_agg_shape[1], -1).contiguous()
            x_agg = self.operator2(x_agg).reshape(x_agg_shape[0], x_agg_shape[1], -1).contiguous()

            x = x_siam + x_agg  # dim [B, k, -1]
        else:
            raise NotImplementedError
        return x


class GEMILayer_V1(nn.Module):
    # this implementation only works when all inputs need the same kind of symmetries, e.g. rot90
    def __init__(self, operator1, operator2, k, layer_type="conv", symmetry_group="rot90"):
        super().__init__()
        self.operator1 = operator1  # equivariant operator
        self.operator2 = operator2  # invariant operator, symmetry operation needs to be applied in the forward
        self.k = k
        self.layer_type = layer_type
        self.symmetry_group = symmetry_group
        self.symmetric_operator = SymmetricOperator(symmetry_group=symmetry_group)

    def forward(self, x):
        if self.layer_type == "conv_block":
            # x dim [B, k, c, H, W]
            x_shape = x.shape
            #  siamese output
            x_siam = x.reshape(x_shape[0]*x_shape[1], x_shape[2], x_shape[3], x_shape[4])
            x_siam = self.operator1(x_siam)
            x_siam_shape = x_siam.shape  # dim [B*k, c1, H1, W1]
            x_siam = x_siam.reshape(x_shape[0], x_shape[1], x_siam_shape[1], x_siam_shape[2], x_siam_shape[3])

            x_agg = x  # dim [batch_size, k, c_in, H, W]
            x_agg = x_agg.reshape(x_shape[0] * x_shape[1], x_shape[2], x_shape[3], x_shape[4])
            x_agg = self.operator2(x_agg)  # symmetric and invariant
            x_agg_shape = x_agg.shape  # dim [B*k, c1, H1, W1]
            x_agg = x_agg.reshape(x_shape[0], x_shape[1], x_agg_shape[1], x_agg_shape[2], x_agg_shape[3])  # dim [B, k, c1, H1, W1]
            x_agg_sum = torch.sum(x_agg, dim=(1,), keepdim=True)
            x_agg_sum = x_agg_sum - x_agg
            x_agg_sum = self.symmetric_operator(x_agg_sum)

            x = x_siam + x_agg_sum  # dim [B, k, c, H, W]

        elif self.layer_type == "linear":
            x_shape = x.shape  # dim [B, k, -1]
            x = x.reshape(-1, x_shape[2])

            # siamese output
            x_siam = self.operator1(x)
            x_siam_shape = x_siam.shape  # dim [B, k, -1]
            x_siam = x_siam.reshape(-1, self.k, x_siam_shape[1])

            # aggregation output
            x = x.reshape(-1, x_shape[1], x_shape[2])  # dim [B, k, -1]
            x_agg = x  # dim [batch_size, k, -1]
            x_agg_shape = x_agg.shape
            x_agg = x_agg.reshape(x_agg_shape[0] * x_agg_shape[1], -1).contiguous()
            x_agg = self.operator2(x_agg).reshape(x_agg_shape[0], x_agg_shape[1], -1).contiguous()
            x_agg_sum = torch.sum(x_agg, dim=(1,), keepdim=True)
            x_agg_sum = x_agg_sum - x_agg

            x = x_siam + x_agg_sum  # dim [B, k, -1]
        else:
            raise NotImplementedError
        return x


class GEMILayer_V2(nn.Module):
    # extended implementation where multiple symmetries are taken care of and is faster because of parallel processing
    def __init__(self, multi_equi_operator, multi_inv_operator, k, layer_type="conv", list_symmetry_groups=["rot90", "hflip"]):
        super().__init__()
        self.multi_equi_operator = multi_equi_operator  # equivariant operators list
        self.multi_inv_operator = multi_inv_operator  # invariant operators list, symmetry operation needs to be applied in the forward
        self.k = k
        self.layer_type = layer_type
        self.list_symmetry_groups = list_symmetry_groups
        # assert k == len(list_symmetry_groups), f"k ({k}) must be equal to len(list_symmetry_groups)"
        self.list_symmetric_operators = [SymmetricOperator(symmetry_group=symmetry_group) for symmetry_group
                                         in list_symmetry_groups]

    def forward(self, x):
        if self.layer_type == "conv_block":
            # x dim [B, k, c, H, W]
            # siam operators
            x_shape = x.shape
            x_siam = self.multi_equi_operator(x)  # dim [B, k, c, H1, W1]

            # fusion operators
            x_agg = self.multi_inv_operator(x)  # dim [B, k, c, H1, W1]
            x_agg_sum = torch.sum(x_agg, dim=(1,), keepdim=True)
            x_agg_sum = x_agg_sum - x_agg  # dim [B, k, c, H1, W1]
            x_agg_sum_symm = []
            for i in range(x_shape[1]):
                x_agg_sum_symm.append(self.list_symmetric_operators[i](x_agg_sum[:, i, :, :, :]))
            x_agg_sum_symm = torch.stack(x_agg_sum_symm, dim=1)  # dim [B, k, c, H1, W1]

            x = x_siam + x_agg_sum_symm  # dim [B, k, c, H, W]

        elif self.layer_type == "linear":
            x_shape = x.shape  # dim [B, k, -1]
            x = x.reshape(-1, x_shape[2])

            # siamese output
            x_siam = self.operator1(x)
            x_siam_shape = x_siam.shape  # dim [B, k, -1]
            x_siam = x_siam.reshape(-1, self.k, x_siam_shape[1])

            # aggregation output
            x = x.reshape(-1, x_shape[1], x_shape[2])  # dim [B, k, -1]
            x_sum = torch.sum(x, dim=1, keepdim=True)  # dim [batch_size, 1, -1]
            x_agg = x_sum - x  # dim [batch_size, k, -1]
            x_agg_shape = x_agg.shape
            x_agg = x_agg.reshape(x_agg_shape[0] * x_agg_shape[1], -1).contiguous()
            x_agg = self.operator2(x_agg).reshape(x_agg_shape[0], x_agg_shape[1], -1).contiguous()

            x = x_siam + x_agg  # dim [B, k, -1]
        else:
            raise NotImplementedError
        return x


class GEMILayer(nn.Module):
    # extended implementation where multiple symmetries are taken care of (but the group actions are applied sequentially,
    # hence not used in final use). kept for future reference, e.g. to check the correctness of the faster version
    def __init__(self, operator1_list, operator2_list, k, layer_type="conv", list_symmetry_groups=["rot90", "hflip"]):
        super().__init__()
        self.operator1_list = operator1_list  # equivariant operators list
        self.operator2_list = operator2_list  # invariant operators list, symmetry operation needs to be applied in the forward
        self.k = k
        self.layer_type = layer_type
        self.list_symmetry_groups = list_symmetry_groups
        # assert k == len(list_symmetry_groups), f"k ({k}) must be equal to len(list_symmetry_groups)"
        self.list_symmetric_operators = [SymmetricOperator(symmetry_group=symmetry_group) for symmetry_group
                                         in list_symmetry_groups]

    def forward(self, x):
        if self.layer_type == "conv_block":
            # x dim [B, k, c, H, W]
            x_shape = x.shape
            x_siam = []
            for i in range(x_shape[1]):
                x_siam.append(self.operator1_list[i](x[:, i, :, :, :]))
            x_siam = torch.stack(x_siam, dim=1)  # dim [B, k, c, H1, W1]

            x_agg = []
            for i in range(x_shape[1]):
                x_agg.append(self.operator2_list[i](x[:, i, :, :, :]))
            x_agg = torch.stack(x_agg, dim=1)  # dim [B, k, c, H1, W1]
            x_agg_sum = torch.sum(x_agg, dim=(1,), keepdim=True)
            x_agg_sum = x_agg_sum - x_agg  # dim [B, k, c, H1, W1]
            x_agg_sum_symm = []
            for i in range(x_shape[1]):
                x_agg_sum_symm.append(self.list_symmetric_operators[i](x_agg_sum[:, i, :, :, :]))
            x_agg_sum_symm = torch.stack(x_agg_sum_symm, dim=1)  # dim [B, k, c, H1, W1]

            x = x_siam + x_agg_sum_symm  # dim [B, k, c, H, W]

        elif self.layer_type == "linear":
            x_shape = x.shape  # dim [B, k, -1]
            x = x.reshape(-1, x_shape[2])

            # siamese output
            x_siam = self.operator1(x)
            x_siam_shape = x_siam.shape  # dim [B, k, -1]
            x_siam = x_siam.reshape(-1, self.k, x_siam_shape[1])

            # aggregation output
            x = x.reshape(-1, x_shape[1], x_shape[2])  # dim [B, k, -1]
            x_sum = torch.sum(x, dim=1, keepdim=True)  # dim [batch_size, 1, -1]
            x_agg = x_sum - x  # dim [batch_size, k, -1]
            x_agg_shape = x_agg.shape
            x_agg = x_agg.reshape(x_agg_shape[0] * x_agg_shape[1], -1).contiguous()
            x_agg = self.operator2(x_agg).reshape(x_agg_shape[0], x_agg_shape[1], -1).contiguous()

            x = x_siam + x_agg  # dim [B, k, -1]
        else:
            raise NotImplementedError
        return x
