from torchvision.transforms.functional import hflip, vflip
import torch.nn.functional as F

import torch
import torch.nn as nn


class kpool(nn.Module):
    def __init__(self, pool):
        super().__init__()
        self.pool = pool

    def forward(self, x):
        # dim [B, k, c, H, W]
        x_shape = x.shape
        x = x.reshape(x_shape[0]*x_shape[1], x_shape[2], x_shape[3], x_shape[4])  # dim [B*k, c, H, W]
        x_pool = self.pool(x)
        x_pool_shape = x_pool.shape  # dim [B*k, c, H1, W1]
        x_pool = x_pool.reshape(x_shape[0], x_shape[1], x_shape[2], x_pool_shape[2], x_pool_shape[3])
        return x_pool


class SymmetricOperator(nn.Module):
    '''
    Outputs symmetric results out of any model output (used in the fusion layers of multimodal equivariant models)
    '''
    def __init__(self, symmetry_group="rot90"):
        super().__init__()
        self.symmetry_group = symmetry_group

    def symmetric_output(self, x):
        if self.symmetry_group == "id":
            out = torch.stack([x])
        elif self.symmetry_group == "rot90":
            hidden_output_list = []  # x dim [batch_size, c, d, d]
            for i in range(4):
                hidden_output_list.append(torch.rot90(x, k=i, dims=[-2, -1]))
            out = torch.stack(hidden_output_list)
        elif self.symmetry_group == "hflip":
            hidden_output_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(hflip(x))  # hflips the last two dimension of x
            out = torch.stack(hidden_output_list)
        elif self.symmetry_group == "vflip":
            hidden_output_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(vflip(x))  # hflips the last two dimension of x
            out = torch.stack(hidden_output_list)
        elif self.symmetry_group == "rot90_hflip":
            input_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                input_list.append(hflip(x))  # hflips the last two dimension of x
            hidden_output_list = []
            for i in range(2):
                for j in range(4):
                    hidden_output_list.append(torch.rot90(input_list[i], k=j, dims=[-2, -1]))
            out = torch.stack(hidden_output_list)
        elif self.symmetry_group == "rot90_vflip":
            input_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                input_list.append(vflip(x))  # vflips the last two dimension of x
            hidden_output_list = []
            for i in range(2):
                for j in range(4):
                    hidden_output_list.append(torch.rot90(input_list[i], k=j, dims=[-2, -1]))
            out = torch.stack(hidden_output_list)
        else:
            raise NotImplementedError

        # dim out [G, B, k, c, H, W]
        out = torch.mean(out, dim=(0,), keepdim=False)
        return out

    def forward(self, x):
        # x dim [B, k, c, H, W]
        x = self.symmetric_output(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(args.in_channels, args.conv_channels, 5)
        self.conv2 = nn.Conv2d(args.conv_channels, args.conv_channels, 5)
        self.conv3 = nn.Conv2d(args.conv_channels, args.conv_channels, 3)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv layer  # dim [B, c, H, W]
        x = self.pool(F.relu(self.conv2(x)))  # Conv layer  # dim [B, c, H, W]
        x = self.pool(F.relu(self.conv3(x)))  # Conv layer  # dim [B, c, H, W]
        return x


class EquiOperator(nn.Module):
    '''
    :returns equivariant operator for any operator such as Conv, Linear, Deepset, etc.
    '''
    def __init__(self, operator, symmetry_group="rot90", operator_type="conv"):
        super().__init__()
        self.operator = operator
        self.symmetry_group = symmetry_group
        self.operator_type = operator_type
        self.pool = nn.MaxPool2d(2, 2)

    def group_transformed_input(self, x):
        '''
        :param x: dim [batch_size, k, c, H, W]
        :return: dim [group_size, batch_size, k, c, H, W]
        '''
        if self.symmetry_group == "id":
            return torch.stack([x])
        elif self.symmetry_group == "rot90":
            hidden_output_list = []  # x dim [batch_size, c, d, d]
            for i in range(4):
                hidden_output_list.append(torch.rot90(x, k=i, dims=[-2, -1]))
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "hflip":
            hidden_output_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(hflip(x))  # hflips the last two dimension of x
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "vflip":
            hidden_output_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(vflip(x))  # hflips the last two dimension of x
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "rot90_hflip":
            input_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                input_list.append(hflip(x))  # hflips the last two dimension of x
            hidden_output_list = []
            for i in range(2):
                for j in range(4):
                    hidden_output_list.append(torch.rot90(input_list[i], k=j, dims=[-2, -1]))
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "rot90_vflip":
            input_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                input_list.append(vflip(x))  # vflips the last two dimension of x
            hidden_output_list = []
            for i in range(2):
                for j in range(4):
                    hidden_output_list.append(torch.rot90(input_list[i], k=j, dims=[-2, -1]))
            return torch.stack(hidden_output_list)
        else:
            raise NotImplementedError

    def inverse_group_transformed_hidden_output(self, x_list):
        if self.symmetry_group == "id":
            return x_list
        elif self.symmetry_group == "rot90":
            hidden_output_list = []  # x dim [batch_size, d, d, c]
            for i in range(4):
                hidden_output_list.append(torch.rot90(x_list[i], k=4-i, dims=[-2, -1]))
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "hflip":
            hidden_output_list = [x_list[0]]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(hflip(x_list[i+1]))  # hflips the last two dimension of x
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "vflip":
            hidden_output_list = [x_list[0]]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(vflip(x_list[i+1]))  # hflips the last two dimension of x
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "rot90_hflip":
            for i in range(len(x_list)):
                x_list[i] = torch.rot90(x_list[i], k=4-(i % 4), dims=[-2, -1])  # x dim [batch_size, c, d, d]

            for i in range(len(x_list)):
                if i > 3:
                    x_list[i] = hflip(x_list[i])  # hflips the last two dimension of x
            return torch.stack(x_list)
        elif self.symmetry_group == "rot90_vflip":
            for i in range(len(x_list)):
                x_list[i] = torch.rot90(x_list[i], k=4-(i % 4), dims=[-2, -1])  # x dim [batch_size, c, d, d]

            for i in range(len(x_list)):
                if i > 3:
                    x_list[i] = vflip(x_list[i])  # hflips the last two dimension of x
            return torch.stack(x_list)
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.operator_type == "conv":
            # x dim [B, k, c, H, W]
            x_group = self.group_transformed_input(x)  # dim [G, B, k, c, H, W]
            x_group_shape = x_group.shape
            x_group = x_group.reshape(x_group_shape[0]*x_group_shape[1]*x_group_shape[2], x_group_shape[3],
                                      x_group_shape[4], x_group_shape[5])  # dim [G*B*k, c, H, W]
            out = self.pool(F.relu(self.operator(x_group)))  # dim [G*B*k, c, H, W]
            out_shape = out.shape  # dim [G*B*k, c, H, W]
            out = out.reshape(x_group_shape[0], x_group_shape[1], x_group_shape[2], out_shape[1],
                                      out_shape[2], out_shape[3])  # dim [G, B, k, c, H, W]
            out = self.inverse_group_transformed_hidden_output(out)  # dim [G, B, k, c, H, W]
            out = torch.mean(out, dim=(0,), keepdim=False)
        elif self.operator_type == "deepset":
            # x dim [B, k, c, H, W]
            x_group = self.group_transformed_input(x)  # dim [G, B, k, c, H, W]
            x_group_shape = x_group.shape
            x_group = x_group.reshape(x_group_shape[0] * x_group_shape[1], x_group_shape[2], x_group_shape[3],
                                      x_group_shape[4], x_group_shape[5])  # dim [G*B, k, c, H, W]
            out = self.operator(x_group)  # dim [G*B, -1]  # classification output
            out = out.reshape(x_group_shape[0], x_group_shape[1], -1)  # dim [G, B, -1]
            out = torch.mean(out, dim=(0,), keepdim=False)
        elif self.operator_type == "conv_block":
            # x dim [B, c, H, W]
            x_group = self.group_transformed_input(x)  # dim [G, B, c, H, W]
            x_group_shape = x_group.shape
            x_group = x_group.reshape(x_group_shape[0] * x_group_shape[1], x_group_shape[2],
                                      x_group_shape[3], x_group_shape[4])  # dim [G*B*k, c, H, W]
            out = self.operator(x_group)  # dim [G*B, c, H, W]
            out_shape = out.shape  # dim [G*B, c, H, W]
            out = out.reshape(x_group_shape[0], x_group_shape[1], out_shape[1],
                              out_shape[2], out_shape[3])  # dim [G, B, c, H, W]
            out = self.inverse_group_transformed_hidden_output(out)  # dim [G, B, c, H, W]
            out = torch.mean(out, dim=(0,), keepdim=False)
        else:
            raise NotImplementedError
        return out


class MultiEquiOperator(nn.Module):
    '''
    :returns equivariant operator for any operator for a list of symmetry groups
    '''
    def __init__(self, operator, list_symmetry_groups=["rot90", "hflip"], operator_type="conv"):
        super().__init__()
        self.operator = operator
        self.list_symmetry_groups = list_symmetry_groups
        self.operator_type = operator_type
        self.pool = nn.MaxPool2d(2, 2)

    def group_transformed_input(self, x, symmetry_group):
        '''
        :param x: dim [batch_size, k, c, H, W]
        :return: dim [group_size, batch_size, k, c, H, W]
        '''
        self.symmetry_group = symmetry_group
        if self.symmetry_group == "id":
            return torch.stack([x])
        elif self.symmetry_group == "rot90":
            hidden_output_list = []  # x dim [batch_size, c, d, d]
            for i in range(4):
                hidden_output_list.append(torch.rot90(x, k=i, dims=[-2, -1]))
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "hflip":
            hidden_output_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(hflip(x))  # hflips the last two dimension of x
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "vflip":
            hidden_output_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(vflip(x))  # hflips the last two dimension of x
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "rot90_hflip":
            input_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                input_list.append(hflip(x))  # hflips the last two dimension of x
            hidden_output_list = []
            for i in range(2):
                for j in range(4):
                    hidden_output_list.append(torch.rot90(input_list[i], k=j, dims=[-2, -1]))
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "rot90_vflip":
            input_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                input_list.append(vflip(x))  # vflips the last two dimension of x
            hidden_output_list = []
            for i in range(2):
                for j in range(4):
                    hidden_output_list.append(torch.rot90(input_list[i], k=j, dims=[-2, -1]))
            return torch.stack(hidden_output_list)
        else:
            raise NotImplementedError

    def inverse_group_transformed_hidden_output(self, x_list, symmetry_group):
        self.symmetry_group = symmetry_group
        if self.symmetry_group == "id":
            return x_list
        elif self.symmetry_group == "rot90":
            hidden_output_list = []  # x dim [batch_size, d, d, c]
            for i in range(4):
                hidden_output_list.append(torch.rot90(x_list[i], k=4-i, dims=[-2, -1]))
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "hflip":
            hidden_output_list = [x_list[0]]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(hflip(x_list[i+1]))  # hflips the last two dimension of x
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "vflip":
            hidden_output_list = [x_list[0]]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(vflip(x_list[i+1]))  # hflips the last two dimension of x
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "rot90_hflip":
            for i in range(len(x_list)):
                x_list[i] = torch.rot90(x_list[i], k=4-(i % 4), dims=[-2, -1])  # x dim [batch_size, c, d, d]

            for i in range(len(x_list)):
                if i > 3:
                    x_list[i] = hflip(x_list[i])  # hflips the last two dimension of x
            return torch.stack(x_list)
        elif self.symmetry_group == "rot90_vflip":
            for i in range(len(x_list)):
                x_list[i] = torch.rot90(x_list[i], k=4-(i % 4), dims=[-2, -1])  # x dim [batch_size, c, d, d]

            for i in range(len(x_list)):
                if i > 3:
                    x_list[i] = vflip(x_list[i])  # hflips the last two dimension of x
            return torch.stack(x_list)
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.operator_type == "conv_block":
            x_shape = x.shape  # x dim [B, k, c, H, W]
            k = x_shape[1]
            x_group = []
            group_sizes = []
            for i in range(k):
                # a list of different dimensions of transformed input x
                transformed_x = self.group_transformed_input(x[:, i, :, :, :], symmetry_group=self.list_symmetry_groups[i])  # dim [G_i, B, c, H, W]
                group_sizes.append(len(transformed_x))
                x_group.append(transformed_x)
            x_group = torch.cat(x_group, dim=0)  # dim [\sum_{i=1 to k} G_i, B, c, H, W]
            x_group_shape = x_group.shape  # dim [\sum_{i=1 to k} G_i, B, c, H, W]
            x_group = x_group.reshape(x_group_shape[0] * x_group_shape[1], x_group_shape[2],
                                      x_group_shape[3], x_group_shape[4])  # dim [(\sum_{i=1 to k} G_i)*B, c, H, W]
            out = self.operator(x_group)  # dim [G*B, c, H, W]
            out_shape = out.shape  # dim [(\sum_{i=1 to k} G_i)*B, c, H, W]
            out = out.reshape(x_group_shape[0], x_group_shape[1], out_shape[1],
                              out_shape[2], out_shape[3])  # dim [(\sum_{i=1 to k} G_i), B, c, H, W]
            out_group = []
            group_sizes_sum = 0
            for i in range(k):
                inv_trans_out = self.inverse_group_transformed_hidden_output(out[group_sizes_sum : group_sizes_sum + group_sizes[i]], symmetry_group=self.list_symmetry_groups[i])
                group_sizes_sum += group_sizes[i]
                out_group.append(torch.mean(inv_trans_out, dim=(0,), keepdim=False))
            out_group = torch.stack(out_group, dim=1)  # dim [B, k, c, H1, W1]
        else:
            # MultiEquiOperator only implemented for conv_block operators
            raise NotImplementedError
        return out_group


class InvOperator(nn.Module):
    '''
    :returns invariant operator operator for any operator such as Conv, Linear, Deepset, etc.
    '''
    def __init__(self, operator, symmetry_group="rot90", operator_type="conv"):
        super().__init__()
        self.operator = operator
        self.symmetry_group = symmetry_group
        self.operator_type = operator_type
        self.pool = nn.MaxPool2d(2, 2)

    def group_transformed_input(self, x):
        '''
        :param x: dim [batch_size, k, c, H, W]
        :return: dim [group_size, batch_size, k, c, H, W]
        '''
        if self.symmetry_group == "id":
            return torch.stack([x])
        elif self.symmetry_group == "rot90":
            hidden_output_list = []  # x dim [batch_size, c, d, d]
            for i in range(4):
                hidden_output_list.append(torch.rot90(x, k=i, dims=[-2, -1]))
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "hflip":
            hidden_output_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(hflip(x))  # hflips the last two dimension of x
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "vflip":
            hidden_output_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(vflip(x))  # hflips the last two dimension of x
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "rot90_hflip":
            input_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                input_list.append(hflip(x))  # hflips the last two dimension of x
            hidden_output_list = []
            for i in range(2):
                for j in range(4):
                    hidden_output_list.append(torch.rot90(input_list[i], k=j, dims=[-2, -1]))
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "rot90_vflip":
            input_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                input_list.append(vflip(x))  # vflips the last two dimension of x
            hidden_output_list = []
            for i in range(2):
                for j in range(4):
                    hidden_output_list.append(torch.rot90(input_list[i], k=j, dims=[-2, -1]))
            return torch.stack(hidden_output_list)
        else:
            raise NotImplementedError

    def inverse_group_transformed_hidden_output(self, x_list):
        if self.symmetry_group == "id":
            return x_list
        elif self.symmetry_group == "rot90":
            hidden_output_list = []  # x dim [batch_size, d, d, c]
            for i in range(4):
                hidden_output_list.append(torch.rot90(x_list[i], k=4-i, dims=[-2, -1]))
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "hflip":
            hidden_output_list = [x_list[0]]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(hflip(x_list[i+1]))  # hflips the last two dimension of x
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "vflip":
            hidden_output_list = [x_list[0]]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(vflip(x_list[i+1]))  # hflips the last two dimension of x
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "rot90_hflip":
            for i in range(len(x_list)):
                x_list[i] = torch.rot90(x_list[i], k=4-(i % 4), dims=[-2, -1])  # x dim [batch_size, c, d, d]

            for i in range(len(x_list)):
                if i > 3:
                    x_list[i] = hflip(x_list[i])  # hflips the last two dimension of x
            return torch.stack(x_list)
        elif self.symmetry_group == "rot90_vflip":
            for i in range(len(x_list)):
                x_list[i] = torch.rot90(x_list[i], k=4-(i % 4), dims=[-2, -1])  # x dim [batch_size, c, d, d]

            for i in range(len(x_list)):
                if i > 3:
                    x_list[i] = vflip(x_list[i])  # hflips the last two dimension of x
            return torch.stack(x_list)
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.operator_type == "conv":
            # x dim [B, k, c, H, W]
            x_group = self.group_transformed_input(x)  # dim [G, B, k, c, H, W]
            x_group_shape = x_group.shape
            x_group = x_group.reshape(x_group_shape[0]*x_group_shape[1]*x_group_shape[2], x_group_shape[3],
                                      x_group_shape[4], x_group_shape[5])  # dim [G*B*k, c, H, W]
            out = self.pool(F.relu(self.operator(x_group)))  # dim [G*B*k, c, H, W]
            out_shape = out.shape  # dim [G*B*k, c, H, W]
            out = out.reshape(x_group_shape[0], x_group_shape[1], x_group_shape[2], out_shape[1],
                                      out_shape[2], out_shape[3])  # dim [G, B, k, c, H, W]
            out = torch.mean(out, dim=(0,), keepdim=False)
        elif self.operator_type == "deepset":
            # x dim [B, k, c, H, W]
            x_group = self.group_transformed_input(x)  # dim [G, B, k, c, H, W]
            x_group_shape = x_group.shape
            x_group = x_group.reshape(x_group_shape[0] * x_group_shape[1], x_group_shape[2], x_group_shape[3],
                                      x_group_shape[4], x_group_shape[5])  # dim [G*B, k, c, H, W]
            out = self.operator(x_group)  # dim [G*B, -1]  # classification output
            out = out.reshape(x_group_shape[0], x_group_shape[1], -1)  # dim [G, B, -1]
            out = torch.mean(out, dim=(0,), keepdim=False)
        elif self.operator_type == "conv_block":
            # x dim [B, c, H, W]
            x_group = self.group_transformed_input(x)  # dim [G, B, c, H, W]
            x_group_shape = x_group.shape
            x_group = x_group.reshape(x_group_shape[0] * x_group_shape[1], x_group_shape[2],
                                      x_group_shape[3], x_group_shape[4])  # dim [G*B*k, c, H, W]
            out = self.operator(x_group)  # dim [G*B, c, H, W]
            out_shape = out.shape  # dim [G*B, c, H, W]
            out = out.reshape(x_group_shape[0], x_group_shape[1], out_shape[1],
                              out_shape[2], out_shape[3])  # dim [G, B, c, H, W]
            out = torch.mean(out, dim=(0,), keepdim=False)
        else:
            raise NotImplementedError
        return out


class MultiInvOperator(nn.Module):
    '''
    :returns equivariant operator for any operator for a list of symmetry groups
    '''
    def __init__(self, operator, list_symmetry_groups=["rot90", "hflip"], operator_type="conv"):
        super().__init__()
        self.operator = operator
        self.list_symmetry_groups = list_symmetry_groups
        self.operator_type = operator_type
        self.pool = nn.MaxPool2d(2, 2)

    def group_transformed_input(self, x, symmetry_group):
        '''
        :param x: dim [batch_size, k, c, H, W]
        :return: dim [group_size, batch_size, k, c, H, W]
        '''
        self.symmetry_group = symmetry_group
        if self.symmetry_group == "id":
            return torch.stack([x])
        elif self.symmetry_group == "rot90":
            hidden_output_list = []  # x dim [batch_size, c, d, d]
            for i in range(4):
                hidden_output_list.append(torch.rot90(x, k=i, dims=[-2, -1]))
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "hflip":
            hidden_output_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(hflip(x))  # hflips the last two dimension of x
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "vflip":
            hidden_output_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(vflip(x))  # hflips the last two dimension of x
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "rot90_hflip":
            input_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                input_list.append(hflip(x))  # hflips the last two dimension of x
            hidden_output_list = []
            for i in range(2):
                for j in range(4):
                    hidden_output_list.append(torch.rot90(input_list[i], k=j, dims=[-2, -1]))
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "rot90_vflip":
            input_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                input_list.append(vflip(x))  # vflips the last two dimension of x
            hidden_output_list = []
            for i in range(2):
                for j in range(4):
                    hidden_output_list.append(torch.rot90(input_list[i], k=j, dims=[-2, -1]))
            return torch.stack(hidden_output_list)
        else:
            raise NotImplementedError

    def inverse_group_transformed_hidden_output(self, x_list, symmetry_group):
        self.symmetry_group = symmetry_group
        if self.symmetry_group == "id":
            return x_list
        elif self.symmetry_group == "rot90":
            hidden_output_list = []  # x dim [batch_size, d, d, c]
            for i in range(4):
                hidden_output_list.append(torch.rot90(x_list[i], k=4-i, dims=[-2, -1]))
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "hflip":
            hidden_output_list = [x_list[0]]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(hflip(x_list[i+1]))  # hflips the last two dimension of x
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "vflip":
            hidden_output_list = [x_list[0]]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(vflip(x_list[i+1]))  # hflips the last two dimension of x
            return torch.stack(hidden_output_list)
        elif self.symmetry_group == "rot90_hflip":
            for i in range(len(x_list)):
                x_list[i] = torch.rot90(x_list[i], k=4-(i % 4), dims=[-2, -1])  # x dim [batch_size, c, d, d]

            for i in range(len(x_list)):
                if i > 3:
                    x_list[i] = hflip(x_list[i])  # hflips the last two dimension of x
            return torch.stack(x_list)
        elif self.symmetry_group == "rot90_vflip":
            for i in range(len(x_list)):
                x_list[i] = torch.rot90(x_list[i], k=4-(i % 4), dims=[-2, -1])  # x dim [batch_size, c, d, d]

            for i in range(len(x_list)):
                if i > 3:
                    x_list[i] = vflip(x_list[i])  # hflips the last two dimension of x
            return torch.stack(x_list)
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.operator_type == "conv_block":
            x_shape = x.shape  # x dim [B, k, c, H, W]
            k = x_shape[1]
            x_group = []
            group_sizes = []
            for i in range(k):
                # a list of different dimensions of transformed input x
                transformed_x = self.group_transformed_input(x[:, i, :, :, :], symmetry_group=self.list_symmetry_groups[i])  # dim [G_i, B, c, H, W]
                group_sizes.append(len(transformed_x))
                x_group.append(transformed_x)
            x_group = torch.cat(x_group, dim=0)  # dim [\sum_{i=1 to k} G_i, B, c, H, W]
            x_group_shape = x_group.shape  # dim [\sum_{i=1 to k} G_i, B, c, H, W]
            x_group = x_group.reshape(x_group_shape[0] * x_group_shape[1], x_group_shape[2],
                                      x_group_shape[3], x_group_shape[4])  # dim [(\sum_{i=1 to k} G_i)*B, c, H, W]
            out = self.operator(x_group)  # dim [G*B, c, H, W]
            out_shape = out.shape  # dim [(\sum_{i=1 to k} G_i)*B, c, H, W]
            out = out.reshape(x_group_shape[0], x_group_shape[1], out_shape[1],
                              out_shape[2], out_shape[3])  # dim [(\sum_{i=1 to k} G_i), B, c, H, W]
            out_group = []
            group_sizes_sum = 0
            for i in range(k):
                trans_out = out[group_sizes_sum : group_sizes_sum + group_sizes[i]]
                group_sizes_sum += group_sizes[i]
                out_group.append(torch.mean(trans_out, dim=(0,), keepdim=False))
            out_group = torch.stack(out_group, dim=1)  # dim [B, k, c, H1, W1]
        else:
            # MultiEquiOperator only implemented for conv_block operators
            raise NotImplementedError
        return out_group
