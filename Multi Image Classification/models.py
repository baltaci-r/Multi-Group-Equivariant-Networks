from layers import SiameseLayer, DeepSetLayer, DSSLayer, GEMILayer, GEMILayer_V1, GEMILayer_V2
from modules import kpool, ConvBlock, EquiOperator, InvOperator, MultiEquiOperator, MultiInvOperator
from modules import SymmetricOperator
from torchvision.transforms.functional import hflip, vflip

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

group_size = {"rot90": 4, "hflip": 2, "id": 1}


class ConvNet(nn.Module):
    # simply concatenates the k images and uses a traditional convolutional network
    # hence no permutation equivariance and no other equivariance
    def __init__(self, args, k, num_classes, input_size, type, logging, symmetry_group=""):
        super().__init__()
        self.conv_block = ConvBlock(args)
        if input_size == 28:
            self.fc1 = nn.Linear(args.conv_channels * math.floor((7 * k - 3 - 2)/2), args.lin_channels)
        elif input_size == 32:
            self.fc1 = nn.Linear(args.conv_channels * math.floor((8 * k - 3 - 2)/2) * 5, args.lin_channels)
        else:
            raise NotImplementedError
        self.fc2 = nn.Linear(args.lin_channels, args.lin_channels)
        if type == 'sum':
            self.fc3 = nn.Linear(args.lin_channels, k * num_classes)
        elif type == 'classify':
            self.fc3 = nn.Linear(args.lin_channels, num_classes)
        else:
            self.fc3 = nn.Linear(args.lin_channels, 1)

    def forward(self, x):
        # x dim [B, k, c, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # dim [B, c, k, H, W]
        x_shape = x.shape
        x = x.reshape(x_shape[0], x_shape[1], x_shape[2] * x_shape[3], x_shape[4])  # dim [B, c, k*H, W]
        x = self.conv_block(x)  # dim [B, c, H1, W1]
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DeepSet(nn.Module):
    '''
    Has an extra conv layer than DeepSet, which helps with H=W=1 out of the convolution
    '''
    def __init__(self, args, k, num_classes, input_size, type, logging, symmetry_group=""):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv2d(args.in_channels, args.conv_channels, 5)
        self.conv1_fusion = nn.Conv2d(args.in_channels, args.conv_channels, 5)
        self.ds_layer1 = DeepSetLayer(operator1=self.conv1, operator2=self.conv1_fusion, k=k, layer_type="conv")
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(args.conv_channels, args.conv_channels, 5)
        self.conv2_fusion = nn.Conv2d(args.conv_channels, args.conv_channels, 5)
        self.ds_layer2 = DeepSetLayer(operator1=self.conv2, operator2=self.conv2_fusion, k=k, layer_type="conv")
        self.conv3 = nn.Conv2d(args.conv_channels, args.conv_channels, 3)
        self.conv3_fusion = nn.Conv2d(args.conv_channels, args.conv_channels, 3)
        self.ds_layer3 = DeepSetLayer(operator1=self.conv3, operator2=self.conv3_fusion, k=k, layer_type="conv")
        self.fc1 = nn.Linear(args.conv_channels, args.lin_channels)
        self.fc1_fusion = nn.Linear(args.conv_channels, args.lin_channels)
        self.ds_layer4 = DeepSetLayer(operator1=self.fc1, operator2=self.fc1_fusion, k=k, layer_type="linear")

        self.fc2 = nn.Linear(args.lin_channels, args.lin_channels)
        if type == 'sum':
            self.fc3 = nn.Linear(args.lin_channels, k * num_classes)
        elif type == 'classify':
            self.fc3 = nn.Linear(args.lin_channels, num_classes)
        else:
            self.fc3 = nn.Linear(args.lin_channels, 1)

    def forward(self, x):
        x_shape = x.shape  # dim [B, k, c, H, W]
        x = x.reshape(x_shape[0] * x_shape[1], x_shape[2], x_shape[3], x_shape[4])
        x = self.pool(F.relu(self.ds_layer1(x)))  # Siamese layer  # dim [B*k, c, H, W]
        x = self.pool(F.relu(self.ds_layer2(x)))  # # Siamese layer  # dim [B*k, c, H, W]
        x = self.pool(F.relu(self.ds_layer3(x)))  # # Siamese layer  # dim [B*k, c, H, W]
        x = torch.mean(x, dim=(-1, -2), keepdim=False)  # remove spatial dimensions
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.ds_layer4(x))  # dim [B*k, 196]  # Siamese layer
        x = x.reshape(x_shape[0], x_shape[1], -1)  # dim [B, k, 196]

        # following layers are not Siamese
        x = torch.sum(x, dim=1, keepdim=False)  # dim [B, 196]
        # x = torch.flatten(x, 1)  # dim [B, k*196]
        x = F.relu(self.fc2(x))  # dim [196, 64]
        x = self.fc3(x)
        return x


class DSS(nn.Module):
    '''
    Has an extra conv layer than DeepSet, which helps with H=W=1 out of the convolution
    '''
    def __init__(self, args, k, num_classes, input_size, type, logging, symmetry_group=""):
        super().__init__()
        self.k = k
        self.conv_block = ConvBlock(args)
        self.operator = EquiOperator(operator=self.conv_block, symmetry_group=symmetry_group,
                                     operator_type="conv_block")
        self.conv_block_fusion = ConvBlock(args)
        self.operator_fusion = EquiOperator(operator=self.conv_block_fusion, symmetry_group=symmetry_group,
                                     operator_type="conv_block")
        self.dss_layer1 = DSSLayer(operator1=self.operator, operator2=self.operator_fusion, k=self.k,
                                  layer_type="conv_block")
        self.fc1 = nn.Linear(args.conv_channels, args.lin_channels)
        self.fc1_fusion = nn.Linear(args.conv_channels, args.lin_channels)
        self.dss_layer2 = DSSLayer(operator1=self.fc1, operator2=self.fc1_fusion, k=k, layer_type="linear")
        self.fc2 = nn.Linear(args.lin_channels, args.lin_channels)
        if type == 'sum':
            self.fc3 = nn.Linear(args.lin_channels, k * num_classes)
        elif type == 'classify':
            self.fc3 = nn.Linear(args.lin_channels, num_classes)
        else:
            self.fc3 = nn.Linear(args.lin_channels, 1)

    def forward(self, x):
        x_shape = x.shape  # dim [B, k, c, H, W]
        x = x.reshape(x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4])
        x = self.dss_layer1(x)  # dim [B, k, c, H, W]
        x = torch.mean(x, dim=(-1, -2), keepdim=False)  # remove spatial dimensions
        x = torch.flatten(x, 2)  # flatten all dimensions except batch
        x = F.relu(self.dss_layer2(x))  # dim [B*k, 196]  # Siamese layer
        x = x.reshape(x_shape[0], x_shape[1], -1)  # dim [B, k, 196]

        # following layers are not Siamese
        x = torch.sum(x, dim=1, keepdim=False)  # dim [B, 196]
        x = F.relu(self.fc2(x))  # dim [196, 64]
        x = self.fc3(x)
        return x


class GEMINN(nn.Module):
    def __init__(self, args, k, num_classes, input_size, type, logging, list_symmetry_groups=""):
        super().__init__()
        self.k = k
        self.conv_block = ConvBlock(args)
        logging.info("*** Symmetries ***")
        for sym in list_symmetry_groups:
            logging.info(sym)
        logging.info("*"*30)
        self.multi_equi_operator = MultiEquiOperator(operator=self.conv_block,
                                                     list_symmetry_groups=list_symmetry_groups,
                                                     operator_type="conv_block")
        self.operator_list = [EquiOperator(operator=self.conv_block, symmetry_group=symmetry_group,
                                     operator_type="conv_block") for symmetry_group in list_symmetry_groups]
        self.conv_block_fusion = ConvBlock(args)
        self.multi_inv_operator = MultiInvOperator(operator=self.conv_block_fusion,
                                                     list_symmetry_groups=list_symmetry_groups,
                                                     operator_type="conv_block")
        self.geminn_layer1 = GEMILayer_V2(multi_equi_operator=self.multi_equi_operator,
                                          multi_inv_operator=self.multi_inv_operator, k=self.k,
                                       layer_type="conv_block", list_symmetry_groups=list_symmetry_groups)
        self.fc1 = nn.Linear(args.conv_channels, args.lin_channels)
        self.fc1_fusion = nn.Linear(args.conv_channels, args.lin_channels)
        self.geminn_layer2 = GEMILayer_V1(operator1=self.fc1, operator2=self.fc1_fusion, k=k, layer_type="linear")  # already invariant beyond the convolution layers
        self.fc2 = nn.Linear(args.lin_channels, args.lin_channels)
        if type == 'sum':
            self.fc3 = nn.Linear(args.lin_channels, k * num_classes)
        elif type == 'classify':
            self.fc3 = nn.Linear(args.lin_channels, num_classes)
        else:
            self.fc3 = nn.Linear(args.lin_channels, 1)

    def forward(self, x):
        x_shape = x.shape  # dim [B, k, c, H, W]
        x = x.reshape(x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4])
        x = self.geminn_layer1(x)  # dim [B, k, c, H, W]
        x = torch.mean(x, dim=(-1, -2), keepdim=False)  # remove spatial dimensions
        x = torch.flatten(x, 2)  # flatten all dimensions except batch
        x = F.relu(self.geminn_layer2(x))  # dim [B*k, 196]  # Siamese layer
        x = x.reshape(x_shape[0], x_shape[1], -1)  # dim [B, k, 196]

        # following layers are not Siamese
        x = torch.sum(x, dim=1, keepdim=False)  # dim [B, 196]
        x = F.relu(self.fc2(x))  # dim [196, 64]
        x = self.fc3(x)
        return x


class SiameseNet(nn.Module):
    # Not used in the experiments since Siamese nets do not have fusion layers, while the others do
    def __init__(self, args, k, num_classes, input_size, type, logging, symmetry_group=""):
        super().__init__()
        self.conv1 = nn.Conv2d(args.in_channels, args.conv_channels, 5)
        self.siam_layer1 = SiameseLayer(operator=self.conv1, layer_type="conv")
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(args.conv_channels, args.conv_channels, 5)
        self.siam_layer2 = SiameseLayer(operator=self.conv2, layer_type="conv")
        if input_size == 28:
            self.fc1 = nn.Linear(args.conv_channels * 4 * 4, args.lin_channels)
        elif input_size == 32:
            self.fc1 = nn.Linear(args.conv_channels * 5 * 5, args.lin_channels)
        else:
            raise NotImplementedError
        self.siam_layer3 = SiameseLayer(operator=self.fc1, layer_type="linear")

        self.fc2 = nn.Linear(args.lin_channels, args.lin_channels)
        if type == 'sum':
            self.fc3 = nn.Linear(args.lin_channels, k * num_classes)
        elif type == 'classify':
            self.fc3 = nn.Linear(args.lin_channels, num_classes)
        else:
            self.fc3 = nn.Linear(args.lin_channels, 1)

    def forward(self, x):
        x_shape = x.shape  # dim [B, k, c, H, W]
        x = x.reshape(x_shape[0] * x_shape[1], x_shape[2], x_shape[3], x_shape[4])
        x = self.pool(F.relu(self.siam_layer1(x)))  # Siamese layer
        x = self.pool(F.relu(self.siam_layer2(x)))  # # Siamese layer
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.siam_layer3(x))  # dim [B*k, 196]  # Siamese layer
        x = x.reshape(x_shape[0], x_shape[1], -1)  # dim [B, k, 196]

        # following layers are not Siamese
        x = torch.sum(x, dim=1, keepdim=False)  # dim [B, 196]
        x = F.relu(self.fc2(x))  # dim [196, 64]
        x = self.fc3(x)
        return x


class Gemini(nn.Module):
    # Not used in the experiments since Siamese nets do not have fusion layers, while the others do
    def __init__(self, args, k, num_classes, input_size, type, logging, list_symmetry_groups=[""]):
        super().__init__()
        self.args = args
        self.k = k
        self.fusion = args.fusion
        self.conv1 = nn.Conv2d(args.in_channels, args.conv_channels, 5)
        self.conv1_fusion = nn.Conv2d(args.in_channels, args.conv_channels, 5)
        self.siam_layer1 = SiameseLayer(operator=self.conv1, layer_type="conv")
        self.siam_layer1_fusion = SiameseLayer(operator=self.conv1_fusion, layer_type="conv")
        self.bn1 = nn.BatchNorm2d(self.args.conv_channels)
        self.bn1_fusion = nn.BatchNorm2d(self.args.conv_channels)

        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(args.conv_channels, args.conv_channels, 5)
        self.conv2_fusion = nn.Conv2d(args.conv_channels, args.conv_channels, 5)
        self.siam_layer2 = SiameseLayer(operator=self.conv2, layer_type="conv")
        self.siam_layer2_fusion = SiameseLayer(operator=self.conv2_fusion, layer_type="conv")
        self.bn2 = nn.BatchNorm2d(self.args.conv_channels)
        self.bn2_fusion = nn.BatchNorm2d(self.args.conv_channels)

        self.conv64 = nn.Conv2d(args.conv_channels, args.conv_channels, 5)  # used when input size is 64
        self.conv64_fusion = nn.Conv2d(args.conv_channels, args.conv_channels, 5)
        self.siam_layer64 = SiameseLayer(operator=self.conv2, layer_type="conv")
        self.siam_layer64_fusion = SiameseLayer(operator=self.conv2_fusion, layer_type="conv")
        self.bn64 = nn.BatchNorm2d(self.args.conv_channels)
        self.bn64_fusion = nn.BatchNorm2d(self.args.conv_channels)

        if input_size == 28:
            self.fc1 = nn.Linear(args.conv_channels * 4 * 4, args.lin_channels)
            self.fc1_fusion = nn.Linear(args.conv_channels * 4 * 4, args.lin_channels)
        elif input_size == 32:
            self.fc1 = nn.Linear(args.conv_channels * 5 * 5, args.lin_channels)
            self.fc1_fusion = nn.Linear(args.conv_channels * 5 * 5, args.lin_channels)
        elif input_size == 64:
            self.fc1 = nn.Linear(args.conv_channels * 4 * 4, args.lin_channels)
            self.fc1_fusion = nn.Linear(args.conv_channels * 4 * 4, args.lin_channels)
        else:
            raise NotImplementedError
        self.siam_layer3 = SiameseLayer(operator=self.fc1, layer_type="linear")
        self.siam_layer3_fusion = SiameseLayer(operator=self.fc1_fusion, layer_type="linear")
        self.bn3 = nn.BatchNorm1d(args.lin_channels)
        self.bn3_fusion = nn.BatchNorm1d(args.lin_channels)


        self.list_symmetry_groups = list_symmetry_groups
        self.group_sizes = [group_size[g] for g in list_symmetry_groups]
        self.list_symmetric_operators = [SymmetricOperator(symmetry_group=symmetry_group) for symmetry_group
                                         in list_symmetry_groups]

        self.fc2 = nn.Linear(args.lin_channels, args.lin_channels)
        if type == 'sum':
            self.fc3 = nn.Linear(args.lin_channels, k * num_classes)
        elif type == 'classify':
            self.fc3 = nn.Linear(args.lin_channels, num_classes)
        else:
            self.fc3 = nn.Linear(args.lin_channels, 1)

        self.fc0 = nn.Linear(k*args.lin_channels, args.lin_channels)
        self.bn0 = nn.BatchNorm1d(self.args.lin_channels)
        self.bn0_fusion = nn.BatchNorm1d(self.args.lin_channels)

        self.dropout1 = nn.Dropout(p=args.dropout1)
        self.dropout2 = nn.Dropout(p=args.dropout2)
        self.fusion_coeff = nn.Parameter(torch.tensor(0.1))

        logging.info('***** Model Configuration *****')
        logging.info('Symmetry Groups: %s', "-".join(list_symmetry_groups))
        logging.info("Fusion: %s", args.fusion)
        logging.info("Input Size: %s", input_size)
        logging.info("Num Classes: %s", num_classes)
        logging.info("K: %s", k)
        logging.info('*' * 30)


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
        if self.symmetry_group is None or self.symmetry_group == "":
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

    def siam_operators(self, x):
        # invariant outputs
        x_shape = x.shape  # dim [\sum_{i=1 to k} G_i, B, c, H, W]
        x = x.reshape(x_shape[0] * x_shape[1], x_shape[2], x_shape[3], x_shape[4])  # dim [(\sum_{i=1 to k} G_i)*B, c, H, W]
        x = self.bn1(self.pool(F.relu(self.siam_layer1(x))))  # Siamese layer
        x = self.bn2(self.pool(F.relu(self.siam_layer2(x))))  # # Siamese layer
        # extra layer for large inputs to avoid overfitting
        if self.args.input_size == 64:
            x = self.bn64(self.pool(F.relu(self.siam_layer64(x))))  # # Siamese layer

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.bn3(self.dropout1(F.relu(self.siam_layer3(x))))  # dim [(\sum_{i=1 to k} G_i)*B, -1]
        x = x.reshape(x_shape[0], x_shape[1], -1)  # dim [(\sum_{i=1 to k} G_i), B, -1]
        x_k = []
        curr_group_size = 0
        for i in range(self.k):
            x_k.append(torch.mean(x[curr_group_size:curr_group_size+self.group_sizes[i]], dim=0))
            curr_group_size += self.group_sizes[i]
        x_k = torch.stack(x_k, dim=1)  # dim [B, k, -1]
        x_k = x_k.reshape(x_k.shape[0], -1)
        return x_k

    def fusion_operators(self, x, x_k):
        # invariant and symmetric (symmetric operators are not used in this case since the siam layers are already invariant)
        # the symmetric operators will be more useful in finetuning cases
        x_shape = x.shape  # dim [\sum_{i=1 to k} G_i, B, c, H, W]
        x = x.reshape(x_shape[0] * x_shape[1], x_shape[2], x_shape[3], x_shape[4])  # dim [(\sum_{i=1 to k} G_i)*B, c, H, W]
        x = self.bn1_fusion(self.pool(F.relu(self.siam_layer1_fusion(x))))  # Siamese layer
        x = self.bn2_fusion(self.pool(F.relu(self.siam_layer2_fusion(x))))  # # Siamese layer
        # extra layer for large inputs to avoid overfitting
        if self.args.input_size == 64:
            x = self.bn64_fusion(self.pool(F.relu(self.siam_layer64_fusion(x))))  # # Siamese layer
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.bn3_fusion(self.dropout1(F.relu(self.siam_layer3_fusion(x))))  # dim [B*k, 196]  # Siamese layer  # dim [(\sum_{i=1 to k} G_i)*B, -1]
        x = x.reshape(x_shape[0], x_shape[1], -1)  # dim [(\sum_{i=1 to k} G_i), B, -1]
        x_k_fusion = []
        curr_group_size = 0
        for i in range(self.k):
            x_k_fusion.append(torch.mean(x[curr_group_size:curr_group_size + self.group_sizes[i]], dim=0))
            curr_group_size += self.group_sizes[i]
        x_k_fusion = torch.stack(x_k_fusion, dim=1)  # dim [B, k, -1]
        x_k_fusion = torch.mean(x_k_fusion, dim=1, keepdim=True)
        x_k_fusion_shape = x_k_fusion.shape  # dim [B, 1, -1]
        x_k = x_k.reshape(x_k_fusion_shape[0], self.k, -1)
        x_k_fusion = x_k_fusion - x_k
        x_k_fusion = x_k_fusion.reshape(x_k_fusion.shape[0], -1)
        return x_k_fusion

    def forward(self, x):
        x_shape = x.shape  # x dim [B, k, c, H, W]
        k = x_shape[1]
        x_group = []
        group_sizes = []
        for i in range(k):
            # a list of different dimensions of transformed input x
            transformed_x = self.group_transformed_input(x[:, i, :, :, :], symmetry_group=self.list_symmetry_groups[
                i])  # dim [G_i, B, c, H, W]
            group_sizes.append(len(transformed_x))
            x_group.append(transformed_x)
        x_group = torch.cat(x_group, dim=0)  # dim [\sum_{i=1 to k} G_i, B, c, H, W]
        x_group_shape = x_group.shape

        x_k = self.siam_operators(x_group)  # dim [B, k, -1]

        if self.fusion:
            x_k_fusion = self.fusion_operators(x_group, x_k)
            x_k = (x_k + self.fusion_coeff * x_k_fusion) / (1 + self.fusion_coeff)

        x = self.dropout2(F.relu(self.fc0(x_k)))  # dim [B, -1]
        x = self.fc3(x)
        return x


