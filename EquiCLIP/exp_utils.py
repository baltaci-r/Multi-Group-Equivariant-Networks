import os
import random
import torch
import numpy as np

from torch import cos, sin


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    np.random.default_rng(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fliplr(x):
    # fixes dimension issues in torch.fliplr
    x_shape = x.shape  # [B, c, d, d] or [B, d, d] or [..., d, d]
    x = x.reshape(-1, x_shape[-2], x_shape[-1])
    x = torch.fliplr(x)
    x = x.reshape(x_shape)
    return x


def get_canon_indices(x, W, group_name="rot90"):
    if group_name == "":
        return 0
    elif group_name == "flip":
        # canonicalize for flip
        # x dim [B, h, h]
        x_shape = x.shape
        x_stack = torch.stack([x[:, 0:x_shape[1]//2, :], x[:, x_shape[1]//2:, :]], dim=0)  # dim [2, B, d/2, d]
        x_groups = torch.mean(x_stack, dim=(-1, -2), keepdim=False)  # dim [2, B]
        vals, indices = torch.max(x_groups, dim=0)
        return indices  # dim [B,]
    elif group_name == "rot90":
        # canonicalize for rot90
        # x dim [B, h, h]
        x_shape = x.shape
        x_stack = torch.stack([
                                x[:, 0:x_shape[1] // 2, 0:x_shape[1] // 2],
                                x[:, 0:x_shape[1] // 2, x_shape[1] // 2:],
                                x[:, x_shape[1] // 2:, x_shape[1] // 2:],
                                x[:, x_shape[1] // 2:, 0: x_shape[1] // 2],
                              ], dim=0)  # dim [4, B, d/2, d/2]
        x_groups = torch.mean(x_stack, dim=(-1, -2), keepdim=False)  # dim [4, B]
        vals, indices = torch.max(x_groups, dim=0)
        return indices  # dim [B,]
    elif group_name == "rot90_flip":
        # canonicalize for rot90_flip
        x_shape = x.shape  # dim [B, h, h]
        indices = []
        for b in range(x_shape[0]):
            max_val = -float("Inf")
            index_i, index_j = -1, -1
            for i in range(4):
                for j in range(2):
                    x_transformed = x[b]
                    if j == 1:
                        x_transformed = fliplr(x_transformed)
                    x_transformed = torch.rot90(x_transformed, k=i, dims=(-2, -1))
                    val = torch.einsum("ikl, ikl -> ", x_transformed, W)
                    if val >= max_val:
                        max_val = val
                        index_i, index_j = i, j
            indices.append([index_i, index_j])

        return indices
    return


def apply_canon_indices(x, indices, group_name):
    """
    x: dim [B, h, hh]
    indices: dim [B,]
    """
    if group_name == "":
        return x
    elif group_name == "flip":
        # dim x [B, h, h]
        x_shape = x.shape
        x_canon = []
        for i in range(x_shape[0]):
            if indices[i] == 1:
                x_canon.append(torch.fliplr(x[i]))
            else:
                x_canon.append(x[i])
        x_stack = torch.stack(x_canon, dim=0)
    elif group_name == "rot90":
        # dim x [B, h, h]
        x_shape = x.shape
        x_canon = []
        for i in range(x_shape[0]):
            x_canon.append(torch.rot90(x[i], k=-indices[i], dims=(-1, -2)))
        x_stack = torch.stack(x_canon, dim=0)
    elif group_name == "rot90_flip":
        # dim x [..., h, h]
        x_shape = x.shape
        x_canon = []
        for b in range(x_shape[0]):
            if indices[b][1] % 2 == 1:
                x[b] = fliplr(x[b])
            x_canon.append(torch.rot90(x[b], k=-indices[b][0], dims=(-1, -2)))
        x_stack = torch.stack(x_canon, dim=0)
    else:
        raise NotImplementedError
    return x_stack


def canon_invariance_test(group_name="rot90_flip"):
    """
    Tests if the canonicalization operation works
    """
    if group_name == "":
        pass
    elif group_name == "rot90_flip":
        batch_size = 16
        h = 48
        c_in = 3
        x = torch.randn(size=(batch_size, c_in, h, h))
        x_shape = x.shape
        W = torch.rand(size=(x_shape[-3], x_shape[-2], x_shape[-1]), device=x.device)
        hx = fliplr(x)
        rx = torch.rot90(x, k=1, dims=(-1, -2))
        hrx = fliplr(rx)

        # get indices for rot90_flip
        hr_indices_hx = get_canon_indices(hx, W=W, group_name="rot90_flip")
        hr_indices_hrx = get_canon_indices(hrx, W=W, group_name="rot90_flip")
        hr_indices_rx = get_canon_indices(rx, W=W, group_name="rot90_flip")
        hr_indices_x = get_canon_indices(x, W=W, group_name="rot90_flip")

        # torch.stack(list(*zip(h_indices_x, r_indices_x)))

        # # apply canon for flip
        hrx_canon_hr = apply_canon_indices(hrx, indices=hr_indices_hrx, group_name="rot90_flip")
        rx_canon_hr = apply_canon_indices(rx, indices=hr_indices_rx, group_name="rot90_flip")
        hx_canon_hr = apply_canon_indices(hx, indices=hr_indices_hx, group_name="rot90_flip")
        x_canon_hr = apply_canon_indices(x, indices=hr_indices_x, group_name="rot90_flip")

        # assert that canonicalization works for rot90_hflip
        assert torch.allclose(hrx_canon_hr, x_canon_hr), print(f"rot90_hflip canonicalization incorrect!")
        assert torch.allclose(hrx_canon_hr, hx_canon_hr), print(f"rot90_hflip canonicalization incorrect!")
        assert torch.allclose(hrx_canon_hr, rx_canon_hr), print(f"rot90_hflip canonicalization incorrect!")
        print(f"rot90_hflip canonicalization done correctly!")
    return


def group_transform_images(images, group_name="rot90", method="equitune"):
    if method == "equitune" or method == "equizero":
        if group_name == "":
            return torch.stack([images])
        elif group_name == "rot90":
            group_transformed_images = []
            for i in range(4):
                g_images = torch.rot90(images, k=i, dims=(-2, -1))
                group_transformed_images.append(g_images)
            group_transformed_images = torch.stack(group_transformed_images, dim=0)
            return group_transformed_images
        elif group_name == "flip":
            group_transformed_images = []
            for i in range(2):
                if i == 0:
                    g_images = images
                else:
                    g_images = fliplr(images)
                group_transformed_images.append(g_images)
            group_transformed_images = torch.stack(group_transformed_images, dim=0)
            return group_transformed_images
        elif group_name == "rot90_flip":
            group_transformed_images = []
            for i in range(4):
                for j in range(2):
                    # apply G_1 = rot90 transform
                    g_images = torch.rot90(images, k=i, dims=(-2, -1))

                    # apply G_2 = fliplr transform
                    if j == 0:
                        g_images = g_images
                    else:
                        g_images = fliplr(g_images)
                    group_transformed_images.append(g_images)
            group_transformed_images = torch.stack(group_transformed_images, dim=0)
            return group_transformed_images
        else:
            raise NotImplementedError
    elif method == "multi_equitune" or method == "multi_equizero":
        if group_name == "rot90_flip":
            # get h(x), the canonicalization function
            # perform canonicalization: x = (h(x)^{-1}) x
            x_shape = images.shape  # dim [... h, h]
            W = torch.rand(size=(x_shape[-3], x_shape[-2], x_shape[-1]), device=images.device)
            hr_indices_hrx = get_canon_indices(images, W=W, group_name="rot90_flip")  # dim [B, 1, 1]
            x_canon = apply_canon_indices(images, hr_indices_hrx, group_name="rot90_flip")

            # obtain G_1 x
            G_1_x_canon = []
            for i in range(4):
                g1_i_x_canon = torch.rot90(x_canon, k=i, dims=(-2, -1))
                G_1_x_canon.append(g1_i_x_canon)
            G_1_x_canon = torch.stack(G_1_x_canon, dim=0)  # dim [G1, B, c, h, h]

            # obtain G_2 x
            G_2_x_canon = []
            for i in range(2):
                if i == 1:
                    g2_i_x_canon = fliplr(x_canon)
                else:
                    g2_i_x_canon = x_canon
                G_2_x_canon.append(g2_i_x_canon)
            G_2_x_canon = torch.stack(G_2_x_canon, dim=0)  # dim [G2, B, c, h, h]

            # concat [G_1 x, G_2 x]
            G_x_canon = torch.cat([G_1_x_canon, G_2_x_canon], dim=0)
            return G_x_canon  # dimension [G_1 + G_2, B, c, h, h]
        else:
            raise NotImplementedError
    elif method == "vanilla":
        images = torch.stack([images], dim=0)
        return images
    else:
        raise NotImplementedError


class RandomRot90(object):
    """
    Random rotation along given axis in multiples of 90
    """
    def __init__(self, dim1=-2, dim2=-1):
        self.dim1 = dim1
        self.dim2 = dim2
        return

    def __call__(self, sample):
        k = np.random.randint(0, 4)
        out = torch.rot90(sample, k=k, dims=[self.dim1, self.dim2])
        return out


class RandomFlip(object):
    """
    Random rotation along given axis in multiples of 90
    """
    def __init__(self, dim1=-2, dim2=-1):
        self.dim1 = dim1
        self.dim2 = dim2
        return

    def __call__(self, sample):
        k = np.random.randint(0, 2)
        if k == 1:
            out = fliplr(sample)
        else:
            out = sample
        return out


class RandomRot90Flip(object):
    """
    Random rotation along given axis in multiples of 90,
    followed by random flip
    """
    def __init__(self, dim1=-2, dim2=-1):
        self.dim1 = dim1
        self.dim2 = dim2
        return

    def __call__(self, sample):
        # random rotation
        k = np.random.randint(0, 4)
        sample = torch.rot90(sample, k=k, dims=[self.dim1, self.dim2])

        # random flip
        k = np.random.randint(0, 2)
        if k == 1:
            out = fliplr(sample)
        else:
            out = sample
        return out


random_rot90 = RandomRot90()
random_flip = RandomFlip()
random_rot90_flip = RandomRot90Flip()


def random_transformed_images(x, data_transformations=""):
    if data_transformations == "":
        x = x
    elif data_transformations == "rot90":
        x = random_rot90(x)
    elif data_transformations == "flip":
        x = random_flip(x)
    elif data_transformations == "rot90_flip":
        # the order of group action is important: G_1=rot90, G_2=flip
        x = random_rot90_flip(x)
    else:
        raise NotImplementedError
    return x


if __name__ == "__main__":
    canon_invariance_test(group_name="rot90_flip")
