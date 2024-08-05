import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import random
import numpy as np
import h5py
from imresize import imresize
import einops


class TrainSetDataLoader(Dataset):
    def __init__(self, args):
        super(TrainSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        self.scale_factor = args.scale_factor
        self.prob = args.prob
        self.augment = args.augment
        self.mask_ratio = args.mask_ratio
        self.mask_patch = args.mask_patch
        self.drop_prob = args.drop_prob
        if args.task == "SR":
            self.dataset_dir = (
                args.path_for_train
                + "SR_"
                + str(args.angRes_in)
                + "x"
                + str(args.angRes_in)
                + "_"
                + str(args.scale_factor)
                + "x/"
            )
        elif args.task == "RE":
            self.dataset_dir = (
                args.path_for_train
                + "RE_"
                + str(args.angRes_in)
                + "x"
                + str(args.angRes_in)
                + "_"
                + str(args.angRes_out)
                + "x"
                + str(args.angRes_out)
                + "/"
            )
            # pass

        if args.data_name == "ALL":
            self.data_list = os.listdir(self.dataset_dir)
        elif args.data_name == "HCI":
            self.data_list = ["HCI_new", "HCI_old"]
        else:
            self.data_list = [args.data_name]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + "/" + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], "r") as hf:
            Lr_SAI_y = np.array(hf.get("Lr_SAI_y"))  # Lr_SAI_y
            Hr_SAI_y = np.array(hf.get("Hr_SAI_y"))  # Hr_SAI_y
            Lr_SAI_y = np.transpose(Lr_SAI_y, (1, 0))
            Hr_SAI_y = np.transpose(Hr_SAI_y, (1, 0))
            if self.augment == "maskblur":
                Lr_SAI_y, Hr_SAI_y = augmentation_maskblur(
                    Lr_SAI_y,
                    Hr_SAI_y,
                    self.angRes_in,
                    self.scale_factor,
                    self.prob,
                    self.mask_ratio,
                    self.mask_patch,
                    self.drop_prob,
                )
            elif self.augment == "default":
                Lr_SAI_y, Hr_SAI_y = augmentation(Lr_SAI_y, Hr_SAI_y)
            Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
            Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out

        return Lr_SAI_y, Hr_SAI_y, [Lr_angRes_in, Lr_angRes_out]

    def __len__(self):
        return self.item_num


def MultiTestSetDataLoader(args):
    # get testdataloader of every test dataset
    data_list = None
    if args.data_name in ["ALL", "RE_Lytro", "RE_HCI"]:
        if args.task == "SR":
            dataset_dir = (
                args.path_for_test
                + "SR_"
                + str(args.angRes_in)
                + "x"
                + str(args.angRes_in)
                + "_"
                + str(args.scale_factor)
                + "x/"
            )
            data_list = os.listdir(dataset_dir)
        elif args.task == "RE":
            dataset_dir = (
                args.path_for_test
                + "RE_"
                + str(args.angRes_in)
                + "x"
                + str(args.angRes_in)
                + "_"
                + str(args.angRes_out)
                + "x"
                + str(args.angRes_out)
                + "/"
                + args.data_name
            )
            data_list = os.listdir(dataset_dir)
    else:
        data_list = [args.data_name]

    test_Loaders = []
    length_of_tests = 0
    for data_name in data_list:
        test_Dataset = TestSetDataLoader(
            args, data_name, Lr_Info=data_list.index(data_name)
        )
        length_of_tests += len(test_Dataset)

        test_Loaders.append(
            DataLoader(
                dataset=test_Dataset,
                num_workers=args.num_workers,
                batch_size=1,
                shuffle=False,
            )
        )

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name="ALL", Lr_Info=None):
        super(TestSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out

        if args.task == "SR":
            self.dataset_dir = (
                args.path_for_test
                + "SR_"
                + str(args.angRes_in)
                + "x"
                + str(args.angRes_in)
                + "_"
                + str(args.scale_factor)
                + "x/"
            )
            self.data_list = [data_name]
        elif args.task == "RE":
            self.dataset_dir = (
                args.path_for_test
                + "RE_"
                + str(args.angRes_in)
                + "x"
                + str(args.angRes_in)
                + "_"
                + str(args.angRes_out)
                + "x"
                + str(args.angRes_out)
                + "/"
                + args.data_name
                + "/"
            )
            self.data_list = [data_name]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + "/" + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], "r") as hf:
            Lr_SAI_y = np.array(hf.get("Lr_SAI_y"))
            Hr_SAI_y = np.array(hf.get("Hr_SAI_y"))
            Sr_SAI_cbcr = np.array(hf.get("Sr_SAI_cbcr"), dtype="single")
            Lr_SAI_y = np.transpose(Lr_SAI_y, (1, 0))
            Hr_SAI_y = np.transpose(Hr_SAI_y, (1, 0))
            Sr_SAI_cbcr = np.transpose(Sr_SAI_cbcr, (2, 1, 0))

        Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
        Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())
        Sr_SAI_cbcr = ToTensor()(Sr_SAI_cbcr.copy())

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out
        LF_name = self.file_list[index].split("/")[-1].split(".")[0]

        return Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, [Lr_angRes_in, Lr_angRes_out], LF_name

    def __len__(self):
        return self.item_num


def flip_SAI(data, angRes):
    if len(data.shape) == 2:
        H, W = data.shape
        data = data.reshape(H, W, 1)

    H, W, C = data.shape
    data = data.reshape(angRes, H // angRes, angRes, W // angRes, C)  # [U, H, V, W, C]
    data = data[::-1, ::-1, ::-1, ::-1, :]
    data = data.reshape(H, W, C)

    return data


def augmentation(data, label):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
    return data, label

def augmentation_maskblur(
    data, label, angRes=5, scale_factor=4, prob=1.0, mask_ratio=0.7, mask_patch=1, drop=0.5
):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
    # data: U * patchsize//scale, V * patchsize//scale
    # label: U * patchsize, V * patchsize
    lr_H, lr_W = data.shape
    hr_H, hr_W = label.shape
    patchsize = hr_H // angRes

    label = einops.rearrange(label, "(u H) (v W) -> (u v) H W", u=angRes, v=angRes)
    data = einops.rearrange(data, "(u H) (v W) -> (u v) H W", u=angRes, v=angRes)

    # mask_ratio = 0.7, prob = 0.25, drop = 0.5, mask_patch = 4
    label, data = maskblur(label, data, scale_factor, prob, mask_ratio, mask_patch, drop)

    # data: U * patchsize//scale, V * patchsize//scale
    # label: U * patchsize, V * patchsize
    label = einops.rearrange(label, "(u v) H W -> (u H) (v W)", u=angRes, v=angRes)
    data = einops.rearrange(data, "(u v) H W -> (u H) (v W)", u=angRes, v=angRes)
    return data, label

def maskblur(im1, im2, scale, prob=0.25, mask_ratio=0.5, mask_patch=4, drop=0.75):
    """
    LR -> HR in HR space paste by cwt;
    HR -> LR in LR space paste;
    im1: label, im2: data
    """
    # im1: label, im2: data
    mask_ratio = np.random.randn() * 0.01 + mask_ratio
    an, h_hr, w_hr = im1.shape  # an: U*V
    an, h_lr, w_lr = im2.shape  # an: U*V

    # random view selection
    random_an = np.random.choice(an, size=int(an*(1-drop)), replace=False)
    # generate spatial mask
    mask = generate_mask((h_lr, w_lr), mask_ratio, mask_patch)

    if np.random.random() < prob:
        if np.random.random() > 0.5:
            # if False:
            # LR -> HR
            im2_aug = im2.copy()

            for i in random_an:
                im2_aug[i] = imresize(im1[i], scalar_scale=1 / scale, method='bilinear')
                im2_aug[i] = im2[i] * mask + im2_aug[i] * (1 - mask)
            im2 = im2_aug
            return im1, im2
        else:
            # HR -> LR
            im2_aug = im2.copy()
            for i in random_an:
                im2_aug[i] = imresize(im1[i], scalar_scale=1 / scale, method='bilinear')
                im2_aug[i] = im2_aug[i] * mask + im2[i] * (1 - mask)
            im2 = im2_aug
            return im1, im2
    else:
        return im1, im2


def generate_mask(size, mask_ratio, mask_patchsize=1):
    """
    Generate a random mask given size and mask ratio.
    shape: [H, W]
    mask_ratio: 0~1
    return: mask
    """
    H, W = size
    L = int((H // mask_patchsize) * (W // mask_patchsize))
    len_keep = round(L * mask_ratio)
    if len_keep == 0:
        mask = np.zeros(size)
        return mask

    noise = np.random.rand(L)  # noise in [0, 1]
    # sort noise for each sample
    # ascend: small is keep, large is remove
    idx_shuffle = np.argsort(noise)
    idx_resort = np.argsort(idx_shuffle)

    # generate the binary mask: 1 is keep, 0 is remove
    mask = np.zeros(L)
    mask[:len_keep] = 1
    # unshuffle to get the binary mask
    mask = np.take(mask, idx_resort)
    mask = mask.reshape(int(H // mask_patchsize), int(W // mask_patchsize))
    # print(mask)
    mask = mask.repeat(mask_patchsize, axis=0).repeat(mask_patchsize, axis=1)
    # print(mask)
    return mask