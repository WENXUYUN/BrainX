# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os
import torch
import glob
import numpy as np
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from timm.data import create_transform

from masking_generator import RandomMaskingGenerator
from dataset_folder import SurfaceFolder,Val_dataset
import random

class DataAugmentationForMAE(object):
    def __init__(self, args):
        #imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        #mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        #std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        # self.transform = transforms.Compose([
        #     # transforms.RandomResizedCrop(args.input_size),
        #     transforms.ToTensor(),
        #     # transforms.Normalize(
        #     #     mean=torch.tensor(mean),
        #     #     std=torch.tensor(std))
        # ])
        self.max_noise_level = 0.15
        self.masked_position_generator = RandomMaskingGenerator(
            args.window_size, args.mask_ratio
        )
        #self.mean = np.load("/data1/dataset/NSD_surface/data_preprocess/processed_data/train_subj125mean.npy")
        #self.std = np.load("/data1/dataset/NSD_surface/data_preprocess/processed_data/train_subj125std.npy")
        #self.ToTensor = torch.tensor()
    def __call__(self, x):
        
        #x = np.nan_to_num(x)
        #x = (x - self.mean) / self.std
        
        
        x = torch.tensor(x)
        x = torch.nan_to_num(x)
        #print(x)
        #noise_level = random.uniform(0, self.max_noise_level)
        #noise = torch.randn(x.size()) * noise_level
        #x = x + noise
        return x, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  transform = %s,\n" % str('Z-Score')
        repr += "  transform = %s,\n" % str('To Tensor')
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForMAE(args)
    print("Data Aug = %s" % str(transform))
    #print(args.window_size)
    return SurfaceFolder(args.data_path, transform=transform)

def build_Val_dataset(args):
    transform = DataAugmentationForMAE(args)
    #print("Data Aug = %s" % str(transform))
    #print(args.window_size)
    return Val_dataset(args.data_path, transform=transform)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = SurfaceFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

if __name__ == '__main__':
    data = torch.rand(1, 1, 40962, 7)
    #test = DataAugmentationForMAE(data)
    #test,mask = test()
    # masked_position_generator = RandomMaskingGenerator(
    #     (40962,1), 0.75
    # )
    #
    # mask = masked_position_generator()
    import nibabel as nib
    import numpy as np
    test_path = 'C:/Users/12993/Desktop/test.mgh'
    data = nib.load(test_path)
    test = data.get_fdata()
    # print(test.shape)
    # #TOTensor = torch.tensor()
    # test = test.reshape(163842,-1)
    # print(test.shape)
    # test = torch.tensor(test)
    #print(test.dtype)
    # from sklearn.preprocessing import StandardScaler
    #
    # scaler = StandardScaler()
    # normalized_matrix = scaler.fit_transform(test)
    # test = normalized_matrix
    # test = torch.tensor(test)
    # mean_value = torch.mean(test).float()
    # std_value = torch.std(test).float()
    # # 判断是否已经正规化
    # is_normalized = torch.allclose(mean_value, torch.tensor(0.0)) and torch.allclose(std_value, torch.tensor(1.0))
    #
    # # 输出结果
    # print(f"均值: {mean_value.item()}")
    # print(f"方差: {std_value.item()}")
    # print(f"是否已正规化: {is_normalized}")
