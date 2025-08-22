# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import glob

from torchvision.datasets.vision import VisionDataset

from PIL import Image
import torch
import os
import os.path
import random
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import nibabel as nib
import numpy as np
# def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
#     """Checks if a file is an allowed extension.
#
#     Args:
#         filename (string): path to a file
#         extensions (tuple of strings): extensions to consider (lowercase)
#
#     Returns:
#         bool: True if the filename ends with one of given extensions
#     """
#     return filename.lower().endswith(extensions)
#
#
# def is_image_file(filename: str) -> bool:
#     """Checks if a file is an allowed image extension.
#
#     Args:
#         filename (string): path to a file
#
#     Returns:
#         bool: True if the filename ends with a known image extension
#     """
#     return has_file_allowed_extension(filename, IMG_EXTENSIONS)


# def make_dataset(
#     directory: str,
#     #class_to_idx: Dict[str, int],
#     extensions: Optional[Tuple[str, ...]] = None,
#     is_valid_file: Optional[Callable[[str], bool]] = None,
# ) -> List[Tuple[str, int]]:
#     instances = []
#     directory = os.path.expanduser(directory)
#     both_none = extensions is None and is_valid_file is None
#     both_something = extensions is not None and is_valid_file is not None
#     if both_none or both_something:
#         raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
#     if extensions is not None:
#         def is_valid_file(x: str) -> bool:
#             return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
#     is_valid_file = cast(Callable[[str], bool], is_valid_file)
#     for target_class in sorted(class_to_idx.keys()):
#         class_index = class_to_idx[target_class]
#         target_dir = os.path.join(directory, target_class)
#         if not os.path.isdir(target_dir):
#             continue
#         for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
#             for fname in sorted(fnames):
#                 path = os.path.join(root, fname)
#                 if is_valid_file(path):
#                     item = path, class_index
#                     instances.append(item)
#     return instances

def split_dataset(dataset_addresses, train_ratio):
    # 设置随机种子，以确保每次运行程序时划分结果一致
    random.seed(33)

    # 随机打乱数据集地址列表
    random.shuffle(dataset_addresses)

    # 划分训练集和测试集的索引
    split_index = int(train_ratio * len(dataset_addresses))

    train_addresses = dataset_addresses[:split_index]
    val_addresses = dataset_addresses[split_index:]

    return train_addresses, val_addresses


def compute_stats(data):
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    count = data.size
    return mean,std,count

def make_dataset(directory):
    """
    寻找给定目录下所有的.mgh文件，并返回它们的路径列表。

    参数：
    - directory: 要搜索的目录路径

    返回：
    - mgh_files: 包含所有.mgh文件路径的列表
    """
    
    surface = np.load('/data1/dataset/NSD_surface/data_preprocess/processed_data/subj125_642.npy')
    #surface = np.memmap("/data1/dataset/HCP_preprocess/preprocessed_data_Z-Score.npy",dtype=np.float32,mode="r")
    #surface_l = surface.reshape(-1,1,327684)[:,:,:642]
    #surface_r = surface.reshape(-1,1,327684)[:,:,163842:164484]
    print("successfully load fMRI data")
    
    #surface = np.concatenate((surface_l,surface_r),axis = -1)
    mean = np.load("/data1/dataset/NSD_surface/data_preprocess/processed_data/subj125_642_mean.npy")
    std = np.load("/data1/dataset/NSD_surface/data_preprocess/processed_data/subj125_642_std.npy")
    surface = (surface-mean)/std
    print('surface shape:',surface.shape)
    #train_files = surface[:5254]
    #val_files = surface[5254:]
    surface_val = torch.randn(1,1,327684)
    return surface,surface_val
class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            #extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            #target_transform: Optional[Callable] = None,
            #is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform)
                                            #target_transform=target_transform)
        #classes, class_to_idx = self._find_classes(self.root)
        samples,vals = make_dataset(self.root)

        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            # if extensions is not None:
            #     msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        #self.extensions = extensions

        #self.classes = classes
        #self.class_to_idx = class_to_idx
        self.samples = samples
        #self.targets = [s[1] for s in samples]

    # def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
    #     """
    #     Finds the class folders in a dataset.
    #
    #     Args:
    #         dir (string): Root directory path.
    #
    #     Returns:
    #         tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
    #
    #     Ensures:
    #         No class is a subdirectory of another.
    #     """
    #     classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    #     classes.sort()
    #     class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    #     return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        while True:
            try:
                #path = self.samples[index]
                sample = self.samples[index]
                #sample = sample.double()
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.samples) - 1)

        if self.transform is not None:
            sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return sample#, target

    def __len__(self) -> int:
        return len(self.samples)


class ValFolder(VisionDataset):


    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            #extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            #target_transform: Optional[Callable] = None,
            #is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(ValFolder, self).__init__(root, transform=transform)
                                            #target_transform=target_transform)
        #classes, class_to_idx = self._find_classes(self.root)
        samples,vals = make_dataset(self.root)

        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            # if extensions is not None:
            #     msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        #self.extensions = extensions

        #self.classes = classes
        #self.class_to_idx = class_to_idx
        self.vals = vals
        #self.targets = [s[1] for s in samples]



    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        while True:
            try:
                path = self.vals[index]
                sample = self.loader(path)
                #sample = sample.double()
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.vals) - 1)

        if self.transform is not None:
            sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return sample#, target

    def __len__(self) -> int:
        return len(self.vals)





def default_loader(path: str):
    data = nib.load(path)
    data = data.get_fdata()
    #data = data.double()
#把球面卷积加入进来 /观察下结构
    return data

#需要修改


class SurfaceFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            #is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(SurfaceFolder, self).__init__(root, loader,
                                          transform=transform,
                                          #target_transform=target_transform,
                                          )
        self.surfaces = self.samples
        #List of FMRI surfaces

class Val_dataset(ValFolder):

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            #is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(Val_dataset, self).__init__(root, loader,
                                          transform=transform,
                                          #target_transform=target_transform,
                                          )
        self.surfaces = self.vals
        #List of FMRI surfaces





if __name__ == '__main__':
    root_directory = 'D:/NSD/data_split'
    result = make_dataset(root_directory)

    # 打印结果
    print(len(result))
