# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models, fMRIEmbedder
import argparse
from torch.utils.data import Dataset
from einops import rearrange, repeat
import numpy as np
from eval_metrics import get_similarity_metric
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image


import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

def find_best_sample(samples, image_latent):
    # 计算每个样本与 image_latent 的均方误差 (MSE)
    # 假设 samples 是 (N, 4, 32, 32)，image_latent 是 (1, 4, 32, 32)

    # 扩展 image_latent 的维度以匹配 samples 的维度
    #image_latent = image_latent.unsqueeze(0)  # (1, 4, 32, 32) -> (N, 4, 32, 32)

    # 计算样本与 image_latent 的均方误差 (MSE)
    mse = torch.mean((samples - image_latent) ** 2, dim=(1, 2, 3))  # 在 (4, 32, 32) 维度上计算均方误差

    # 找到最小的均方误差样本的索引
    best_sample_idx = torch.argmin(mse)


    return best_sample_idx

def save_comparison(gt_image, best_sample, save_path, idx):
    """
    将 GT 图像和最佳生成图像拼接并保存。

    Args:
        gt_image (torch.Tensor): Ground Truth 图像，形状 (C, H, W)。
        best_sample (torch.Tensor): 最佳生成图像，形状 (C, H, W)。
        save_path (str): 保存路径。
        idx (int): 当前图像的索引，用于命名文件。
    """
    # 设置子图
    num_xaxis_subplots = 2  # 两个子图：GT 和最佳生成图像
    fig, ax = plt.subplots(1, num_xaxis_subplots, figsize=(8, 4))

    # 将图像从 Tensor 转换为 PIL 格式
    gt_img_pil = to_pil_image(gt_image)
    best_img_pil = to_pil_image(best_sample)

    # 绘制 GT 图像
    ax[0].imshow(gt_img_pil)
    ax[0].set_title("Ground Truth", fontweight='bold')
    ax[0].axis('off')

    # 绘制最佳生成图像
    ax[1].imshow(best_img_pil)
    ax[1].set_title("Best Reconstruction", fontweight='bold')
    ax[1].axis('off')

    # 保存拼接图像
    plt.tight_layout()
    plt.savefig(f"{save_path}/comparison_{idx}.png")
    plt.close(fig)

def channel_modify(img):
    if img.shape[-1] == 3:
        return rearrange(img, 'h w c -> c h w')
    return img
def identity(x):
    return x

def create_NSD_dataset(path='/media/test/Cui/NSD/nsddata_stimuli/stimuli/nsd', patch_size=16,
                           fmri_transform=identity,
                           image_transform=identity, include_nonavg_test=False):
    fmri_train = torch.rand(128,163842,2)
    #fmri_train = torch.load('/data/dataset/NSD_surface/subj01/fmri_Zscore.pt')["fmri_train"]  # [:100]
    #print(fmri_train.shape)
    fmri_test = np.load('/data1/dataset/NSD_surface/data_preprocess/processed_data/subj01/nsd_test_fmriavg_sub1.npy')
    
    surface_l = fmri_test.reshape(-1,1,327684)[:,:,:642]
    surface_r = fmri_test.reshape(-1,1,327684)[:,:,163842:164484]
    fmri_test = np.concatenate((surface_l,surface_r),axis = -1)
    #fmri_test = torch.rand(5,163842,2)
    #img_train = torch.load('/data/dataset/NSD_surface/subj01/img.pt')["img_train"]  # [:100]
    img_train = torch.rand(128,425,425,3)
    #print(img_train.shape)
    img_test = np.load('/data1/dataset/NSD_surface/data_preprocess/processed_data/subj01/nsd_test_stim_sub1.npy')
    #img_test = torch.load('/data1/dataset/NSD_surface/img_test.pt')["img_test"]#[:10]
    #img_test = torch.rand(5,425,425,3)
    if isinstance(image_transform, list):
        return (NSD_dataset(fmri_train, img_train, fmri_transform, image_transform[0]),
                NSD_dataset(fmri_test, img_test, torch.FloatTensor, image_transform[1]))
    else:
        return (NSD_dataset(fmri_train, img_train, fmri_transform, image_transform),
                NSD_dataset(fmri_test, img_test, torch.FloatTensor, image_transform))

class NSD_dataset(Dataset):
    def __init__(self, fmri, image, fmri_transform=identity, image_transform=identity):
        self.fmri = fmri
        self.image = image
        self.fmri_transform = fmri_transform
        self.image_transform = image_transform
        self.mean = np.load("/data1/dataset/NSD_surface/data_preprocess/processed_data/subj125_642_mean.npy")#to different fmri_encoder,change it
        self.std = np.load("/data1/dataset/NSD_surface/data_preprocess/processed_data/subj125_642_std.npy")#to different fmri_encoder,change it
    def __len__(self):
        return len(self.fmri)

    def __getitem__(self, index):
        fmri = torch.nan_to_num(torch.tensor((self.fmri[index] - self.mean) / self.std)).float()
        img = torch.tensor(self.image[index]).float() / 255.0

        return {'fmri': self.fmri_transform(fmri),
                'image': self.image_transform(img)}

def get_eval_metric(samples, avg=True):
    metric_list = ['pcc', 'ssim']
    res_list = []

    gt_images = [img[0] for img in samples]
    # print(gt_images[0].shape)
    # print(len(gt_images))
    gt_images = rearrange(np.stack(gt_images), 'n c h w -> n h w c')
    samples_to_run = np.arange(1, len(samples[0])) if avg else [1]
    for m in metric_list:
        res_part = []
        for s in samples_to_run:
            pred_images = [img[s] for img in samples]
            # print(pred_images[0].shape)
            # print(len(pred_images))
            pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
            res = get_similarity_metric(pred_images, gt_images, method='pair-wise', metric_name=m)
            res_part.append(np.mean(res))
        res_list.append(np.mean(res_part))
    res_part = []
    res_part1 = []
    res_part2 = []
    res_part3 = []
    for s in samples_to_run:
        pred_images = [img[s] for img in samples]
        pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
        res = get_similarity_metric(pred_images, gt_images, 'class', None,
                                    n_way=50, num_trials=1000, top_k=1, device='cuda')
        res_part.append(np.mean(res))

        res1 = get_similarity_metric(pred_images, gt_images, 'class', None,
                                     n_way=50, num_trials=1000, top_k=5, device='cuda')
        res_part1.append(np.mean(res1))

        res2 = get_similarity_metric(pred_images, gt_images, 'class', None,
                                     n_way=100, num_trials=1000, top_k=1, device='cuda')
        res_part2.append(np.mean(res2))

        res3 = get_similarity_metric(pred_images, gt_images, 'class', None,
                                     n_way=100, num_trials=1000, top_k=5, device='cuda')
        res_part3.append(np.mean(res3))

    res_list.append(np.mean(res_part))
    res_list.append(np.mean(res_part1))
    res_list.append(np.mean(res_part2))
    res_list.append(np.mean(res_part3))
    # res_list.append(np.max(res_part))
    metric_list.append('50-way-top-1-class')
    # metric_list.append('50-way-top-1-class (max)')
    metric_list.append('50-way-top-5-class')
    metric_list.append('100-way-top-1-class')
    metric_list.append('100-way-top-5-class')
    return res_list, metric_list


def main(args):
    plt.switch_backend("agg")
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes)
    fMRI_Encoder_path = args.fMRI_Encoder_path
    model.fMRI_embedder = fMRIEmbedder(fMRI_Encoder_path, 0)
    model.to(device)

    # Load pre-trained weights:
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt['ema'])
    print('Model resumed!')
    model.freeze_whole_model()
    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))

    vae_path = './pretrains/vae/ema'
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)

    transform = transforms.Compose([
        channel_modify,
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

    # Load test dataset:
    _, test_dataset = create_NSD_dataset(args.data_path, fmri_transform=identity, image_transform=transform)
    print('Test dataset size:', len(test_dataset))

    # Prepare for metrics and outputs:
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    all_best_metrics = []
    y_null = torch.zeros(args.num_samples, 1, 1284, device=device)
    for idx, item in enumerate(test_dataset):
        latent = item['fmri'].to(device)
        gt_image = rearrange(item['image'], 'c h w -> 1 c h w').to(device)
        print(f"rendering {args.num_samples} examples in {args.num_sampling_steps} steps.")
        gt_image_clamped = rearrange((torch.clamp((gt_image + 1.0) / 2.0, min=0.0, max=1.0) * 255.).cpu().numpy().astype(np.uint8), 'n c h w -> n h w c') 
        y = repeat(latent, 'h w -> c h w', c=args.num_samples).to(device)
        y = torch.cat([y,y_null], 0)
        
        # Generate multiple samples:
        z = torch.randn(args.num_samples, 4, latent_size, latent_size, device=device)
        z = torch.cat([z,z], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        
        samples,_ = samples.chunk(2,dim=0)
        
        image_latent = vae.encode(gt_image).latent_dist.sample().mul_(0.18215)
        
        best_idx = find_best_sample(samples, image_latent)
        
        x_samples_ddim = vae.decode(samples / 0.18215).sample
        
        x_samples_ddim_clamped = rearrange((torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0) * 255.).cpu().numpy().astype(np.uint8), 'n c h w -> n h w c')

        best_sample = np.expand_dims(x_samples_ddim_clamped[best_idx],axis=0)

        # Compute SSIM and PCC for the best sample:
        ssim = get_similarity_metric(best_sample, gt_image_clamped, metric_name='ssim')
        pcc = get_similarity_metric(best_sample, gt_image_clamped, metric_name='pcc')
        best_metric = {'ssim': ssim[0]}
        #best_metric['ssim'] = ssim[0]#.item()
        best_metric['pcc'] = pcc[0]#.item()

        # Compute N-way-top-K metrics for the best sample:
        n_way_trials = 1000
        for n_way, top_k in [(50, 1), (50, 5), (100, 1), (100, 5)]:
            #top_k_acc = get_similarity_metric(best_sample, gt_image_clamped,metric_name='class',None, n_way=n_way, num_trials=n_way_trials, top_k=top_k, device=device)
            top_k_acc = get_similarity_metric(best_sample, gt_image_clamped, 'class', None,
                                    n_way=n_way, num_trials=n_way_trials, top_k=top_k, device='cuda')
            best_metric[f'{n_way}-way-top-{top_k}'] = top_k_acc[0]#.item()
            
        # Save GT and best reconstruction as a single image:
        save_comparison(
            gt_image_clamped[0],  # Ground Truth 图像
            best_sample.squeeze(0),          # 最佳生成图像
            save_path=output_path,
            idx=idx
        )
        print(best_metric)
        # Add the best metric to the list:
        all_best_metrics.append(best_metric)

    # Compute the average of all metrics:
    avg_metrics = {key: np.mean([m[key] for m in all_best_metrics]) for key in all_best_metrics[0].keys()}
    print("Final Metrics:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")

    output_file = os.path.join(output_path, "results.txt")
    with open(output_file, "w") as f:
        f.write("Final Metrics:\n")
        for metric, value in avg_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--data-path", type=str, default='/data/dataset/NSD_surface/subj01')
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fMRI_Encoder_path", type=str, default='./pretrains/fMRI_encoder/NSD_finetune_642.pth', help="fMRI_Encoder save path")
    parser.add_argument("--ckpt", type=str, default='./results/HCP_pretrain_NSD_finetune/checkpoints/0850000.pt', help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--output_path", type=str, default='./eval/best/subj01',help="image save path")

    args = parser.parse_args()
    main(args)
