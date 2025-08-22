import numpy as np
import wandb
import torch
from dc_ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch.nn as nn
import os
from dc_ldm.models.diffusion.plms import PLMSSampler
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import nibabel as nib
from PIL import Image
from pathlib import Path
import nibabel.gifti as gifti
from timm.models import create_model


import utils
from dc_ldm import modeling_pretrain

from dc_ldm.modeling_pretrain import EncoderVisionTransformer


from datasets import DataAugmentationForMAE

from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from functools import partial
from timm.models.registry import register_model
from timm.models.vision_transformer import trunc_normal_
from masking_generator import RandomMaskingGenerator


def load_partial_state_dict(model, state_dict, required_keys):
    model_state_dict = model.state_dict()
    for key in required_keys:
        if key in state_dict:
            model_state_dict[key] = state_dict[key]
    model.load_state_dict(model_state_dict, strict=False)





def get_args():
    parser = argparse.ArgumentParser('MAE visualization reconstruction script', add_help=False)
    parser.add_argument('--surface_path', default='/media/test/Cui/NSD/data_01/session01_111.mgh', type=str,
                        help='input surface path')
    parser.add_argument('--save_path', default='./run/test1', type=str, help='save image path')
    parser.add_argument('--model_path', default='./results/Z-Score_OHEM_clip/checkpoint-499.pth', type=str,
                        help='checkpoint path of model')

    parser.add_argument('--vertex_size', default=642, type=int,
                        help='images input size for backbone')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.0, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='pretrain_mae_base_patch7_642', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    return parser.parse_args()

#GLM
mean = 0.17789579155608098
var = 3.289685304154258
var = np.sqrt(var)
max_value = 117.6537857055664
min_value = -110.82479858398438


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    return model

class cond_stage_model(nn.Module):
    def __init__(self,metafile,cond_dim=1280,global_pool=True):
        super().__init__()
        
        
        # 初始化部分模型
        partial_model = EncoderVisionTransformer(
        vertex_size=642,
        patch_size=128,
        encoder_embed_dim=1024,
        encoder_depth=12,
        encoder_num_heads=12,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        #加载预训练权重，指定只加载到 x_vis_temp 的权重
        checkpoint = torch.load(metafile, map_location="cpu")
        required_keys = [
        key for key in checkpoint["model"].keys() 
        if key.startswith("conv") or 
        key in [
           "norm1.weight", "norm1.bias",
           "norm1_1.weight", "norm1_1.bias",
           "norm2.weight", "norm2.bias",
           "norm2_1.weight", "norm2_1.bias",
           "norm3.weight", "norm3.bias",
           "norm3_1.weight", "norm3_1.bias",
           "norm4.weight", "norm4.bias",
           "norm.weight", "norm.bias"
        ] or 
        (key.startswith("encoder") and "encoder_to_decoder" not in key)
        ]
        load_partial_state_dict(partial_model, checkpoint["model"], required_keys)
        
        
        # prepare pretrained fmri mae
        #args = get_args()
        #cudnn.benchmark = True
        #model = get_model(args)
        #patch_size = partial_model.encoder.patch_embed.patch_size#model.encoder.patch_embed.patch_size
        #print("Patch size = %s" % str(patch_size))
        #args.window_size = (args.vertex_size // patch_size[0], 1024 // patch_size[1])
        #args.patch_size = patch_size
        
        #masked_position_generator = RandomMaskingGenerator(
        #    args.window_size, args.mask_ratio
        #)
        #bool_masked_pos = masked_position_generator()
        #bool_masked_pos = torch.from_numpy(bool_masked_pos)
        #bool_masked_pos = bool_masked_pos[None, :]
        #bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
        
        self.bool_masked_pos = torch.zeros((856,),dtype=torch.bool)#bool_masked_pos
        
        #checkpoint = torch.load(metafile, map_location='cpu')
        #model.load_state_dict(checkpoint['model'])
        self.mae = partial_model#model
        #self.fmri_mapper = nn.Linear(856, 77, bias=True)
        #具体情况具体修改
        self.fmri_seq_len = 856
        self.fmri_latent_dim = 1024
        
        #test
        self.PCA_dim = 512
        self.PCA = torch.nn.Linear(self.fmri_latent_dim, self.PCA_dim, bias=False)
        if global_pool == False:
            self.channel_mapper = nn.Sequential(
                nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 4, 1, bias=True),
                #nn.Conv1d(self.fmri_seq_len // 2, self.fmri_seq_len // 4, 1, bias=True),
                #nn.Conv1d(self.fmri_seq_len // 4, self.fmri_seq_len // 8, 1, bias=True),
                nn.Conv1d(self.fmri_seq_len // 4, 77, 1, bias=True)
            )
        #self.dim_mapper = nn.Linear(self.fmri_latent_dim, cond_dim, bias=True)
        self.dim_mapper = nn.Linear(self.PCA_dim, cond_dim, bias=True)
        self.global_pool = global_pool
        
    def forward(self, x):
        n, c, w = x.shape
        #device = torch.device(self.args.device)
        #self.mae.to(device)
        
        #surfaces, bool_masked_pos = self.transforms(x)
        
        #surfaces = x.reshape(n, -1, 1)#surfaces.reshape(n, -1, 1)
        
        #surfaces = surfaces.to(device, non_blocking=True)
        
        #surfaces = (surfaces - mean) / var
        bool_masked_pos = self.bool_masked_pos.repeat(n, 1)
        #print(bool_masked_pos.shape)
        #print(bool_masked_pos)
        x = x.to(torch.float)
        surfaces_l = x[:, :163842, :]
        surfaces_r = x[:, 163842:, :]
        x = torch.cat([surfaces_l,surfaces_r],dim=-1)
        outputs = self.mae(x, bool_masked_pos)#.permute(0,2,1)
        #features = torch.cat([outputs_l,outputs_r],dim=1)
        #print(outputs.shape)
        #latent_crossattn = outputs#.permute(0,2,1)
        latent_crossattn = self.PCA(outputs)
        if self.global_pool == False:
            latent_crossattn = self.channel_mapper(latent_crossattn)
        latent_crossattn = self.dim_mapper(latent_crossattn)
        out = latent_crossattn

        return out

class fLDM:

    def __init__(self, metafile, device=torch.device('cpu'),
                 pretrain_root='../pretrains/ldm/label2img',
                 logger=None, ddim_steps=250, global_pool=True, use_time_cond=True):
        self.ckp_path = os.path.join(pretrain_root, 'model.ckpt')
        self.config_path = os.path.join(pretrain_root, 'config.yaml') 
        config = OmegaConf.load(self.config_path)
        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool

        self.cond_dim = config.model.params.unet_config.params.context_dim
        
        model = instantiate_from_config(config.model)
        pl_sd = torch.load(self.ckp_path, map_location="cpu")['state_dict']
       
        m, u = model.load_state_dict(pl_sd, strict=False)
        model.cond_stage_trainable = True
        model.cond_stage_model = cond_stage_model(metafile,self.cond_dim, global_pool=global_pool)
        #model.PCA = torch.nn.Linear(512, 128)
        model.ddim_steps = ddim_steps
        model.re_init_ema()
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult

        self.device = device    
        self.model = model
        self.ldm_config = config
        self.pretrain_root = pretrain_root
        self.fmri_latent_dim = model.cond_stage_model.fmri_latent_dim
        self.metafile = metafile
        
        #self.projector = torch.nn.Linear(39424, 25088)
        ### get distillation model test


    def finetune(self, trainers, dataset, test_dataset, bs1, lr1,
                output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        # self.model.train_dataset = dataset
        self.model.run_full_validation_threshold = 0.15
        # stage one: train the cond encoder with the pretrained one
      
        # # stage one: only optimize conditional encoders
        print('\n##### Stage One: only optimize conditional encoders #####')
        dataloader = DataLoader(dataset, batch_size=8,shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        self.model.unfreeze_whole_model()
        self.model.freeze_first_stage()

        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = True
        self.model.eval_avg = config.eval_avg
        trainers.fit(self.model, dataloader, val_dataloaders=test_loader)
        
        self.model.unfreeze_whole_model()
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'config': config,
                'state': torch.random.get_rng_state()

            },
            os.path.join(output_path, 'checkpoint.pth')
        )
        

    @torch.no_grad()
    def generate(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None, vqgan_model=None):
        # fmri_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels, 
                self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model)
        #if state is not None:
            #torch.cuda.set_rng_state(state)
            
        with model.ema_scope():
            model.eval()
            for count, item in enumerate(fmri_embedding):
                if limit is not None:
                    if count >= limit:
                        break
                latent = item['fmri']

                gt_image = rearrange(item['image'], 'h w c -> 1 c h w') # h w c

                print(f"rendering {num_samples} examples in {ddim_steps} steps.")
                # assert latent.shape[-1] == self.fmri_latent_dim, 'dim error'
                
                #item['pad_fmri'] = torch.tensor(item['pad_fmri']).to(self.device)
                #item['image']=torch.tensor(item['image']).to(self.device)
                #vqgan_model = vqgan_model.to(self.device)
                #x = vqgan_model(item['fmri'].unsqueeze(0),
                #item['image'].unsqueeze(0).permute(0,3,1,2).float())
                
           
                c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                #samples_ddim, _ = sampler.sample(S= 1, #ddim_steps,
                                                #x_T = x[1], 
                                                #conditioning=c,
                                                #batch_size=num_samples,
                                                #shape=shape,
                                                #verbose=False)
                samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                conditioning=c,
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)
                
                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
                
        
        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to('cpu')
        
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)


