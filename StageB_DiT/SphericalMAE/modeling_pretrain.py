# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from SphericalMAE.modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from SphericalMAE.util.layers import onering_conv_layer_batch,upconv_layer_batch,Get_feature,pool_layer_batch
from SphericalMAE.util.utils import Get_neighs_order,Get_upconv_index
neigh_orders_163842, neigh_orders_40962, neigh_orders_10242, \
        neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
upconv_top_index_163842, upconv_down_index_163842, upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, \
    upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index()
# conv = onering_conv_layer_batch(1,1, neigh_orders_163842).cuda()
# conv2 = onering_conv_layer_batch(1, 1, neigh_orders_40962).cuda()
# get_feature = Get_feature(1,neigh_orders_40962).cuda()
# pool = pool_layer_batch(neigh_orders_163842).cuda()
# upconv2 = upconv_layer_batch(7, 1, upconv_top_index_163842, upconv_down_index_163842).cuda()
# upconv = upconv_layer_batch(7, 7, upconv_top_index_40962, upconv_down_index_40962).cuda()
def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'pretrain_mae_base_patch7_642',
    'pretrain_mae_large_patch7_40962',
]



class PatchEmbed1D(nn.Module):
    """ 1 Dimensional version of data (fmri voxels) to Patch Embedding
    """
    def __init__(self, num_voxels=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = num_voxels // patch_size
        self.patch_shape = patch_size
        self.num_voxels = num_voxels
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, V = x.shape # batch, channel, voxels
        # assert V == self.num_voxels, \
        #     f"Input fmri length ({V}) doesn't match model ({self.num_voxels})."
        x = self.proj(x).transpose(1, 2).contiguous() # put embed_dim at the last dimension
        return x


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, vertex_size=327684, patch_size=7, in_chans=1, num_classes=0, embed_dim=7, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed1D(vertex_size, patch_size, in_chans, embed_dim)

        num_patches = self.patch_embed.num_patches

        # TODO: Add the cls token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim, dtype=torch.float)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):
        #print(x.shape)
        #print(mask.shape)
        x = self.patch_embed(x)
        #print(x.shape)
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        # x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible
        #x_vis = x_vis.double()
        #print(x_vis.shape)
        #x_vis = x[torch.logical_not(mask)].reshape(B, -1, C)
        for blk in self.blocks:
            x_vis = blk(x_vis)
        #x_vis = x_vis.double()
        #print(x_vis.dtype)
        x_vis = x_vis.float()
        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x

class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=7, num_classes=0, embed_dim=7, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196,
                 ):
        super().__init__()
        self.num_classes = num_classes
        #assert num_classes == 3 * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        # if return_token_num > 0:
        #     x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        # else:
        x = self.head(self.norm(x)) # [B, N, 3*16^2]

        return x



class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 vertex_size=40962,
                 patch_size=7,
                 encoder_in_chans=1,
                 encoder_num_classes=0, 
                 encoder_embed_dim=7,
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=768,
                 decoder_embed_dim=7,
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 norm_layer2=nn.BatchNorm1d
                 #relu = nn.Relu
                 ):
        super().__init__()

        self.encoder = PretrainVisionTransformerEncoder(
            vertex_size=vertex_size,
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, 
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False,dtype=torch.float)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def patchify(self, imgs):
        """
        imgs: (N, 1, num_voxels)
        x: (N, L, patch_size)
        """
        p = self.patch_embed.patch_size
        assert imgs.ndim == 3 and imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], h, p))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size)
        imgs: (N, 1, num_voxels)
        """
        p = self.encoder.patch_embed.patch_size
        h = x.shape[1]

        imgs = x.reshape(shape=(x.shape[0], 1, h * p))
        return imgs

    def forward(self, x, mask):

        x_vis_temp = self.encoder(x, mask) # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis_temp) # [B, N_vis, C_d]
            
        B, N, C = x_vis.shape

        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
        

        ## notice: if N_mask==0, the shape of x is [B, N_mask, 3 * 16 * 16]
        x = self.decoder(x_full, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16] (B,856,768)
        x = self.unpatchify(x)

        return x



class EncoderVisionTransformer(nn.Module):
    """部分 Vision Transformer 模型，只计算到 x_vis_temp"""
    def __init__(self,
                 vertex_size=327684,
                 patch_size=332,
                 encoder_in_chans=1,
                 encoder_num_classes=0, 
                 encoder_embed_dim=7,
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 norm_layer2=nn.BatchNorm1d
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            vertex_size=vertex_size,
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, 512, bias=False, dtype=torch.float)

       
    def forward(self, x, mask):
     
        x_vis_temp = self.encoder(x, mask) # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis_temp) # [B, N_vis, C_d]
        return x_vis




@register_model
def pretrain_mae_small_patch7_40962(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        vertex_size=10242,
        patch_size=7,
        encoder_embed_dim=70,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=0,
        decoder_embed_dim=35,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_mae_base_patch7_642(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        vertex_size=1284,
        patch_size=1,
        encoder_embed_dim=1024,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=4,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
 

@register_model
def pretrain_mae_large_patch7_40962(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        vertex_size=40962,
        patch_size=7,
        encoder_embed_dim=21,
        encoder_depth=24, 
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=77,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-4), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

if __name__ == '__main__':
    #model = PretrainVisionTransformer()
    # import nibabel as nib
    # test_path = 'C:/Users/12993/Desktop/test.mgh'
    # data = nib.load(test_path)
    # test = data.get_fdata()
    # # test = test.reshape(163842,-1).transpose(1,0)
    # print(test.shape)
    data =torch.rand(1,40962,7)
    #print(data.dtype)

    #data = torch.Tensor(data)
    #print(data.dtype)
    #data = data.double()
    #print(data.dtype)
    #     #.transpose(0,1)
    # data = test[:,:40962]
    #data = torch.rand(750,1,40962,7)
    from masking_generator import RandomMaskingGenerator
    masking_generator = RandomMaskingGenerator((40962,1),0.9)
    mask = masking_generator()
    mask = torch.tensor(mask)
    mask= mask.flatten(0).to(torch.bool).reshape(1,-1)
    model = PretrainVisionTransformerEncoder()
    output = model(data,mask)
    print(output.shape)
