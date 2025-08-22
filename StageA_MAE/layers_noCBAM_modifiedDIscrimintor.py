import numpy as np
import torch.nn as nn
import util.utils as utils
import torch
import scipy.io as sio
import os
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
abspath="/Users/xuepengcheng/Xue/CHD_classification/SphericalUNetPackage/sphericalunet/utils"
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import glob
from torchvision.datasets.vision import VisionDataset
import os
import os.path
import random
from typing import Any, Callable, Tuple
import nibabel as nib

class onering_conv_layer(nn.Module):
    def __init__(self, in_features, out_features, neigh_orders):
        super(onering_conv_layer,self).__init__()

        self.in_features = in_features
        self.out_featrues = out_features
        self.neigh_orders = neigh_orders
        self.weight = nn.Linear(7*in_features,out_features)

    def forward(self,x):
        mat = x[self.neigh_orders]
        mat = mat.view(len(x),7*self.in_features)
        out_features = self.weight(mat)
        return out_features

class onering_conv_layer_batch(nn.Module):
    def __init__(self, in_features,out_features,neigh_orders):
        super(onering_conv_layer_batch,self).__init__()

        self.in_features = in_features
        self.out_featrues = out_features
        self.neigh_orders = neigh_orders
        self.weight = nn.Linear(7 * in_features, out_features,dtype=torch.float)
    ## x.shape = N * features * vertices
    def forward(self,x):
        mat = x[:,:, self.neigh_orders]
        mat = mat.view(x.shape[0], self.in_features, x.shape[2],7).permute(0,2,3,1)
        mat = mat.contiguous().view(x.shape[0],x.shape[2],7*self.in_features)
        out_features = self.weight(mat).permute(0,2,1)
        return out_features

class pool_layer(nn.Module):
    def __init__(self,neigh_orders,pool_type="mean"):
        super().__init__()
        self.neigh_orders = neigh_orders
        self.pool_type = pool_type
    # x.shape = N * output_features
    def forward(self,x):
        number_nodes = int((x.size()[0]+6)/4)
        features_num = x.size()[1]
        x = x[self.neigh_orders[0:number_nodes*7]].view(number_nodes,7,features_num)
        if self.pool_type == "mean":
            x=torch.mean(x,dim=1)
        if self.pool_type == "max":
            x = torch.max(x,dim=1)
            return x[0],x[1]
        return x
class pool_layer_batch(nn.Module):
    def __init__(self,neigh_orders,pool_type="mean"):
        super().__init__()
        self.neigh_orders=neigh_orders
        self.pool_type=pool_type
    # x.shape = B * output_features * N
    def forward(self,x):
        number_nodes = int((x.size()[2]+6)/4)
        features_number = x.size()[1]
        x = x[:,:,self.neigh_orders[0:number_nodes*7]]
        x = x.view(x.size()[0],features_number,number_nodes,7)
        if self.pool_type == "mean":
            x = torch.mean(x, dim=3)
        if self.pool_type == "max":
            x = torch.max(x, dim=3)
            return x[0]
        return x

class upconv_layer(nn.Module):
    def __init__(self, in_features, out_features, upconv_center_indices, upconv_edge_indices):
        super(upconv_layer,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.upcon_center_indices = upconv_center_indices
        self.upcon_edge_indices = upconv_edge_indices
        self.weight = nn.Linear(in_features,7*out_features)
    # N*in_features
    def forward(self,x):
        raw_nodes = x.size()[0]
        new_nodes = int(raw_nodes * 4 - 6)
        x = self.weight(x)
        x = x.view(x.shape[0]*7, self.out_features)
        x1 = x[self.upcon_center_indices]
        assert (x1.size() == torch.Size([raw_nodes, self.out_features]))
        x2 = x[self.upcon_edge_indices].view(-1,self.out_features,2)
        x = torch.cat((x1,torch.mean(x2,dim=2)),0)
        assert(x.size() == torch.Size([new_nodes, self.out_features]))
        return x

class upconv_layer_batch(nn.Module):
    def __init__(self, in_features,out_features,upconv_center_indices,upconv_edge_indices):
        super(upconv_layer_batch,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.upconv_center_indices = upconv_center_indices
        self.upconv_edge_indices = upconv_edge_indices
        self.weight = nn.Conv1d(in_features, 7 * out_features, kernel_size=1)
    # input N * vertices * features
    def forward(self,x):
        raw_nodes = x.size()[2]
        new_nodes = int(raw_nodes * 4 - 6)
        x = self.weight(x) # N * (7*out_features) * vertices
        x = x.permute(0,2,1)
        x = x.contiguous().view(x.shape[0],raw_nodes*7,self.out_features).permute(0,2,1)

        x1 = x[:, :, self.upconv_center_indices]
        assert (x1.size() == torch.Size([x.shape[0], self.out_features, raw_nodes]))
        x2 = x[:, :, self.upconv_edge_indices].view(x.shape[0], self.out_features, -1, 2)
        x = torch.cat((x1, torch.mean(x2, 3)), 2)
        assert (x.size() == torch.Size([x.shape[0], self.out_features, new_nodes]))
        # x = self.norm(x)
        return x


class down_block(nn.Module):
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, pool_neigh_orders, first=False):
        super(down_block,self).__init__()
        # no pooling
        if first:
            self.block = nn.Sequential(
                conv_layer(in_ch,out_ch,neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15,affine=True,track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv_layer(out_ch,out_ch,neigh_orders),
                nn.BatchNorm1d(out_ch,momentum=0.15,affine=True,track_running_stats=False),
                nn.LeakyReLU(0.2,inplace=True)
            )
        else:
            self.block = nn.Sequential(
                pool_layer(pool_neigh_orders,"mean"),
                conv_layer(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv_layer(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self,x):
        out_features= self.block(x)
        return out_features
class down_block_batch(nn.Module):
    def __init__(self, onering_conv_layer_batch, in_ch, out_ch, neigh_orders, pool_neigh_orders = None,first=False):
        super(down_block_batch,self).__init__()
        if first:
            self.block = nn.Sequential(
                onering_conv_layer_batch(in_ch,out_ch,neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                onering_conv_layer_batch(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.block = nn.Sequential(
                pool_layer_batch(pool_neigh_orders,"mean"),
                onering_conv_layer_batch(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                onering_conv_layer_batch(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self,x):
        out_features= self.block(x)
        return out_features

class hierarchical_down_block_batch(nn.Module):
    def __init__(self, in_ch, out_ch, neigh_orders, pool_neigh_orders = None):
        super(hierarchical_down_block_batch,self).__init__()

        self.block = nn.Sequential(
            pool_layer_batch(pool_neigh_orders,"mean"),
            onering_conv_layer_batch(in_ch, out_ch, neigh_orders),
            nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.weight = nn.Conv1d(out_ch*2,out_ch,kernel_size=1)
    def forward(self, x,x1):
        out_features = self.block(x)
        out_features = torch.cat((out_features,x1),dim=1)
        out_features = self.weight(out_features)
        return out_features
class up_block(nn.Module):
    def __init__(self,in_features,out_features,upconv_center_indices,upconv_edge_indices,neigh_orders):
        super(up_block,self).__init__()
        self.up = upconv_layer(in_features,out_features,upconv_center_indices,upconv_edge_indices)
        self.block = nn.Sequential(
            onering_conv_layer(out_features*2,out_features,neigh_orders=neigh_orders),
            nn.BatchNorm1d(out_features, momentum=0.15, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            onering_conv_layer(out_features, out_features, neigh_orders=neigh_orders),
            nn.BatchNorm1d(out_features, momentum=0.15, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self,x1,x2):
        up = self.up(x1)
        x = torch.cat((up,x2),1)
        x = self.block(x)
        return x

class up_block_batch(nn.Module):
    def __init__(self,in_features,out_features,upconv_center_indices,upconv_edge_indices,neigh_orders):
        super(up_block_batch,self).__init__()
        self.up = upconv_layer_batch(in_features, out_features, upconv_center_indices, upconv_edge_indices)
        self.block = nn.Sequential(
            onering_conv_layer_batch(out_features * 2, out_features, neigh_orders=neigh_orders),
            nn.BatchNorm1d(out_features, momentum=0.15, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            onering_conv_layer_batch(out_features, out_features, neigh_orders=neigh_orders),
            nn.BatchNorm1d(out_features, momentum=0.15, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self,x1,x2):
        up = self.up(x1)
        x = torch.cat((up,x2),1)
        x = self.block(x)
        return x

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv1d(channel, channel // ratio,  kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // ratio, channel, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self,neigh_orders):
        super(SpatialAttentionModule,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,None))
        self.max_pool = nn.AdaptiveMaxPool2d((1,None))
        self.conv = onering_conv_layer_batch(2,1,neigh_orders)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        pool = torch.cat((self.avg_pool(x),self.max_pool(x)),dim=1)
        pool = self.conv(pool)
        spatialAttention = self.sigmoid(pool)
        return spatialAttention


class CBAM(nn.Module):
    def __init__(self, channel,neigh_orders):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule(neigh_orders)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out_age = (1-self.channel_attention(x))*x
        out_age = (1 - self.spatial_attention(out)) * out_age
        out = self.spatial_attention(out) * out

        return out,out_age

class Spherical_Attention_Block(nn.Module):
    def __init__(self):
        super(Spherical_Attention_Block,self).__init__()
        self.mean_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(3,1,kernel_size=1)
        self.sigmod = nn.Sigmoid()

    def forward(self,x_l,x_r):
        x = torch.concat((x_l,x_r),dim=2)
        channel_pool = self.mean_pool(x)
        attention = self.conv(channel_pool) * self.conv(x)
        attention = self.sigmod(attention)
        return attention

class Age_predictor(nn.Module):
    def __init__(self,in_features):
        super(Age_predictor,self).__init__()
        self.block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features,256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128,64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32,1)
        )
    def forward(self,x_l,x_r):
        return self.block(torch.concat((x_l,x_r),dim=2))


class Discriminator(nn.Module):
    def __init__(self,in_features):
        super(Discriminator,self).__init__()
        neigh_orders = utils.Get_neighs_order(0)
        indices = utils.Get_upconv_index(0)
        # self.block = nn.Sequential(
        #     down_block_batch(onering_conv_layer_batch,in_features,32,neigh_orders=neigh_orders[2],first=True),
        #     down_block_batch(onering_conv_layer_batch,32,64,neigh_orders=neigh_orders[3],pool_neigh_orders=neigh_orders[2]),
        #     down_block_batch(onering_conv_layer_batch,64,128,neigh_orders=neigh_orders[4],pool_neigh_orders=neigh_orders[3]),
        #     down_block_batch(onering_conv_layer_batch,128,256,neigh_orders=neigh_orders[5],pool_neigh_orders=neigh_orders[4]),
        #     down_block_batch(onering_conv_layer_batch,256,512,neigh_orders=neigh_orders[6],pool_neigh_orders=neigh_orders[5]),
        # )
        # self.pred = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(512*42*2, 256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(256,128),
        #     nn.LeakyReLU(0.2,inplace=True),
        #     nn.Linear(128,64),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(64,2),
        #     nn.Softmax(dim=0)
        # )
        self.mean_pool = nn.AdaptiveAvgPool1d(1)
        self.block = nn.Conv1d(in_channels=3,out_channels=2,kernel_size=1)
        self.softmax = nn.Softmax(dim=0)
    def forward(self,x_l,x_r):
        x = torch.cat((x_l,x_r),dim=2)
        x = self.mean_pool(x)
        x = self.block(x)
        return torch.squeeze(self.softmax(x), dim=-1)

def load_data(csv_path):
    data = pd.read_csv(csv_path)
    # 假设数据有 features 和 labels 两个列，你需要根据实际情况调整
    features = data.drop('labels', axis=1).values
    labels = data['labels'].values
    return features, labels

def correlation_loss(source,recons):
    source_mean = torch.mean(source,dim=2,keepdim=True)
    recons_mean = torch.mean(source,dim=2,keepdim=True)
    numerator = torch.sum((source - source_mean) * (recons - recons_mean),dim=2)
    denominator = torch.sqrt(torch.sum((source - source_mean)**2,dim=2) * torch.sum((recons - recons_mean)**2,dim=2))
    return torch.sum(numerator/denominator)

def contrast_loss(attention_map,x_recon_l,x_recon_r,targets):
    x_recon_positive = attention_map[targets[:, 1] == 1, :, :] * torch.concat((x_recon_l, x_recon_r), dim=2)[
                                                                 targets[:, 1] == 1, :, :]
    x_recon_wei_positive = (1 - attention_map[targets[:, 1] == 1, :, :]) * torch.concat((x_recon_l, x_recon_r), dim=2)[
                                                                           targets[:, 1] == 1, :, :]
    x_recon_negative = attention_map[targets[:, 1] == 0, :, :] * torch.concat((x_recon_l, x_recon_r), dim=2)[
                                                                 targets[:, 1] == 0, :, :]

    loss_1 = torch.sum(x_recon_positive-x_recon_wei_positive)
    x_recon_positive_temp = x_recon_positive.repeat(x_recon_negative.shape[0],1,1)
    x_recon_wei_positive_temp = x_recon_wei_positive.repeat(x_recon_negative.shape[0],1,1)
    x_recon_negative_positive = x_recon_negative.repeat(1,x_recon_positive.shape[0],1).reshape([x_recon_positive.shape[0]*x_recon_negative.shape[0],3,20484])
    x_recon_negative_wei_positive = x_recon_negative.repeat(1,x_recon_wei_positive.shape[0],1).reshape([x_recon_negative.shape[0]*x_recon_wei_positive.shape[0],3,20484])
    loss_2 = torch.sum(x_recon_wei_positive_temp-x_recon_negative_wei_positive)
    loss_3 = torch.sum(x_recon_negative_positive-x_recon_positive_temp)
    if 1-loss_1 > 0:
        loss_1 = 1-loss_1
    else:
        loss_1=0
    if 1 - loss_3 > 0:
        loss_3 = 1 - loss_3
    else:
        loss_3 = 0
    return loss_1+loss_2+loss_3

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        # 如果是卷积层或线性层，使用Kaiming初始化
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            # 初始化偏置
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        # 如果是批归一化层，使用标准正态分布初始化
        nn.init.normal_(m.weight, mean=0, std=1)
        nn.init.zeros_(m.bias)

class sperical_unet(nn.Module):
    def __init__(self,in_features):
        super(sperical_unet,self).__init__()
        neigh_orders = utils.Get_neighs_order(0)
        indices = utils.Get_upconv_index(0)
        self.down = nn.ModuleList([])
        self.down.append(
            down_block_batch(onering_conv_layer_batch, in_features, 32, neigh_orders=neigh_orders[2], first=True))
        self.down.append(down_block_batch(onering_conv_layer_batch, 32, 64, neigh_orders=neigh_orders[3],
                                          pool_neigh_orders=neigh_orders[2], first=False))
        self.down.append(down_block_batch(onering_conv_layer_batch, 64, 128, neigh_orders=neigh_orders[4],
                                          pool_neigh_orders=neigh_orders[3], first=False))
        self.down.append(down_block_batch(onering_conv_layer_batch, 128, 256, neigh_orders=neigh_orders[5],
                                          pool_neigh_orders=neigh_orders[4], first=False))



        self.up = nn.ModuleList([])
        self.up.append(up_block_batch(256,128,indices[-4],indices[-3],neigh_orders=neigh_orders[4]))
        self.up.append(up_block_batch(128,64,indices[-6],indices[-5],neigh_orders=neigh_orders[3]))
        self.up.append(up_block_batch(64,32,indices[-8],indices[-7],neigh_orders=neigh_orders[2]))
        self.outc = nn.Conv1d(32,in_features,kernel_size=1)
        # self.attention = Spherical_Attention_Block()
        # self.age_prediction = Age_predictor(256*324)
        # self.discriminator = Discriminator(in_features)
    def forward(self,x_l,x_r):
        # left head
        xs_l = [x_l]
        out_age_l=[]
        for i in range(4):
            if i == 0:
                out = self.down[i](xs_l[i])
                # out, out_age_1 = self.CBAM[i](temp)
                xs_l.append(out)
                # out_age_l.append(out_age_1)
            else:
                out = self.down[i](xs_l[i])
                # out, out_age_1 = self.CBAM[i](temp)
                # out_age_1 = self.hierarchical[i-1](out_age_l[i-1],out_age_1)
                xs_l.append(out)
                # out_age_l.append(out_age_1)

        xt_l = xs_l[-1]
        for i in range(3):
            xs_l.append(self.up[i](xt_l,xs_l[3-i]))
            xt_l = xs_l[-1]
        x_recon_l = self.outc(xt_l)
        ##  right
        xs_r = [x_r]
        out_age_r = []
        for i in range(4):
            if i == 0:
                out = self.down[i](xs_r[i])
                # out, out_age_1 = self.CBAM[i](temp)
                xs_r.append(out)
                # out_age_r.append(out_age_1)
            else:
                out = self.down[i](xs_r[i])
                # out, out_age_1 = self.CBAM[i](temp)
                # out_age_1 = self.hierarchical[i - 1](out_age_r[i - 1], out_age_1)
                xs_r.append(out)
                # out_age_r.append(out_age_1)

        xt_r = xs_r[-1]
        for i in range(3):
            xs_r.append(self.up[i](xt_r, xs_r[3 - i]))
            xt_r = xs_r[-1]
        x_recon_r = self.outc(xt_r)

        # attention_map = self.attention(x_recon_l,x_recon_r)
        # pred = self.discriminator(x_recon_l*attention_map[:,:,:10242],x_recon_r*attention_map[:,:,10242:])
        # age_pred = self.age_prediction(out_age_l[-1],out_age_r[-1])
        return x_recon_l,x_recon_r

final_1=0
final_2=0
final_3=0

# def count_parameters(model):
#     # 统计模型中每个子模块的参数数量
#     params_count = [(name, p.numel()) for name, p in model.named_parameters()]
#
#     # 排序子模块，按照参数数量降序排序
#     params_count.sort(key=lambda x: x[1], reverse=True)
#
#     return params_count
# model = sperical_unet(3)
# # 统计模型的参数数量
# params_count = count_parameters(model)
#
# # 打印每个子模块的参数数量
# for name, count in params_count:
#     print(f"{name}: {count} parameters")
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total Parameters: {total_params}")

#获取数据集
def split_dataset(dataset_addresses, train_ratio):
    # 设置随机种子，以确保每次运行程序时划分结果一致
    random.seed(42)

    # 随机打乱数据集地址列表
    random.shuffle(dataset_addresses)

    # 划分训练集和测试集的索引
    split_index = int(train_ratio * len(dataset_addresses))

    train_addresses = dataset_addresses[:split_index]
    val_addresses = dataset_addresses[split_index:]

    return train_addresses, val_addresses

def make_dataset(directory):
    directory = os.path.abspath(directory)

    # 使用glob模块匹配所有.mgh文件
    mgh_files = glob.glob(os.path.join(directory, '*.mgh'))
    train_files,val_files = split_dataset(mgh_files,0.9)
    return train_files,val_files
class DatasetFolder(VisionDataset):


    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],

    ) -> None:
        super(DatasetFolder, self).__init__(root)

        samples,vals = make_dataset(self.root)

        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            # if extensions is not None:
            #     msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader

        self.samples = samples


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        while True:
            try:
                path = self.samples[index]
                sample = self.loader(path)
                #sample = sample.double()
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.samples) - 1)


        return sample#, target

    def __len__(self) -> int:
        return len(self.samples)


class ValFolder(VisionDataset):


    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
    ) -> None:
        super(ValFolder, self).__init__(root)

        samples,vals = make_dataset(self.root)

        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)

            raise RuntimeError(msg)

        self.loader = loader

        self.vals = vals


    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        while True:
            try:
                path = self.vals[index]
                sample = self.loader(path)
                #sample = sample.double()
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.vals) - 1)

        return sample

    def __len__(self) -> int:
        return len(self.vals)





def default_loader(path: str):
    data = nib.load(path)
    data = data.get_fdata()
    data = torch.tensor(data)
    data = data.to(torch.float)
    #data = data.double()
#把球面卷积加入进来 /观察下结构
    return data

#需要修改


class SurfaceFolder(DatasetFolder):


    def __init__(
            self,
            root: str,

            loader: Callable[[str], Any] = default_loader,

    ):
        super(SurfaceFolder, self).__init__(root, loader
                                          )
        self.surfaces = self.samples


class Val_dataset(ValFolder):

    def __init__(
            self,
            root: str,

            loader: Callable[[str], Any] = default_loader,

    ):
        super(Val_dataset, self).__init__(root, loader
                                          )
        self.surfaces = self.vals
        #List of FMRI surfaces

def build_pretraining_dataset(data_path):
    return SurfaceFolder(data_path)
def build_Val_dataset(data_path):
    return Val_dataset(data_path)


def record_loss_to_txt(losses, file_path):
    with open(file_path, 'w') as file:
        for epoch, loss in enumerate(losses):
            file.write(f"Epoch {epoch + 1}: {loss}\n")

if __name__ == '__main__':
    # 超参数

    learning_rate = 0.001
    epochs = 100
    batch_size = 100
    #num_classes = 2
    num_workers = 20
    # 加载数据
    #csv_path = './data/features_10242.csv'
    data_path = '/media/amax/Cui/NSD/data_01'

    # features, labels = load_data(csv_path)
    #features = pd.read_csv(csv_path,header=None).values
    #features = StandardScaler().fit_transform(features).reshape([998,3,20484])
    #labels = pd.read_csv("./data/cov.csv",header=None).values[:,1:]

    # get dataset
    dataset_train = build_pretraining_dataset(data_path)
    dataset_val = build_Val_dataset(data_path)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.RandomSampler(dataset_val)

    # 转换为 PyTorch 的 Tensor
    #features = torch.tensor(features, dtype=torch.float32)
    # print(labels[:,0].astype("int"))
    # classifity_labels = F.one_hot(torch.Tensor(labels[:,0].astype("int")).to(torch.int64), num_classes).float()
    # age_labels = labels[:,1].astype("float")
    # labels = torch.Tensor(np.hstack([classifity_labels,age_labels.reshape([-1,1])])).float()
    # 创建 DataLoader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )

    # 初始化模型
    # print(5)
    # 定义损失函数和优化器
    # criterion_1= nn.CrossEntropyLoss()
    L2_loss = nn.MSELoss(reduction="sum")

    if torch.cuda.is_available():
        device = torch.device("cuda:3")
        print("CUDA is available! Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    # 使用 5 折交叉验证
    #kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 训练过程
    best_accuracy = 9999999999
    best_model_params = None
    # for lambda_1 in np.arange(0.1,1,0.1):
    #     for lambda_2 in np.arange(0.1,1,0.1):
    #         for lambda_3 in np.arange(0.1,1,0.1):

    #for fold, (train_indices, val_indices) in enumerate(kf.split(features)):
    model = sperical_unet(1)
    model = model.to(device)
    model.apply(weight_init)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Total Parameters: {total_params}")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 划分训练集和验证集
    # train_features, train_labels = features[train_indices], labels[train_indices]
    # val_features, val_labels = features[val_indices], labels[val_indices]
    # 创建 DataLoader
    # train_dataset = TensorDataset(train_features, train_labels)
    # val_dataset = TensorDataset(val_features, val_labels)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 训练模型
    losses = []
    for epoch in range(epochs):
        model.train()
        for surfaces in data_loader_train:
            surfaces = surfaces.to(device)
            batch_size = surfaces.shape[0]
            surfaces = surfaces.reshape(batch_size,1,-1)
            x_l = surfaces[:, :, :10242]
            x_r = surfaces[:, :, 163842:174084]
            surfaces = torch.cat((x_l, x_r), dim=-1)

            # Z-Score
            mean = surfaces.mean(dim=-1, keepdim=True)
            std = surfaces.std(dim=-1, keepdim=True)
            surfaces = (surfaces - mean) / std

            # 前向传播
            x_l = surfaces[:,:,:10242]
            x_r = surfaces[:,:,10242:]

            x_recon_l,x_recon_r = model(x_l,x_r)
            #print(x_recon_r.shape)
            recons_loss_1 = L2_loss(x_recon_l,x_l)
            recons_loss_2 = L2_loss(x_recon_r,x_r)
            # contrast_loss_1 = contrast_loss(attention_map,x_recon_l,x_recon_r,targets)
            loss = recons_loss_1+recons_loss_2 #-lambda_1*corr_loss+lambda_2*entropy_loss+lambda_3*contrast_loss_1
            losses.append(loss)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(f'lambda_1_{lambda_1}_lambda_2_{lambda_2}_lambda_3_{lambda_3}',loss.item(),f'age_loss:{age_loss}')
        print(f'Epoch {epoch}, loss：',loss.item())
        # 验证模型
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            #loss_temp = 0
            for surfaces in data_loader_val:
                surfaces = surfaces.to(device)
                batch_size = surfaces.shape[0]
                surfaces = surfaces.reshape(batch_size, 1, -1)
                x_l = surfaces[:, :, :10242]
                x_r = surfaces[:, :, 163842:174084]
                surfaces = torch.cat((x_l, x_r), dim=-1)

                # Z-Score
                mean = surfaces.mean(dim=-1, keepdim=True)
                std = surfaces.std(dim=-1, keepdim=True)
                surfaces = (surfaces - mean) / std

                x_l = surfaces[:, :, :10242]
                x_r = surfaces[:, :, 10242:]

                recons_l, recons_r = model(x_l,x_r)
                recons_loss_1 = L2_loss(recons_l, x_l)
                recons_loss_2 = L2_loss(recons_r, x_r)
                loss_val = recons_loss_1 + recons_loss_2
        # print(f'Fold {fold + 1}, Epoch {epoch + 1}, Validation Accuracy: {accuracy}, lambda_1:{lambda_1},lambda_2:{lambda_2},lambda_3:{lambda_3}')
        print(f'Epoch {epoch+1}, Validation Accuracy: {loss_val}')
        # 保存最好的模型参数
        if loss_val < best_accuracy:
            best_accuracy = loss_val
            best_model_params = model.state_dict()

    # 保存最好的模型参数
    # torch.save(best_model_params, f'lambda_1_{lambda_1}_lambda_2_{lambda_2}_lambda_3_{lambda_3}_best_model_params.pth')
    torch.save(best_model_params, f'./results/U-Net/best_model_params.pth')

    #保存Loss
    file_path = "./results/U-Net/loss_record.txt"
    record_loss_to_txt(losses, file_path)
