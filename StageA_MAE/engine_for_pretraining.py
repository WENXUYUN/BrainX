# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from util.layers import onering_conv_layer_batch,onering_conv_layer
from util.utils import Get_neighs_order
# neigh_orders_163842, neigh_orders_40962, neigh_orders_10242, \
#         neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
# conv = onering_conv_layer_batch(1,neigh_orders_163842)
# conv2 = onering_conv_layer(1,1,neigh_orders_163842)



def percentile_clipping(data, percentile=1, replace_with=0): 
    temp = data.view(-1)
    lower_bound = temp.kthvalue(int(len(temp) * percentile / 100)).values
    upper_bound = temp.kthvalue(int(len(temp) * (100 - percentile) / 100)).values
    clipped_data = torch.where(temp < lower_bound, replace_with, temp)
    clipped_data = torch.where(temp > upper_bound, replace_with, clipped_data)
    clipped_data = clipped_data.view(-1,327684,1)
    return clipped_data
# 定义OHEM损失函数
class OHMELoss(nn.Module):
    def __init__(self, ratio=2):
        super(OHMELoss, self).__init__()
        self.ratio = ratio

    def forward(self, input, target):
        loss = nn.functional.cross_entropy(input, target, reduction='none')
        #loss = nn.MSELoss(input,target,reduction='mean')
        loss = (input - target).pow(2)  # 计算均方误差
        #loss = torch.abs(input - target)
        loss = torch.mean(loss,dim=1,keepdim=True)
        num_samples = loss.size(0)
        #loss = nn.L1Loss()
        num_samples = len(loss)
        num_hard_samples = int(num_samples // self.ratio)
        sorted_loss, indices = torch.sort(loss, descending=True, dim=0)
        topk_loss = sorted_loss[:min(num_hard_samples, num_samples)]
        ohem_loss = torch.mean(topk_loss)
        return ohem_loss

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True,log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        #需要修改
        surfaces, bool_masked_pos = batch
        #print(bool_masked_pos.shape)
        surfaces = surfaces.to(device, non_blocking=True)
        #surfaces = surfaces.reshape(-1,327684,1)

       

        # Z变化：只保留绝对值大于1的，保留真正的有效的激活
        # surfaces[torch.abs(surfaces) <= 1] = 0

        # MIN-MAX
        # input_min = surfaces.min(dim=1, keepdim=True)[0]
        # input_max = surfaces.max(dim=1, keepdim=True)[0]
        #surfaces = 20 *(surfaces - min_value) / (max_value - min_value) - 10

	
        # Z-Score
        #mean = surfaces.mean(dim=1, keepdim=True)
        #std = surfaces.std(dim=1, keepdim=True)
        #surfaces = (surfaces - mean) / var

	# 百分比裁剪
        #surfaces = percentile_clipping(surfaces)
        #print("裁剪后的数据范围：", torch.min(clipped_batch_data), torch.max(clipped_batch_data))
        
        
        surfaces_l = surfaces[:,:,:1]
        surfaces_r = surfaces[:,:,1:]
        #surfaces = torch.cat((surfaces_l,surfaces_r),dim=-1)
        #sigmoid
        #surfaces = torch.sigmoid(surfaces)

        #surfaces = surfaces.double()
        #surfaces = conv(surfaces)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        #print(bool_masked_pos.shape)
        #需要修改
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # calculate the predict label
            #需要修改
            # mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
            # std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
            # unnorm_surfaces = surfaces * std + mean  # in [0, 1]


            #需要修改
            # if normlize_target:
            #     surfaces_squeeze = rearrange(unnorm_surfaces, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
            #     surfaces_norm = (surfaces_squeeze - surfaces_squeeze.mean(dim=-2, keepdim=True)
            #         ) / (surfaces_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
            #     # we find that the mean is about 0.48 and standard deviation is about 0.08.
            #     surfaces_patch = rearrange(surfaces_norm, 'b n p c -> b n (p c)')
            # else:
            #     surfaces_patch = rearrange(unnorm_surfaces, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
            model = model.to(torch.float32)
            surfaces = surfaces.to(torch.float32)
            labels = surfaces

        #loss_func = OHMELoss()
        #loss_func = nn.L1Loss(reduction="mean")
        loss_func = nn.MSELoss(reduction='mean')
        with torch.cuda.amp.autocast():
            #surfaces = surfaces.cuda()
            outputs = model(surfaces,bool_masked_pos)
            #outputs = outputs.to(torch.float)
            #labels = labels.to(torch.float32)
            #epsilon = 1e-12
            #loss = torch.mean(torch.abs((labels - outputs) / (labels+ epsilon))) * 100
            loss = loss_func(outputs, labels)
            loss = loss.float()
            loss = torch.nan_to_num(loss)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        #loss = loss.float()
        #model_parameters_float = [param.to(torch.float32) for param in model.parameters()]
        # for param in model.parameters():
        #     param.data = param.data.to(torch.float32)
        #     if param.grad is not None:
        #         param.grad = param.grad.to(torch.float32)
        #
        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if torch.is_tensor(v):
        #             state[k] = v.to(torch.float32)


        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)

        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

