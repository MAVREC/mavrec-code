# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from torchvision.utils import save_image
from ..functions import MSDeformAttnFunction
import numpy as np



from PIL import Image
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Patch




def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0

color = np.array([1, 0.0, 0.0])

def plot_sample_location_all_levels(image_dir, all_levels_sample_location_points, all_levels_attention_weights, spatial_shapes, query_location, save_path):
    image = Image.open(image_dir)
    aspect_ratio = image.size[0] / image.size[1]  # width / height
    fig, ax = plt.subplots(figsize=(8, 8 / aspect_ratio))  # Adjust the size accordingly
    ax.imshow(image, aspect='auto')
    h, w = image.size

    colors = [np.random.rand(3,) for _ in range(len(all_levels_sample_location_points))]

    # Plot the green plus for the query location
    query_x = query_location[0] * w
    query_y = query_location[1] * h
    ax.plot(query_x, query_y, 'g+', markersize=10)

    for level, (sample_location_points, attention_weights) in enumerate(zip(all_levels_sample_location_points, all_levels_attention_weights)):
        h_, w_ = spatial_shapes[level]
        color = colors[level]  # Use a distinct color for each level

        for point, weight in zip(sample_location_points, attention_weights):
            x = point[0] * h_ * (h / h_)
            y = point[1] * w_ * (w / w_)
            plt.plot(x.item(), y.item(), 'o', color=weight.item() * color)

    plt.savefig(save_path)  # Save the image
    plt.show()


#   for i in range(900):
#     x = sample_location_point[indices[i]][0]*h_*(h/h_)
#     y = sample_location_point[indices[i]][1]*w_*(w/w_)
#     #print ('attention_weight[i]',attention_weight[i])
#     plt.plot(x.item(), y.item(), 'o', color=sorted[i].item() * color);
#     legend_elements = [Patch(facecolor=color, edgecolor=color, label='High Attention'),
#                        Patch(facecolor=color * 0.5, edgecolor=color * 0.5, label='Medium Attention'),
#                        Patch(facecolor=color * 0.1, edgecolor=color * 0.1, label='Low Attention')]
#     #plt.legend(handles=legend_elements, loc='upper right', title='Attention Weights')
#     ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.05), loc='upper center', title='Attention Weights')

#     plt.subplots_adjust(bottom=0.2) 
#     plt.show()
#   plt.savefig('abcd.png')

class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")


        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.sampling_locationss = False 
        

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def get_sampling_location () :
      return self.sampling_locationss

    def set_sampling_location (sampling_location) :
      sampling_locationss.append(sampling_location)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        print ("input_padding_mask", input_padding_mask.shape, input_padding_mask)


        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)

        print ('sampling_offsets' , sampling_offsets.shape)




        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        print ('attention_weights' , attention_weights.shape)


        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        print (' after softmax, attention_weights' , attention_weights.shape)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        
        print (sampling_locations[0,0,0,0,])

        point = []
        attention_weight = []

        #level = 1

        # In the forward method of MSDeformAttn class

        # Initialize lists to store points and weights for all levels
        all_levels_points = []
        all_levels_weights = []

        # Assume you are interested in the first query for simplicity
        query_idx = 0

        # Extract the query location (assuming the center of the reference point box)
        query_location = reference_points[query_idx, 0, 0, :2].cpu().numpy()
        if reference_points.shape[-1] == 4:
            query_location += reference_points[query_idx, 0, 0, 2:4].cpu().numpy() * 0.5

        # Iterate over levels and collect sampling points and attention weights
        for level in range(self.n_levels):
            level_points = []
            level_weights = []

            for i in range(Len_q):  # Iterate over all queries
                for head in range(self.n_heads):
                    point = sampling_locations[query_idx, i, head, level, 0, :].cpu()
                    weight = attention_weights[query_idx, i, head, level, 0]
                    level_points.append(point)
                    level_weights.append(weight)

            all_levels_points.append(level_points)
            all_levels_weights.append(level_weights)

        # Call the plotting function with the collected data
        plot_sample_location_all_levels(
            '/data/SDU-GAMODv4/SDU-GAMODv4/scaled_dataset/val/droneview/scene_1_Cortex_Park_crossroads_drone_000861.PNG',  # Replace with your image path
            all_levels_points,
            all_levels_weights,
            input_spatial_shapes,
            query_location,
            'output_visualization.png'  # Save path for the image
        )

        print ('input_flatten.shape', input_flatten.shape)
        print ("input_spatial_shapes", input_spatial_shapes[0])
        print ('sampling_locations', sampling_locations.shape)

        #plot_sample_location_point('/data/SDU-GAMODv4/SDU-GAMODv4/scaled_dataset/val/droneview/scene_1_Cortex_Park_crossroads_drone_000861.PNG', point, input_spatial_shapes[level], attention_weight)

        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output
