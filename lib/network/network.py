import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils_model import uv_to_grid
from lib.network.modules import UNet5DS, UNet6DS, UNet7DS, ShapeDecoder

class Network(nn.Module):
    def __init__(
                self,
                input_channel=3,
                pose_feature_channel=64,
                geometry_feature_channel=64,
                posmap_size=128,
                uv_feature_dimension=2,
                pq_feature_dimension=2,
                hidden_size=128
                ):
        super(Network, self).__init__()
        self.posmap_size = posmap_size

        unets = {32: UNet5DS, 64: UNet6DS, 128: UNet7DS, 256: UNet7DS}
        # the pose encoder
        unet_loaded = unets[self.posmap_size]
        self.unet_pose_feature = unet_loaded(input_channel, pose_feature_channel)

        decoder_input_dimension = uv_feature_dimension + pq_feature_dimension + \
                                  pose_feature_channel + geometry_feature_channel 
        
        print("hidden size: {}".format(hidden_size))
        self.decoder = ShapeDecoder(decoder_input_dimension, hidden_size=hidden_size)

    def forward(self, posmap, geometry_feature_map, uv_location, pq_coords):
        # print("x shape: {}".format(posmap.shape))
        pose_feature_map = self.unet_pose_feature(posmap)

        pixel_feature = torch.cat([pose_feature_map, geometry_feature_map], dim=1)

        feature_resolution = pose_feature_map.shape[2]    # 128
        uv_resolution = int(uv_location.shape[1] ** 0.5)  # 256

        # bilinear interpolation
        if feature_resolution != uv_resolution:
            query_grid = uv_to_grid(uv_location, uv_resolution)
            pixel_feature = F.grid_sample(pixel_feature, query_grid, mode='bilinear', align_corners=False)

        B, C, H, W = pixel_feature.shape
        N_samples = 1

        uv_feature_dim = uv_location.shape[-1]  # 2
        uv_location = uv_location.view(B, -1, uv_feature_dim).permute([0, 2, 1])
        pq_coords = pq_coords.view(B, -1, 2).permute([0, 2, 1])

        # pixel_feature = pixel_feature.view(B, C, -1).expand(N_samples, -1, -1, -1).permute([1, 2, 3, 0])
        pixel_feature = pixel_feature.reshape(B, C, -1)
        # [4, 132, 65536]
        input_feature = torch.cat([pixel_feature, uv_location, pq_coords], dim=1)

        residuals, normals = self.decoder(input_feature)

        residuals = residuals.view(B, 3, H, W, N_samples)
        normals = normals.view(B, 3, H, W, N_samples)

        return residuals, normals