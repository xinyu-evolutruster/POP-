import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils_model import uv_to_grid
# from myPOP.lib.network.modules import UNet5DS, UNet6DS, UNet7DS, ShapeDecoder
from lib.network.pointnet_modules import PointNet, ShapeDecoder

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
        self.pointnet_pose_feature = PointNet(input_channel=input_channel, num_classes=pose_feature_channel)
        decoder_input_dimension = pose_feature_channel + geometry_feature_channel 
        self.decoder = ShapeDecoder(decoder_input_dimension, hidden_size=hidden_size)
        
    def forward(self, posmap, geometry_feature_map, uv_location, pq_coords, valid_idx):
        batch = posmap.shape[0]
        posmap = posmap.reshape(batch, -1, 3) 
        posmap = posmap[:, valid_idx, :].permute([0, 2, 1])
        # print("posmap shape is {}".format(posmap.shape))

        # pose_feature_map = self.unet_pose_feature(posmap)
        pose_feature_map = self.pointnet_pose_feature(posmap)
        # print("pose_feature_map shape: {}".format(pose_feature_map.shape))
        
        B, C, H, W = geometry_feature_map.shape
        new_feature_map = geometry_feature_map.new_zeros((B, H * W, C))
        # print("new feature map shape: {}".format(new_feature_map.shape))
        
        new_feature_map[:, valid_idx,:] = pose_feature_map
        pose_feature_map = new_feature_map.reshape(B, H, W, C).permute([0, 3, 1, 2])
        # print("pose_feature_map shape: {}".format(pose_feature_map.shape))

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
        # input_feature = torch.cat([pixel_feature, uv_location, pq_coords], dim=1)
        input_feature = pixel_feature

        residuals, normals = self.decoder(input_feature)
        residuals = residuals.view(B, 3, H, W, N_samples)
        normals = normals.view(B, 3, H, W, N_samples)

        return residuals, normals