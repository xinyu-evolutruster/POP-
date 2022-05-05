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
        
    def forward(self, posmap, geometry_feature_map, uv_location, pq_coords):
        B = posmap.shape[0]
        # print(posmap.shape)
        posmap = posmap.permute([0, 2, 1])
 
        # pose_feature_map = self.unet_pose_feature(posmap)
        pose_feature_map = self.pointnet_pose_feature(posmap)
        # print("pose_feature_map shape: {}".format(pose_feature_map.shape))
        # print("geometry_feature_map shape: {}".format(geometry_feature_map.shape))

        pose_feature_map = pose_feature_map.permute([0, 2, 1])
        pixel_feature = torch.cat([pose_feature_map, geometry_feature_map], dim=1)

        input_feature = pixel_feature

        residuals, normals = self.decoder(input_feature)
        # print("residuals shape: {}".format(residuals.shape))
        # print("normals shape: {}".format(normals.shape))

        residuals = residuals.permute([0, 2, 1])
        normals = normals.permute([0, 2, 1])

        return residuals, normals