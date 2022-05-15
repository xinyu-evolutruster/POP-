import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils_model import uv_to_grid
from lib.network.pointnet_modules import PointNet, ShapeDecoder, SmallShapeDecoder, FeatureExpansion
from lib.network.modules import UNet5DS1d, GeomConvLayers1d, GeomConvBottleneckLayers1d, GaussianSmoothingLayers1d

class Network(nn.Module):
    def __init__(
                self,
                input_channel=3,
                pose_feature_channel=64,
                geometry_feature_channel=64,
                posmap_size=128,
                geom_layer_type='conv',
                gaussian_kernel_size = 3,
                uv_feature_dimension=2,
                pq_feature_dimension=2,
                hidden_size=128,
                repeat=6
                ):
        super(Network, self).__init__()
        self.posmap_size = posmap_size
        self.pose_feature_channel = pose_feature_channel
        self.geometry_feature_channel = geometry_feature_channel

        self.pointnet_pose_feature = PointNet(input_channel=input_channel, num_classes=pose_feature_channel)
        
        c_geom = geometry_feature_channel
        geom_proc_layers = {
            'unet': UNet5DS1d(c_geom, c_geom, hidden_size), # use a unet
            'conv': GeomConvLayers1d(c_geom, c_geom, c_geom, use_relu=False), # use 3 trainable conv layers
            'bottleneck': GeomConvBottleneckLayers1d(c_geom, c_geom, c_geom, use_relu=False), # use 3 trainable conv layers
            'gaussian': GaussianSmoothingLayers1d(channels=c_geom, kernel_size=gaussian_kernel_size, sigma=1.0), # use a fixed gaussian smoother
        }
        # optional layer for spatially smoothing the geometric feature tensor
        self.geom_layer_type = geom_layer_type 
        if geom_layer_type is not None:
            self.geom_proc_layers = geom_proc_layers[geom_layer_type]

        # temporarily set to 4
        self.repeat = repeat
        decoder_input_dimension = pose_feature_channel + geometry_feature_channel 
        # self.feature_expansion = FeatureExpansion(in_size=64, r=self.repeat)
        self.feature_expansion = FeatureExpansion(
            in_size=pose_feature_channel, 
            r=self.repeat
        )

        self.decoder = ShapeDecoder(decoder_input_dimension, hidden_size=hidden_size)
        
    def forward(self, posmap, geometry_feature_map, uv_location, pq_coords):
        B = posmap.shape[0]
        # print(posmap.shape)
        posmap = posmap.permute([0, 2, 1])
 
        # geometric feature tensor
        # if self.geom_layer_type is not None:
        #     geometry_feature_map = self.geom_proc_layers(geometry_feature_map)

        pose_feature_map = self.pointnet_pose_feature(posmap)
        pose_feature_map = pose_feature_map.permute([0, 2, 1])
        
        # print("pose_feature_map shape: {}".format(pose_feature_map.shape))
        # print("geometry_feature_map shape: {}".format(geometry_feature_map.shape))

        pose_feature_map = self.feature_expansion(pose_feature_map).permute([0, 1, 3, 2])
        # new_feature_map = torch.cat([pose_feature_map, geometry_feature_map], dim=1)
        # pose_feature_map = self.feature_expansion(new_feature_map).permute([0, 1, 3, 2])

        # print("pose_feature_map shape: {}".format(pose_feature_map.shape))
        
        pose_feature_map = pose_feature_map.reshape(B, -1, self.pose_feature_channel)
        # print("pose_feature_map shape: {}".format(pose_feature_map.shape))
        
        geometry_feature_map = geometry_feature_map.unsqueeze(1).repeat(1, self.repeat, 1, 1)
        geometry_feature_map = geometry_feature_map.permute([0, 1, 3, 2])
        # print("geometry_feature_map shape: {}".format(geometry_feature_map.shape))

        geometry_feature_map = geometry_feature_map.reshape(B, -1, self.geometry_feature_channel)
        # print("geometry_feature_map shape: {}".format(geometry_feature_map.shape))

        pixel_feature = torch.cat([pose_feature_map, geometry_feature_map], dim=2).permute([0, 2, 1])
        # print("pixel_feature shape: {}".format(pixel_feature.shape))

        input_feature = pixel_feature

        # input_feature: [B, C, N]
        residuals, normals = self.decoder(input_feature)
        # print("residuals shape: {}".format(residuals.shape))
        # print("normals shape: {}".format(normals.shape))

        residuals = residuals.permute([0, 2, 1])
        normals = normals.permute([0, 2, 1])

        return residuals, normals