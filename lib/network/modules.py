import torch
import torch.nn as nn

class Conv2DBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=4, stride=2, padding=1):
        super(Conv2DBlock, self).__init__()

        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(output_channel)
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.actvn(x)
        x = self.bn(x)
        return x

class UpConv2DBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=4, stride=2, padding=1):
        super(UpConv2DBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(output_channel, affine=False)
        self.drop = nn.Dropout(0.5)
        self.actvn = nn.ReLU()

    def forward(self, x, skip_input=None):
        x = self.actvn(x)
        x = self.upconv(x)
        x = self.bn(x)
        # x = self.drop(x)  # not use dropout for now
        if skip_input is not None:
            x = torch.cat([x, skip_input], dim=1)
        return x

class UNet5DS(nn.Module):
    def __init__(
                self, 
                input_channel=3, 
                output_channel=64,
                hidden_channel=64):
        super(UNet5DS, self).__init__()

        hid_c = hidden_channel
        self.conv1 = Conv2DBlock(input_channel, hid_c, kernel_size=4, stride=2, padding=1)
        self.conv2 = Conv2DBlock(hid_c * 1, hid_c * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = Conv2DBlock(hid_c * 2, hid_c * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = Conv2DBlock(hid_c * 4, hid_c * 8, kernel_size=4, stride=2, padding=1)
        self.conv5 = Conv2DBlock(hid_c * 8, hid_c * 8, kernel_size=4, stride=2, padding=1)

        self.upconv1 = UpConv2DBlock(hid_c * 8, hid_c * 8, kernel_size=4, stride=2, padding=1)
        self.upconv2 = UpConv2DBlock(2 * hid_c * 8, hid_c * 4, kernel_size=4, stride=2, padding=1)
        self.upconv3 = UpConv2DBlock(2 * hid_c * 4, hid_c * 2, kernel_size=4, stride=2, padding=1)
        self.upconv4 = UpConv2DBlock(2 * hid_c * 2, hid_c * 1, kernel_size=4, stride=2, padding=1)
        self.upconv5 = UpConv2DBlock(2 * hid_c * 1, output_channel, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv1(d1)
        d3 = self.conv2(d2)
        d4 = self.conv3(d3)
        d5 = self.conv4(d4)

        u1 = self.upconv1(d5, d4)
        u2 = self.upconv2(u1, d3)
        u3 = self.upconv3(u2, d2)
        u4 = self.upconv4(u3, d1)
        u5 = self.upconv5(u4)

        return u5

class UNet6DS(nn.Module):
    def __init__(
                self, 
                input_channel=3,
                output_channel=64,
                hidden_channel=64):
        super(UNet6DS, self).__init__()
        hid_c = hidden_channel
        self.conv1 = Conv2DBlock(input_channel, hid_c, kernel_size=4, stride=2, padding=1)
        self.conv2 = Conv2DBlock(hid_c * 1, hid_c * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = Conv2DBlock(hid_c * 2, hid_c * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = Conv2DBlock(hid_c * 4, hid_c * 8, kernel_size=4, stride=2, padding=1)
        self.conv5 = Conv2DBlock(hid_c * 8, hid_c * 8, kernel_size=4, stride=2, padding=1)
        self.conv6 = Conv2DBlock(hid_c * 8, hid_c * 8, kernel_size=4, stride=2, padding=1)

        self.upconv1 = UpConv2DBlock(hid_c * 8, hid_c * 8, kernel_size=4, stride=2, padding=1)
        self.upconv2 = UpConv2DBlock(2 * hid_c * 8, hid_c * 8, kernel_size=4, stride=2, padding=1)
        self.upconv3 = UpConv2DBlock(2 * hid_c * 8, hid_c * 8, kernel_size=4, stride=2, padding=1)
        self.upconv4 = UpConv2DBlock(3 * hid_c * 4, hid_c * 4, kernel_size=4, stride=2, padding=1)
        self.upconv5 = UpConv2DBlock(3 * hid_c * 2, hid_c * 3, kernel_size=4, stride=2, padding=1)
        self.upconv6 = UpConv2DBlock(3 * hid_c * 1, output_channel, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        d5 = self.conv5(d4)
        d6 = self.conv6(d5)

        u1 = self.upconv1(d6, d5)
        u2 = self.upconv2(u1, d4)
        u3 = self.upconv3(u2, d3)
        u4 = self.upconv4(u3, d2)
        u5 = self.upconv5(u4, d1)
        u6 = self.upconv6(u5)

        return u6

class UNet7DS(nn.Module):
    def __init__(
                self, 
                input_channel=3,
                output_channel=64,
                hidden_channel=64):
        super(UNet7DS, self).__init__()
        hid_c = hidden_channel
        self.conv1 = Conv2DBlock(input_channel, hid_c, kernel_size=4, stride=2, padding=1)
        self.conv2 = Conv2DBlock(hid_c * 1, hid_c * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = Conv2DBlock(hid_c * 2, hid_c * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = Conv2DBlock(hid_c * 4, hid_c * 8, kernel_size=4, stride=2, padding=1)
        self.conv5 = Conv2DBlock(hid_c * 8, hid_c * 8, kernel_size=4, stride=2, padding=1)
        self.conv6 = Conv2DBlock(hid_c * 8, hid_c * 8, kernel_size=4, stride=2, padding=1)
        self.conv7 = Conv2DBlock(hid_c * 8, hid_c * 8, kernel_size=4, stride=2, padding=1)

        self.upconv1 = UpConv2DBlock(hid_c * 8, hid_c * 8, kernel_size=4, stride=2, padding=1)
        self.upconv2 = UpConv2DBlock(2 * hid_c * 8, hid_c * 8, kernel_size=4, stride=2, padding=1)
        self.upconv3 = UpConv2DBlock(2 * hid_c * 8, hid_c * 8, kernel_size=4, stride=2, padding=1)
        self.upconv4 = UpConv2DBlock(2 * hid_c * 8, hid_c * 4, kernel_size=4, stride=2, padding=1)
        self.upconv5 = UpConv2DBlock(2 * hid_c * 4, hid_c * 2, kernel_size=4, stride=2, padding=1)
        self.upconv6 = UpConv2DBlock(2 * hid_c * 2, hid_c * 1, kernel_size=4, stride=2, padding=1)
        self.upconv7 = UpConv2DBlock(2 * hid_c * 1, output_channel, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        d5 = self.conv5(d4)
        d6 = self.conv6(d5)
        d7 = self.conv7(d6)

        u1 = self.upconv1(d7, d6)
        u2 = self.upconv2(u1, d5)
        u3 = self.upconv3(u2, d4)
        u4 = self.upconv4(u3, d3)
        u5 = self.upconv5(u4, d2)
        u6 = self.upconv6(u5, d1)
        u7 = self.upconv7(u6)

        return u7

class UnetNoCond5DS(nn.Module):
    # 5DS: downsample 5 times, for posmap size=32
    def __init__(self, input_nc=3, output_nc=3, nf=64, up_mode='upconv', use_dropout=False, 
                return_lowres=False, return_2branches=False):
        super().__init__()
        assert up_mode in ('upconv', 'upsample')

        self.return_lowres = return_lowres
        self.return_2branches = return_2branches

        self.conv1 = Conv2DBlock(input_nc, nf, 4, 2, 1, use_bias=False, use_bn=False, use_relu=False)
        self.conv2 = Conv2DBlock(1 * nf, 2 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv3 = Conv2DBlock(2 * nf, 4 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv4 = Conv2DBlock(4 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv5 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=False)

        self.upconv1 = UpConv2DBlock(8 * nf, 8 * nf, 4, 2, 1, up_mode=up_mode) #2x2, 512
        self.upconv2 = UpConv2DBlock(8 * nf * 2, 4 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 4x4, 512
        self.upconv3 = UpConv2DBlock(4 * nf * 2, 2 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 8x8, 512

        # Coord regressor
        self.upconv4 = UpConv2DBlock(2 * nf * 2, 1 * nf, 4, 2, 1, up_mode=up_mode) # 16
        self.upconv5 = UpConv2DBlock(1 * nf * 2, output_nc, 4, 2, 1, use_bn=False, use_bias=True, up_mode=up_mode) # 32

        if return_2branches:
            self.upconvN4 = UpConv2DBlock(2 * nf * 2, 1 * nf, 4, 2, 1, up_mode=up_mode) # 16
            self.upconvN5 = UpConv2DBlock(1 * nf * 2, output_nc, 4, 2, 1, use_bn=False, use_bias=True, up_mode='upconv') # 32

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        d5 = self.conv5(d4)

        u1 = self.upconv1(d5, d4)
        u2 = self.upconv2(u1, d3)
        u3 = self.upconv3(u2, d2)

        u4 = self.upconv4(u3, d1)
        u5 = self.upconv5(u4)

        if self.return_2branches:
            uN4 = self.upconvN4(u3, d1)
            uN5 = self.upconvN5(uN4)
            return u5, uN5

        return u5


class UnetNoCond6DS(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, nf=64, up_mode='upconv', use_dropout=False, return_lowres=False, return_2branches=False):
        super(UnetNoCond6DS, self).__init__()
        assert up_mode in ('upconv', 'upsample')

        self.return_lowres = return_lowres
        self.return_2branches = return_2branches

        self.conv1 = Conv2DBlock(input_nc, nf, 4, 2, 1, use_bias=False, use_bn=False, use_relu=False)
        self.conv2 = Conv2DBlock(1 * nf, 2 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv3 = Conv2DBlock(2 * nf, 4 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv4 = Conv2DBlock(4 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv5 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv6 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=False)

        self.upconv1 = UpConv2DBlock(8 * nf, 8 * nf, 4, 2, 1, up_mode=up_mode) #2x2, 512
        self.upconv2 = UpConv2DBlock(8 * nf * 2, 8 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 4x4, 512
        self.upconv3 = UpConv2DBlock(8 * nf * 2, 8 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 8x8, 512
        self.upconv4 = UpConv2DBlock(4 * nf * 3, 4 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 8x8, 512

        # Coord regressor
        self.upconvC5 = UpConv2DBlock(2 * nf * 3, 2 * nf, 4, 2, 1, up_mode='upsample') # 16
        self.upconvC6 = UpConv2DBlock(1 * nf * 3, output_nc, 4, 2, 1, use_bn=False, use_bias=True, up_mode='upsample') # 64x64, 128

        if return_2branches:
            # Normal regressor
            self.upconvN5 = UpConv2DBlock(2 * nf * 3, 2 * nf, 4, 2, 1, up_mode='upconv') # 32x32, 256
            self.upconvN6 = UpConv2DBlock(1 * nf * 3, 3, 4, 2, 1, use_bn=False, use_bias=True, up_mode='upconv') # 64x64, 128

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        d5 = self.conv5(d4)
        d6 = self.conv6(d5)

        # shared decoder layers
        u1 = self.upconv1(d6, d5)
        u2 = self.upconv2(u1, d4)
        u3 = self.upconv3(u2, d3)
        u4 = self.upconv4(u3, d2)

        # coord regressor
        uc5 = self.upconvC5(u4, d1)
        uc6 = self.upconvC6(uc5)

        if self.return_2branches:
            # normal regressor
            un5 = self.upconvN5(u4, d1)
            un6 = self.upconvN6(un5)

            return uc6, un6

        return uc6


class UnetNoCond7DS(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, nf=64, up_mode='upconv', use_dropout=False, return_lowres=False, return_2branches=False):
        super(UnetNoCond7DS, self).__init__()
        assert up_mode in ('upconv', 'upsample')

        self.return_lowres = return_lowres
        self.return_2branches = return_2branches

        # self.conv1 = Conv2DBlock(input_nc, nf, 4, 2, 1, use_bias=False, use_bn=False, use_relu=False)
        self.conv1 = Conv2DBlock(input_nc, nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv2 = Conv2DBlock(1 * nf, 2 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv3 = Conv2DBlock(2 * nf, 4 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv4 = Conv2DBlock(4 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv5 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv6 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        # self.conv7 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=False)
        self.conv7 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)

        self.upconv1 = UpConv2DBlock(8 * nf, 8 * nf, 4, 2, 1, up_mode=up_mode) #2x2, 512
        # self.upconv2 = UpConv2DBlock(8 * nf * 2, 8 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 4x4, 512
        # self.upconv3 = UpConv2DBlock(8 * nf * 2, 8 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 8x8, 512
        # self.upconv4 = UpConv2DBlock(8 * nf * 2, 4 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 8x8, 512
        self.upconv2 = UpConv2DBlock(8 * nf * 2, 8 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=False) # 4x4, 512
        self.upconv3 = UpConv2DBlock(8 * nf * 2, 8 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=False) # 8x8, 512
        self.upconv4 = UpConv2DBlock(8 * nf * 2, 4 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=False) # 8x8, 512

        # Coord regressor
        # self.upconvC5 = UpConv2DBlock(4 * nf * 3, 2 * nf, 4, 2, 1, up_mode='upsample') # 16
        # self.upconvC6 = UpConv2DBlock(2 * nf * 2, 1 * nf, 4, 2, 1, up_mode='upsample') # 32
        # self.upconvC7 = UpConv2DBlock(1 * nf * 2, output_nc, 4, 2, 1, use_bn=False, use_bias=True, up_mode='upsample') # 64x64, 128
        self.upconvC5 = UpConv2DBlock(4 * nf * 3, 2 * nf, 4, 2, 1, up_mode='upconv') # 16
        self.upconvC6 = UpConv2DBlock(2 * nf * 2, 1 * nf, 4, 2, 1, up_mode='upconv') # 32
        self.upconvC7 = UpConv2DBlock(1 * nf * 2, output_nc, 4, 2, 1, up_mode="upconv") # 64x64, 128

        if return_2branches:
            # Normal regressor
            self.upconvN5 = UpConv2DBlock(4 * nf * 3, 2 * nf, 4, 2, 1, up_mode='upconv') # 32x32, 256
            self.upconvN6 = UpConv2DBlock(2 * nf * 2, 1 * nf, 4, 2, 1, up_mode='upconv') # 64x64, 128
            self.upconvN7 = UpConv2DBlock(1 * nf * 2, 3, 4, 2, 1, use_bn=False, use_bias=True, up_mode='upconv') # 64x64, 128

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        d5 = self.conv5(d4)
        d6 = self.conv6(d5)
        d7 = self.conv7(d6)

        # shared decoder layers
        u1 = self.upconv1(d7, d6)
        u2 = self.upconv2(u1, d5)
        u3 = self.upconv3(u2, d4)
        u4 = self.upconv3(u3, d3)

        # coord regressor
        uc5 = self.upconvC5(u4, d2)
        uc6 = self.upconvC6(uc5, d1)
        uc7 = self.upconvC7(uc6)

        if self.return_2branches:
            # normal regressor
            un5 = self.upconvN5(u4, d2)
            un6 = self.upconvN6(un5, d1)
            un7 = self.upconvN7(un6)

            return uc7, un7

        return uc7

class ShapeDecoder(nn.Module):
    def __init__(self, in_size, hidden_size=256, actv_fn="softplus"):
        super(ShapeDecoder, self).__init__()

        self.conv1 = nn.Conv1d(in_size, hidden_size, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.conv4 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.conv5 = nn.Conv1d(hidden_size + in_size, hidden_size, kernel_size=1)
        self.conv6 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.conv7 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.conv8 = nn.Conv1d(hidden_size, 3, kernel_size=1)

        self.conv6N = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.conv7N = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.conv8N = nn.Conv1d(hidden_size, 3, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.bn6 = nn.BatchNorm1d(hidden_size)
        self.bn7 = nn.BatchNorm1d(hidden_size)

        self.bn6N = nn.BatchNorm1d(hidden_size)
        self.bn7N = nn.BatchNorm1d(hidden_size)

        self.actvn = nn.ReLU() if actv_fn == "relu" else nn.Softplus()

    def forward(self, x):
        x1 = self.actvn(self.bn1(self.conv1(x)))
        x2 = self.actvn(self.bn2(self.conv2(x1)))
        x3 = self.actvn(self.bn3(self.conv3(x2)))
        x4 = self.actvn(self.bn4(self.conv4(x3)))
        x5 = self.actvn(self.bn5(self.conv5(torch.cat([x, x4], dim=1))))

        # predict residuals
        x6 = self.actvn(self.bn6(self.conv6(x5)))
        x7 = self.actvn(self.bn7(self.conv7(x6)))
        x8 = self.conv8(x7)

        # predict normals
        x6N = self.actvn(self.bn6N(self.conv6N(x5)))
        x7N = self.actvn(self.bn7N(self.conv7N(x6N)))
        x8N = self.conv8N(x7N)

        return x8, x8N