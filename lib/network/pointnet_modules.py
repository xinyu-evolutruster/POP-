import torch
import torch.nn as nn
import torch.nn.functional as F

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _  = src.shape
    _, M, _  = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples (number of centroids?)
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10  # inf
    farthest = torch.randint(low=0, high=N, size=(B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(start=0, end=B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points=None, returnfps=False):
    """
    Input:
        npoint:
        radius: the radius of the query ball
        nsample: the number of points sampled from each query baoo
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [b, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(in_channels=last_channel, out_channels=out_channel, kernel_size=1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
    
    def forward(self, xyz, points=None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # print("xyz shape in forward: {}".format(xyz.shape))
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample, npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)  # [B, N, C]
        xyz2 = xyz2.permute(0, 2, 1)  # [B, S, C]

        points2 = points2.permute(0, 2, 1)  # [B, S, D]
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            # the reciprocal (x's reciprocal is 1/x)
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        
        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

class PointNet(nn.Module):
    def __init__(self, input_channel=3, num_classes=64):
        super(PointNet, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, input_channel, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + input_channel, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + input_channel, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + input_channel, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        # print("l0 xyz shape: {}".format(l0_xyz.shape))
        # print("l0 point shape: {}".format(l0_points.shape))

        l1_xyz, l1_points = self.sa1(l0_xyz)
        # print("l1 xyz shape: {}, l1 points shape: {}".format(l1_xyz.shape, l1_points.shape))

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print("l2 xyz shape: {}, l2 points shape: {}".format(l2_xyz.shape, l2_points.shape))
        
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # print("l3 xyz shape: {}, l3 points shape: {}".format(l3_xyz.shape, l3_points.shape))
        
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        # print("l4 xyz shape: {}, l4 points shape: {}".format(l4_xyz.shape, l4_points.shape))

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        # print("new l3 points shape: {}".format(l3_points.shape))
        
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # print("new l2 points shape: {}".format(l2_points.shape))
        
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # print("new l1 points shape: {}".format(l1_points.shape))
        
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        # print("new l0 points shape: {}".format(l0_points.shape))

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        # print("x shape: {}".format(x.shape))

        x = self.conv2(x)
        # print("x shape: {}".format(x.shape))

        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        # print("x shape: {}".format(x.shape))

        return x

class FeatureExpansion(nn.Module):
    def __init__(self, in_size, hidden_size=64, out_size=64, r=9):
        super(FeatureExpansion, self).__init__()

        self.num_replicate = r
        self.expansions = nn.ModuleList()

        for i in range(self.num_replicate):
            mlp_convs = nn.ModuleList()
            mlp_convs.append(nn.Conv1d(in_size, hidden_size, kernel_size=1))
            mlp_convs.append(nn.BatchNorm1d(hidden_size))
            mlp_convs.append(nn.Conv1d(hidden_size, hidden_size, kernel_size=1))
            mlp_convs.append(nn.BatchNorm1d(hidden_size))
            mlp_convs.append(nn.Conv1d(hidden_size, out_size, kernel_size=1))
            
            self.expansions.append(mlp_convs)

    def forward(self, x):
        output = None
        for i in range(self.num_replicate):
            mlp_convs = self.expansions[i]
            conv0 = mlp_convs[0]
            conv1 = mlp_convs[2]
            conv2 = mlp_convs[4]
            bn0 = mlp_convs[1]
            bn1 = mlp_convs[3]
            fea = F.relu(conv1(bn0(conv0(x))))
            fea = F.relu(conv2(bn1(fea)))

            if i == 0:
                output = fea.unsqueeze(1)
            else:
                fea = fea.unsqueeze(1)
                output = torch.cat([output, fea], dim=1)
             
        return output

class SmallShapeDecoder(nn.Module):
    def __init__(self, in_size, hidden_size=256, actv_fn="softplus"):
        super(ShapeDecoder, self).__init__()

        self.conv1 = nn.Conv1d(in_size, hidden_size, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.conv4 = nn.Conv1d(hidden_size + in_size, hidden_size, kernel_size=1)
        self.conv5 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.conv6 = nn.Conv1d(hidden_size, 3, kernel_size=1)

        self.conv5N = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.conv6N = nn.Conv1d(hidden_size, 3, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)

        self.bn4N = nn.BatchNorm1d(hidden_size)
        self.bn5N = nn.BatchNorm1d(hidden_size)

        self.actvn = nn.ReLU() if actv_fn == "relu" else nn.Softplus()

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(torch.cat([x, x3], dim=1))))

        # predict residuals
        x5 = F.relu(self.bn5(self.conv5(x4)))
        x6 = self.conv6(x5)

        # predict normals
        x5N = F.relu(self.bn5N(self.conv6N(x4)))
        x6N = F.relu(x5N)

        return x6, x6N

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
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x5 = F.relu(self.bn5(self.conv5(torch.cat([x, x4], dim=1))))

        # predict residuals
        x6 = F.relu(self.bn6(self.conv6(x5)))
        x7 = F.relu(self.bn7(self.conv7(x6)))
        x8 = self.conv8(x7)

        # predict normals
        x6N = F.relu(self.bn6N(self.conv6N(x5)))
        x7N = F.relu(self.bn7N(self.conv7N(x6N)))
        x8N = self.conv8N(x7N)

        return x8, x8N