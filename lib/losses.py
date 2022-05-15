import torch 
import torch.nn.functional as F
import pytorch3d.ops.knn as knn
import pytorch3d.ops as ops

from emd.emd import earth_mover_distance

def chamfer_loss_separate(output, target, weight=1e4, phase='train', debug=False):
    from chamferdist.chamferdist import ChamferDistance
    cdist = ChamferDistance()
    model2scan, scan2model, idx1, idx2 = cdist(output, target)
    if phase == 'train':
        return model2scan, scan2model, idx1, idx2
    else: # in test, show both directions, average over points, but keep batch
        return torch.mean(model2scan, dim=-1)* weight, torch.mean(scan2model, dim=-1)* weight,


def chamfer_loss(output, target, phase="train", weight=1e4):
    from chamferdist.chamferdist import ChamferDistance
    cdist = ChamferDistance()
    model2scan, scan2model, idx1, idx2 = cdist(output, target)
    if phase == "train":
        return model2scan, scan2model, idx1, idx2
    else:
        return torch.mean(model2scan, dim=-1) * weight, torch.mean(scan2model, dim=-1) * weight

def normal_loss(output_normals, target_normals, nearest_idx, weight=1.0, phase='train'):
    '''
    Given the set of nearest neighbors found by chamfer distance, calculate the
    L1 discrepancy between the predicted and GT normals on each nearest neighbor point pairs.
    Note: the input normals are already normalized (length==1).
    '''
    nearest_idx = nearest_idx.expand(3, -1, -1).permute([1,2,0]).long() # [batch, N] --> [batch, N, 3], repeat for the last dim
    target_normals_chosen = torch.gather(target_normals, dim=1, index=nearest_idx)

    assert output_normals.shape == target_normals_chosen.shape

    if phase == 'train':
        lnormal = F.l1_loss(output_normals, target_normals_chosen, reduction='mean')  # [batch, 8000, 3])
        return lnormal, target_normals_chosen
    else:
        lnormal = F.l1_loss(output_normals, target_normals_chosen, reduction='none')
        lnormal = lnormal.mean(-1).mean(-1) # avg over all but batch axis
        return lnormal, target_normals_chosen
    
def pcd_density_loss(output, rep_time=4):
    dists, _, _ = knn.knn_points(output, output, K=rep_time)
    # dists, _, _ = ops.ball_query(output, output,K=21, radius=0.1)
    dists = dists[:, :, 1:]
    eta_dists = -1 * dists
    
    # h temporarily set to 0.03
    h = 0.03
    omega_dists = torch.exp(-dists * dists / h)
    density = eta_dists * omega_dists
    density = density.mean(-1).mean(-1).mean(-1)
    return density

def emd_loss(output, target):
    return torch.mean(earth_mover_distance(output, target, transpose=False))