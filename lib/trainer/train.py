import torch

from lib.losses import chamfer_loss, normal_loss, pcd_density_loss
from lib.utils_model import gen_transf_mtx_from_vtransf

def train(
         model, 
         train_loader,
         geometry_feature_map,
         optimizer,
         epoch_idx=0,
         summary_writer=None,
         uv_coords_map=None,
         subpixel_sampler=None,
         valid_idx=None,
         mean_valid_idx=None,
         loss_weights=None,
         face_list_uv=None,
         bary_coords_map=None,
         device="cuda",
         transf_scaling=1.0):
    
    num_train_samples = len(train_loader.dataset)
    w_s2m, w_m2s, w_normal, w_lrd, w_lrg = loss_weights

    # add, temporary
    w_dense = 10000.0

    train_s2m, train_m2s, train_normal, train_lrd, train_lrg, train_total = 0., 0., 0., 0., 0., 0.
    train_ldense = 0.

    print("w_s2m: {:.3f}, w_m2s: {:.3f}, w_normal: {:.3f}, w_lrd: {:.3f}, w_lrg: {:.3f}".format(
        w_s2m, w_m2s, w_normal, w_lrd, w_lrg  
    ))

    model.train()
    for step, data in enumerate(train_loader):
        [posmap, posmap_meanshape, scan_pc, scan_normals, body_verts, scan_name, vtransf, clothing_label] = data

        gpu_data = [posmap, posmap_meanshape, scan_pc, scan_normals, body_verts, vtransf, clothing_label]
        [posmap, posmap_meanshape, scan_pc, scan_normals, body_verts, vtransf, clothing_label] = list(map(lambda x: x.to(device), gpu_data))
        batch, _, H, W = posmap.shape

        optimizer.zero_grad()

        transf_mtx_map = gen_transf_mtx_from_vtransf(vtransf, bary_coords_map, face_list_uv, scaling=transf_scaling)

        index = clothing_label
        geometry_feature_map_batch = geometry_feature_map[index, ...].to(device)

        uv_coords_map_batch = uv_coords_map.expand(batch, -1, -1).contiguous()

        pq_samples = subpixel_sampler.sample_regular_points()
        pq_batch = pq_samples.expand(batch, H * W, -1, -1)
        
        N_subsample = 1
        bp_locations = posmap.expand(N_subsample, -1, -1, -1, -1).permute([1, 2, 3, 4, 0])
        transf_mtx_map = transf_mtx_map.expand(N_subsample, -1, -1, -1, -1, -1).permute([1, 2, 3, 0, 4, 5])

        pred_res, pred_normals = model(posmap_meanshape,
                                       geometry_feature_map_batch,
                                       uv_coords_map_batch,
                                       pq_batch)

        pred_res = pred_res.permute([0, 2, 3, 4, 1]).unsqueeze(-1)
        pred_normals = pred_normals.permute([0, 2, 3, 4, 1]).unsqueeze(-1)
        
        # out --> [B, 3, H, W, N_samples] --> [4, 3, 256, 256, 1]
        # [4, 256, 256, 1, 3, 1]

        # transf_mtx_map.shape: [4, 256, 256, 1, 3, 3] 

        # [4, 256, 256, 1, 3]
        pred_res = torch.matmul(transf_mtx_map, pred_res).squeeze(-1)
        pred_normals = torch.matmul(transf_mtx_map, pred_normals).squeeze(-1)
        pred_normals = torch.nn.functional.normalize(pred_normals, dim=-1)

        # [4, 3, 256, 256, 1]
        full_pred = pred_res.permute([0, 4, 1, 2, 3]).contiguous() + bp_locations     #[bs, C, H, W, N_sample],
        # full_pred = pred_res.contiguous() + bp_locations
        # [4, 256, 256, 1, 3]
        full_pred = full_pred.permute([0, 2, 3, 4, 1])
        # [4, 66536, 1, 3]
        full_pred = full_pred.reshape(batch, -1, N_subsample, 3)
        # [4, valid_idx, 1, 3]
        full_pred = full_pred[:, valid_idx, ...]
        # [4, 65536, 1, 2]
        pred_normals = pred_normals.reshape(batch, -1, N_subsample, 3)
        # [4, valid_idx, 1, 3]
        pred_normals = pred_normals[:, valid_idx, ...]

        # [4, valid_idx, 3]
        full_pred = full_pred.reshape(batch, -1, 3).contiguous()
        # [4, valid_idx, 3]
        pred_normals = pred_normals.reshape(batch, -1, 3).contiguous()

        # ------------- loss ---------------

        # chamfer dist loss
        m2s, s2m, index_closest_gt, _ = chamfer_loss(full_pred, scan_pc)
        s2m = torch.mean(s2m)
        # index_closest_gt: [4, 47911]

        # normal loss
        lnormals, closest_target_normals = normal_loss(pred_normals, scan_normals, index_closest_gt)

        # first term of chamfer distance
        nearest_idx = index_closest_gt.expand(3, -1, -1).permute([1, 2, 0]).long()
        target_points_chosen = torch.gather(scan_pc, dim=1, index=nearest_idx)
        pc_difference = -full_pred + target_points_chosen
        m2s = torch.sum(pc_difference * closest_target_normals, dim=-1)
        m2s = torch.mean(m2s ** 2)

        # Lrd, discourages the predicted point displacements from being extremely large
        L_rd = torch.mean(pred_res ** 2)

        # Lrg, penalized the L2-norm of the vectorized geometric feature tensor to 
        # regularize the garment shape space
        L_rg = torch.mean(geometry_feature_map_batch ** 2)

        # L_dense = pcd_density_loss(full_pred)

        loss = w_s2m * s2m + w_m2s * m2s + w_normal * lnormals + \
               w_lrd * L_rd + w_lrg * L_rg#  + w_dense * L_dense
        loss.backward()

        optimizer.step()

        # ------------- accumulate stats -------------
        """
        stats = {
            "s2m": s2m, 
            "m2s": m2s, 
            "lnormals": lnormals,
            "lrd": L_rd, 
            "lrg": L_rg, 
            # "ldense": L_dense,
            "total_loss": loss
        }
        for key in stats.keys():
            summary_writer.add_scalar("{}".format(key), stats[key], step)
        """
        print("epoch: {}, step: {}, s2m: {:.3e}, m2s: {:.3e}, lnormal: {:.3e}, lrd: {:.3e}, lrg: {:.3e}, total_loss: {:.3e}".format(
            epoch_idx, step, s2m, m2s, lnormals, L_rd, L_rg, loss
        ))

        train_s2m += s2m * batch
        train_m2s += m2s * batch
        train_normal += lnormals * batch
        train_lrd += L_rd * batch
        train_lrg += L_rg * batch
        # train_ldense += L_dense * batch
        train_total += loss * batch

    train_s2m /= num_train_samples
    train_m2s /= num_train_samples
    train_normal /= num_train_samples
    train_lrd /= num_train_samples
    train_lrg /= num_train_samples
    # train_ldense /= num_train_samples
    train_total /= num_train_samples

    return train_s2m, train_m2s, train_normal, train_lrd, train_lrg, train_total