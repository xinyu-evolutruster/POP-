import os 
import torch
from tqdm import tqdm

from lib.losses import chamfer_loss, normal_loss
from lib.utils_model import gen_transf_mtx_from_vtransf
from lib.utils_io import save_result_examples, customized_export_ply, vertex_normal_2_vertex_color

def test_seen_clothing( 
                      model,
                      geom_feature_map,
                      test_loader,
                      sample_dir,
                      epoch_idx,
                      model_name=None,
                      face_list_uv=None,
                      valid_idx=None,
                      mean_valid_idx=None,
                      uv_coords_map=None,
                      bary_coords_map=None,
                      subpixel_sampler=None,
                      save_all_results=False,
                      transf_scaling=1.0,
                      device="cuda",
                      mode="val"
                      ):
    '''
    If the test outfit is seen, just use the optimal clothing code found during training
    '''
    model.eval()
    print("Evaluating...")

    n_test_samples = len(test_loader.dataset)

    N_subsample = 1

    test_s2m, test_m2s, test_lnormal, test_lrd, test_lrg = 0., 0., 0., 0., 0.

    with torch.no_grad():
        for data in tqdm(test_loader):
            [posmap, posmap_meanshape, scan_pc, scan_normals, body_verts, scan_names, vtransf, clothing_label] = data
            gpu_data = [posmap, posmap_meanshape, scan_pc, scan_normals, body_verts, vtransf, clothing_label]
            [posmap, posmap_meanshape, scan_pc, scan_normals, body_verts, vtransf, clothing_label] = list(map(lambda x: x.to(device, non_blocking=True), gpu_data))
            
            batch, _, H, W = posmap.shape

            if mode == "test_unseen":
                clothing_label = torch.zeros_like(clothing_label).cuda()
            geom_feature_map_batch = geom_feature_map[clothing_label, ...].cuda()

            transf_mtx_map = gen_transf_mtx_from_vtransf(vtransf, bary_coords_map, face_list_uv, scaling=transf_scaling)
            uv_coords_map_batch = uv_coords_map.expand(batch, -1, -1).contiguous()
            pq_samples = subpixel_sampler.sample_regular_points()
            pq_batch = pq_samples.expand(batch, H * W, -1, -1)

            bp_locations = posmap.expand(N_subsample, -1, -1, -1, -1).permute([1, 2, 3, 4, 0])
            transf_mtx_map = transf_mtx_map.expand(N_subsample, -1, -1, -1, -1, -1).permute([1, 2, 3, 0, 4, 5])

            # ---------------- move forward pass -------------------

            pred_res, pred_normals = model(posmap_meanshape, 
                                           geom_feature_map_batch,
                                           uv_coords_map_batch,
                                           pq_batch)
        
            pred_res = pred_res.permute([0, 2, 3, 4, 1]).unsqueeze(-1)
            pred_normals = pred_normals.permute([0, 2, 3, 4, 1]).unsqueeze(-1)

            pred_res = torch.matmul(transf_mtx_map, pred_res).squeeze(-1)
            pred_normals = torch.matmul(transf_mtx_map, pred_normals).squeeze(-1)
            pred_normals = torch.nn.functional.normalize(pred_normals, dim=-1)

            full_pred = pred_res.permute([0, 4, 1, 2, 3]).contiguous() + bp_locations     #[bs, C, H, W, N_sample],
            full_pred = full_pred.permute([0, 2, 3, 4, 1])
            full_pred = full_pred.reshape(batch, -1, N_subsample, 3)
            full_pred = full_pred[:, valid_idx, ...]

            pred_normals = pred_normals.reshape(batch, -1, N_subsample, 3)
            pred_normals = pred_normals[:, valid_idx, ...]

            full_pred = full_pred.reshape(batch, -1, 3).contiguous()
            pred_normals = pred_normals.reshape(batch, -1, 3).contiguous()

            # ---------------- loss ------------------
            # chamfer dist loss
            _, s2m, index_closest_gt, _ = chamfer_loss(full_pred, scan_pc)
            s2m = torch.mean(s2m)

            # normal loss
            lnormals, closest_target_normals = normal_loss(pred_normals, scan_normals, index_closest_gt)

            # first term of chamfer distance
            nearest_idx = index_closest_gt.expand(3, -1, -1).permute([1, 2, 0]).long()
            target_points_chosen = torch.gather(scan_pc, dim=1, index=nearest_idx)
            pc_difference = full_pred - target_points_chosen
            m2s = torch.sum(pc_difference * closest_target_normals, dim=-1)
            m2s = torch.mean(m2s ** 2)

            # Lrd, discourages the predicted point displacements from being extremely large
            L_rd = torch.mean(pred_res ** 2)

            # Lrg, penalized the L2-norm of the vectorized geometric feature tensor to 
            # regularize the garment shape space
            L_rg = torch.mean(geom_feature_map_batch ** 2)

            test_s2m += s2m
            test_m2s += m2s
            test_lnormal += lnormals
            test_lrd += L_rd
            test_lrg += L_rg

            if "test" in mode:
                save_spacing = 1 if save_all_results else 10
                batch_num = full_pred.shape[0]
                for i in range(0, batch_num, step=save_spacing):
                    save_result_examples(sample_dir, model_name, scan_names[i], 
                                         points=full_pred[i], normals=pred_normals[i])
            if "val" in mode:
                save_spacing = 10
                new_sample_dir = os.path.join(sample_dir, "pop_orig")
                batch_num = full_pred.shape[0]
                if not os.path.exists(new_sample_dir):
                    os.mkdir(new_sample_dir)
                for i in range(0, batch_num, 10):
                    save_result_examples(new_sample_dir, model_name, scan_names[i], 
                                         points=full_pred[i], normals=pred_normals[i])
                                         
        test_s2m /= n_test_samples
        test_m2s /= n_test_samples
        test_lnormal /= n_test_samples
        test_lrd /= n_test_samples
        test_lrg /= n_test_samples

        test_s2m = test_s2m.detach().cpu().numpy()
        test_m2s = test_m2s.detach().cpu().numpy()
        test_lnormal = test_lnormal.detach().cpu().numpy()
        test_lrd = test_lrd.detach().cpu().numpy()
        test_lrg = test_lrg.detach().cpu().numpy()

        print("m2s loss: {:.3e}, s2m loss: {:.3e}, normal loss: {:.3e}, lrd: {:.3e}, lrg: {:.3e}".format(
            test_s2m, test_m2s, test_lnormal, test_lrd, test_lrg
        ))

        if mode == "val":
            # if epoch_idx == 0 or epoch_idx % 20 == 0:
            if True:
                save_result_examples(sample_dir, model_name, scan_names[0],
                                     points=full_pred[0], normals=pred_normals[0],
                                     patch_color=None, epoch=epoch_idx)
    return [test_s2m, test_m2s, test_lnormal, test_lrd, test_lrg]        
    
def test_unseen_clothing(
                        model,
                        geom_feature_map,
                        test_loader,
                        sample_dir,
                        model_name=None,
                        face_list_uv=None,
                        valid_idx=None,
                        uv_coords_map=None,
                        bary_coords_map=None,
                        subpixel_sampler=None,
                        save_all_results=False,
                        device="cuda",
                        mode="test_unseen",
                        num_optim_iterations=400
                        ):
    '''
    Test when the outfit is unseen during training.
        - first optimize the latent clothing geometric features (with the network weights fixed)
        - then fix the geometric feature, and vary the input pose to predict the pose-dependent shape
    '''
    model.eval()
    