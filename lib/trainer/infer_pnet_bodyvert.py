import os 
import torch
from tqdm import tqdm

from lib.losses import chamfer_loss, normal_loss, pcd_density_loss
from lib.utils_model import gen_transf_mtx_from_vtransf
from lib.utils_io import save_result_examples, customized_export_ply, vertex_normal_2_vertex_color
from lib.utils_io import get_scan_pcl_by_name

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
                      mode="val",
                      repeat=4,
                      ):
    '''
    If the test outfit is seen, just use the optimal clothing code found during training
    '''
    model.eval()
    print("Evaluating...")

    n_test_samples = len(test_loader.dataset)

    N_subsample = 1

    test_s2m, test_m2s, test_lnormal, test_lrd, test_lrg, test_ldense = 0., 0., 0., 0., 0., 0.

    with torch.no_grad():
        for data in tqdm(test_loader):
            [posmap, posmap_meanshape, scan_pc, scan_normals, body_verts, scan_names, vtransf, clothing_label] = data
            gpu_data = [posmap, posmap_meanshape, scan_pc, scan_normals, body_verts, vtransf, clothing_label]
            [posmap, posmap_meanshape, scan_pc, scan_normals, body_verts, vtransf, clothing_label] = list(map(lambda x: x.to(device, non_blocking=True), gpu_data))
            
            batch, _, H, W = posmap.shape

            if mode == "test_unseen":
                clothing_label = torch.zeros_like(clothing_label).cuda()
            geom_feature_map_batch = geom_feature_map[clothing_label, ...].cuda()

            # transf_mtx_map = gen_transf_mtx_from_vtransf(vtransf, bary_coords_map, face_list_uv, scaling=transf_scaling)
            uv_coords_map_batch = uv_coords_map.expand(batch, -1, -1).contiguous()
            pq_samples = subpixel_sampler.sample_regular_points()
            pq_batch = pq_samples.expand(batch, H * W, -1, -1)

            # bp_locations = posmap.expand(N_subsample, -1, -1, -1, -1).permute([1, 2, 3, 4, 0])
            # transf_mtx_map = transf_mtx_map.expand(N_subsample, -1, -1, -1, -1, -1).permute([1, 2, 3, 0, 4, 5])
            # bp_locations = body_verts
            rep_time = repeat
            bp_locations = body_verts.unsqueeze(1).repeat(1, rep_time, 1, 1)
            bp_locations = bp_locations.reshape(batch, -1, 3)

            # ---------------- move forward pass -------------------

            pred_res, pred_normals = model(body_verts, 
                                           geom_feature_map_batch,
                                           uv_coords_map_batch,
                                           pq_batch)
        
            pred_res = pred_res.unsqueeze(-1)
            pred_normals = pred_normals.unsqueeze(-1)
            
            # transf_mtx_map = vtransf
            # temporarily do a simple copy (instead of interpolation)
            transf_mtx_map = vtransf.unsqueeze(1).repeat(1, rep_time, 1, 1, 1).reshape(batch, -1, 3, 3)

            pred_res = torch.matmul(transf_mtx_map, pred_res).squeeze(-1)
            pred_normals = torch.matmul(transf_mtx_map, pred_normals).squeeze(-1)
            pred_normals = torch.nn.functional.normalize(pred_normals, dim=-1)

            # [4, N, 3]
            full_pred = pred_res.contiguous() + bp_locations     
            full_pred = full_pred.reshape(batch, -1, 3).contiguous()
            # [4, N, 3]
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

            L_dense = pcd_density_loss(full_pred, rep_time=rep_time)

            test_s2m += s2m
            test_m2s += m2s
            test_lnormal += lnormals
            test_lrd += L_rd
            test_lrg += L_rg
            test_ldense += L_dense

            if "test" in mode:
                save_spacing = 1 if save_all_results else 10
                batch_num = full_pred.shape[0]
                for i in range(0, batch_num, save_spacing):
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
        test_ldense /= n_test_samples

        test_s2m = test_s2m.detach().cpu().numpy()
        test_m2s = test_m2s.detach().cpu().numpy()
        test_lnormal = test_lnormal.detach().cpu().numpy()
        test_lrd = test_lrd.detach().cpu().numpy()
        test_lrg = test_lrg.detach().cpu().numpy()
        test_ldense = test_ldense.detach().cpu().numpy()

        print("m2s loss: {:.3e}, s2m loss: {:.3e}, normal loss: {:.3e}, lrd: {:.3e}, lrg: {:.3e}, ldense: {:.3e}".format(
            test_s2m, test_m2s, test_lnormal, test_lrd, test_lrg, test_ldense
        ))

        if mode == "val":
            # if epoch_idx == 0 or epoch_idx % 20 == 0:
            if True:
                save_result_examples(sample_dir, model_name, scan_names[0],
                                     points=full_pred[0], normals=pred_normals[0],
                                     patch_color=None, epoch=epoch_idx)
    return [test_s2m, test_m2s, test_lnormal, test_lrd, test_lrg, test_ldense]        

def model_forward_and_loss( 
                            model, 
                            geom_featmap, 
                            device, 
                            test_loader,
                            flist_uv, 
                            valid_idx, 
                            uv_coord_map, 
                            samples_dir=None, 
                            model_name=None,
                            bary_coords_map=None, 
                            transf_scaling=1.0,
                            subpixel_sampler=None, 
                            optim_step_id=0,
                            dense_scan_pc=None, 
                            dense_scan_n=None, 
                            random_subsample_scan=False, 
                            num_unseen_frames=1
                        ):
    '''
    A forward pass of the model and compute the loss
    for the test-unseen case optimization
    '''

    model.eval()

    if num_unseen_frames == 1:
        data = test_loader.dataset[0] # take the first example to optimize
    else:
        data = next(iter(test_loader))

    [query_posmap, inp_posmap, target_pc_n, target_pc, body_verts, target_names, vtransf, index] = data
    gpu_data = [query_posmap, inp_posmap, target_pc_n, target_pc, body_verts]

    if num_unseen_frames == 1:
        [query_posmap, inp_posmap, target_pc_n, target_pc, body_verts] = list(map(lambda x: x.unsqueeze(0).to(device, non_blocking=True), gpu_data))
        vtransf = vtransf.unsqueeze(0).to(device)
    else:
        [query_posmap, inp_posmap, target_pc_n, target_pc, body_verts] = list(map(lambda x: x.to(device, non_blocking=True), gpu_data))

    bs, _, H, W = query_posmap.size()

    if ((dense_scan_pc is not None) and (dense_scan_n is not None)):
        target_pc = dense_scan_pc.unsqueeze(0).to(device)
        target_pc_n = dense_scan_n.unsqueeze(0).to(device)

        """
        if random_subsample_scan:
            rand_idx = torch.randperm(target_pc.shape[1])[:25000]
            target_pc = target_pc[:, rand_idx]
            target_pc_n = target_pc_n[:, rand_idx]
        """

    N_subsample = subpixel_sampler.npoints
    
    geom_featmap_batch = geom_featmap.expand(bs, -1, -1).to(device)

    vtransf = vtransf.to(device)
    # transf_mtx_map = gen_transf_mtx_from_vtransf(vtransf, bary_coords_map, flist_uv, scaling=transf_scaling)

    uv_coord_map_batch = uv_coord_map.expand(bs, -1, -1).contiguous()

    pq_samples = subpixel_sampler.sample_regular_points()
    pq_repeated = pq_samples.expand(bs, H * W, -1, -1)  # B, H*W, samples_per_pix, 2

    # bp_locations = query_posmap.expand(N_subsample, -1, -1,-1,-1).permute([1, 2, 3, 4, 0]) # bs, C, H, W, N_sample
    repeat = 6
    bp_locations = body_verts.unsqueeze(1).repeat(1, repeat, 1, 1)
    bp_locations = bp_locations.reshape(bs, -1, 3)

    # transf_mtx_map = transf_mtx_map.expand(N_subsample, -1, -1, -1, -1, -1).permute([1, 2, 3, 0, 4, 5])  # [bs, H, W, N_subsample, 3, 3]
    transf_mtx_map = vtransf.unsqueeze(1).repeat(1, repeat, 1, 1, 1).reshape(bs, -1, 3, 3)

    # core: model forward
    pred_res, pred_normals = model(body_verts, geometry_feature_map=geom_featmap_batch,
                                    uv_location=uv_coord_map_batch,
                                    pq_coords=pq_repeated)

    # permute, local --> global, add to body basis points
    # pred_res = pred_res.permute([0,2,3,4,1]).unsqueeze(-1)
    # pred_normals = pred_normals.permute([0, 2, 3, 4, 1]).unsqueeze(-1)
    # [4, N, 3]
    pred_res = pred_res.unsqueeze(-1)
    pred_normals = pred_normals.unsqueeze(-1)

    pred_res = torch.matmul(transf_mtx_map, pred_res).squeeze(-1)
    pred_normals = torch.matmul(transf_mtx_map, pred_normals).squeeze(-1)

    # full_pred = pred_res.permute([0,4,1,2,3]).contiguous() + bp_locations
    # take valid points from UV map
    # full_pred = full_pred.permute([0,2,3,4,1]).reshape(bs, -1, N_subsample, 3)[:, valid_idx, ...]
    # pred_normals = pred_normals.reshape(bs, -1, N_subsample, 3)[:, valid_idx, ...]
    # full_pred = full_pred.reshape(bs, -1, 3).contiguous()
    # pred_normals = pred_normals.reshape(bs, -1, 3).contiguous()
    # pred_normals = torch.nn.functional.normalize(pred_normals, dim=-1)

    # [4, N, 3]
    full_pred = pred_res.contiguous() + bp_locations     
    full_pred = full_pred.reshape(bs, -1, 3).contiguous()
    # [4, N, 3]
    pred_normals = pred_normals.reshape(bs, -1, 3).contiguous()

    print("target name: {}".format(target_names))

    # loss calc
    _, s2m, idx_closest_gt, _ = chamfer_loss(full_pred, target_pc) #idx1: [#pred points]
    s2m = s2m.mean()
    lnormal, closest_target_normals = normal_loss(pred_normals, target_pc_n, idx_closest_gt)
    nearest_idx = idx_closest_gt.expand(3, -1, -1).permute([1,2,0]).long() # [batch, N] --> [batch, N, 3], repeat for the last dim
    target_points_chosen = torch.gather(target_pc, dim=1, index=nearest_idx)
    pc_diff = target_points_chosen - full_pred # vectors from prediction to its closest point in gt pcl
    m2s = torch.sum(pc_diff * closest_target_normals, dim=-1) # project on direction of the normal of these gt points
    m2s = torch.mean(m2s**2) # the length (squared) is the approx. pred point to scan surface dist.
    rgl_len = torch.mean(pred_res ** 2)

    ldense = pcd_density_loss(full_pred)

    if isinstance(target_names, str):
        target_names = [target_names]

    if optim_step_id % 50 ==0:
        for i in range(len(target_names)):
            save_result_examples(samples_dir, model_name, '{}_{}'.format(target_names[i], optim_step_id),
                            points=full_pred[i], normals=pred_normals[i], patch_color=None)

    # save the gt for the optimization
    if optim_step_id == 0:
        for i, name in enumerate(target_names):
            gt_save_fn = os.path.join(samples_dir, 'GT_{}.ply'.format(name))
            gt_vn = target_pc_n[i].detach().cpu().numpy()
            gt_vc = vertex_normal_2_vertex_color(gt_vn)
            customized_export_ply(gt_save_fn, v=target_pc[i].detach().cpu().numpy(),
                                  v_n=gt_vn, v_c=gt_vc)

    return s2m, m2s, lnormal, rgl_len, ldense

def reconstruct(
                model, 
                geom_featmap_init, 
                device, 
                test_loader,
                flist_uv, 
                valid_idx, 
                uv_coord_map,
                bary_coords_map=None,
                subpixel_sampler=None, 
                samples_dir='', 
                model_name='',
                loss_weights=None,
                transf_scaling=1.0,
                lr=5e-4, 
                num_optim_iterations=1000, 
                dense_scan_pc=None, 
                dense_scan_n=None, 
                random_subsample_scan=False, 
                num_unseen_frames=1
            ):
    '''
    partially borrowed from DeepSDF codes

    optimize the latent geometric feature tensor w.r.t. the given observation (scan point cloud),
    while keeping the network weights fixed.
    '''
    def adjust_learning_rate(initial_lr, optimizer, num_optim_iterations, decreased_by, adjust_lr_every):
        lr = initial_lr * ((1 / decreased_by) ** (num_optim_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_optim_iterations / 2)

    geom_featmap = torch.zeros_like(geom_featmap_init)
    geom_featmap.data[:] = geom_featmap_init[:]
    geom_featmap.requires_grad = True

    optimizer = torch.optim.Adam([geom_featmap], lr=lr)

    w_s2m, w_m2s, w_lnormal, w_rgl_len, w_rgl_latent, w_ldense = loss_weights

    for e in range(num_optim_iterations):
        model.eval()
        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()
        s2m, m2s, lnormal, rgl_len, ldense = model_forward_and_loss(model, geom_featmap, device, test_loader,
                                                            flist_uv, valid_idx, uv_coord_map,
                                                            samples_dir=samples_dir, model_name=model_name,
                                                            subpixel_sampler=subpixel_sampler,
                                                            optim_step_id=e,
                                                            bary_coords_map=bary_coords_map,transf_scaling=transf_scaling,
                                                            dense_scan_pc=dense_scan_pc, dense_scan_n=dense_scan_n,
                                                            random_subsample_scan=random_subsample_scan, num_unseen_frames=num_unseen_frames)
        loss = s2m*w_s2m + m2s*w_m2s+ lnormal*w_lnormal + \
                rgl_len* w_rgl_len + w_rgl_latent * torch.mean(geom_featmap**2) + w_ldense * ldense

        loss.backward()
        optimizer.step()

        # if e % 50 == 0:
        if True:
            s2m, m2s, lnormal, rgl_len = list(map(lambda x: x.cpu().data.numpy(), [s2m, m2s, lnormal, rgl_len]))
            print('Step {:<4}, s2m: {:.3e}, m2s: {:.3e}, normal: {:.3e}, rgl_len: {:.3e}, ldense: {:.3e}'.format(e, s2m, m2s, lnormal, rgl_len, ldense))

    return s2m, m2s, lnormal, geom_featmap

def test_unseen_clothing(
                    model, 
                    geometry_feature_map, 
                    test_loader, 
                    test_loader_for_optim, 
                    epoch_idx, 
                    samples_dir, 
                    mode='test_unseen',
                    face_list_uv=None, 
                    valid_idx=None, 
                    uv_coords_map=None, 
                    bary_coords_map=None,
                    transf_scaling=1.0,
                    device='cuda',
                    model_name=None,
                    subpixel_sampler=None, 
                    loss_weights=None,
                    dataset_type='cape',
                    num_optim_iterations=400,
                    random_subsample_scan=False,
                    save_all_results=False,
                    num_unseen_frames=1,
                    repeat=6
                    ):
    '''
    Test when the outfit is unseen during training.
        - first optimize the latent clothing geometric features (with the network weights fixed)
        - then fix the geometric feature, and vary the input pose to predict the pose-dependent shape
    '''
    model.eval()

    # use the trained examples' geom map average value as init values for optimization
    geometry_feature_map_n = torch.ones(1,
                                      64,
                                      10475)
    geometry_feature_map_n.normal_(mean=0., std=0.01).to(device)
    geometry_feature_map_n.requires_grad = True
    geom_featmap_init = geometry_feature_map_n.mean(0).unsqueeze(0)

    if num_unseen_frames == 1:
        data = test_loader_for_optim.dataset[0] # optimize w.r.t. the first (single) scan in the unseen test loader
        target_bname = data[-3]
        # scan_pc, scan_n = get_scan_pcl_by_name(target_bname)
        scan_pc, scan_n = data[2], data[3]
    else:
        raise NotImplementedError

    # get the name of the subject+outfit combo
    if dataset_type.lower() == 'cape': # for CAPE data
        subj, clo, _ = target_bname.split('_',2)
        subj_clo = '{}_{}'.format(subj, clo)
    else: # for ReSynth data
        subj_clo = target_bname.split('.')[0]
    
    # optize w.r.t. the entire point set of the scan, or treat randomly sample a subset of points from the scan at each iteration.
    points_policy = 'active_rand_gt' if random_subsample_scan else 'full_gt' 
    print('\n------Step 1: Optimizing w.r.t. UNSEEN scan with {}\n'.format(points_policy))
    
    samples_dir_optim = os.path.join(samples_dir, 'optim_results_{}_{}'.format(subj_clo, points_policy)) # for saving intermediate results of the optim process
    samples_dir_anim = os.path.join(samples_dir, subj_clo) # for saving pose-depdt shape predictions
    os.makedirs(samples_dir_optim, exist_ok=True)
    os.makedirs(samples_dir_anim, exist_ok=True)

    s2m, m2s, lnormal, geom_featmap_optimized = reconstruct(
                                                            model, 
                                                            geom_featmap_init, 
                                                            device, 
                                                            test_loader_for_optim,
                                                            face_list_uv, 
                                                            valid_idx, 
                                                            uv_coords_map, 
                                                            bary_coords_map=bary_coords_map,
                                                            subpixel_sampler=subpixel_sampler, 
                                                            samples_dir=samples_dir_optim, 
                                                            model_name=model_name,
                                                            loss_weights=loss_weights,
                                                            transf_scaling=transf_scaling,
                                                            lr=5e-4, 
                                                            num_optim_iterations=num_optim_iterations,
                                                            dense_scan_pc=scan_pc, 
                                                            dense_scan_n=scan_n,
                                                            random_subsample_scan=random_subsample_scan, 
                                                            num_unseen_frames=num_unseen_frames
                                                        )

    print('---after optimization, s2m: {:.3e}, m2s: {:.3e}, normal: {:.3e}'.format(s2m, m2s, lnormal))

    print('\n------Step 2: predict the pose-dependent shape of the unseen scan with the optimized geometric feature tensor')
    
    """
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
                      mode="val",
                      repeat=4,
    """
    
    test_stats = test_seen_clothing(
                                model, 
                                geom_featmap_optimized, 
                                test_loader, 
                                samples_dir_anim, 
                                epoch_idx=0,
                                subpixel_sampler=subpixel_sampler,
                                model_name=model_name,
                                device=device,
                                face_list_uv=face_list_uv, 
                                valid_idx=valid_idx,
                                uv_coords_map=uv_coords_map,
                                bary_coords_map=bary_coords_map,
                                transf_scaling=transf_scaling,
                                save_all_results=save_all_results,
                                mode=mode,
                                repeat=6
                            )
    return test_stats


    