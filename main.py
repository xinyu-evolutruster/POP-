from tkinter.messagebox import OK
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from datetime import date, datetime

from lib.config_parser import parse_configs, parse_outfits
from lib.utils_io import load_masks, load_barycentric_coords
from lib.utils_io import save_model, save_latent_features, load_latent_features

# from lib.network.network import Network
# from lib.network.network_pnet import Network
from lib.network.network_pnet_bodyvert import Network

from lib.dataset import Dataset

# from lib.trainer.train import train
from lib.trainer.train_pnet_bodyvert import train
# from lib.trainer.infer import test_seen_clothing, test_unseen_clothing
from lib.trainer.infer_pnet_bodyvert import test_seen_clothing, test_unseen_clothing

from lib.utils_train import adjust_loss_weights
from lib.utils_model import SampleSquarePoints

import os
PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
LOGS_PATH = os.path.join(PROJECT_DIR, "checkpoints")
SAMPLES_PATH = os.path.join(PROJECT_DIR, "results", "saved_samples")

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

torch.manual_seed(12345)
np.random.seed(12345)

def main():
    args = parse_configs()

    # print("val_every = {}".format(args.val_every))

    exp_name = args.name

    data_root = os.path.join(PROJECT_DIR, "data", "packed")
    log_dir = os.path.join(PROJECT_DIR, "tb_logs")
    log_dir = os.path.join(log_dir, "{}".format(date.today().strftime("%m%d")), exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    samples_dir_val = os.path.join(SAMPLES_PATH, exp_name, "val")
    samples_dir_test_seen_base = os.path.join(SAMPLES_PATH, exp_name, "test_seen")
    samples_dir_test_unseen_base = os.path.join(SAMPLES_PATH, exp_name, "test_unseen")
    os.makedirs(samples_dir_val, exist_ok=True)
    os.makedirs(samples_dir_test_seen_base, exist_ok=True)
    os.makedirs(samples_dir_test_unseen_base, exist_ok=True)
    
    ckpt_dir = os.path.join(LOGS_PATH, exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    body_model = "smplx"

    # valid index: the points that represent **human body** on the uv map
    face_list_uv, valid_index, uv_coords_map = load_masks(PROJECT_DIR, args.query_posmap_size, body_model=body_model, device=DEVICE)
    _, mean_valid_index, _ = load_masks(PROJECT_DIR, args.meanshape_posmap_size, body_model=body_model, device=DEVICE)
    
    bary_coords = load_barycentric_coords(PROJECT_DIR, args.query_posmap_size, body_model=body_model, device=DEVICE)

    outfits = parse_outfits(args.name)  
    num_outfits_seen = len(outfits["seen"])
    num_outfits_unseen = len(outfits["unseen"])

    # build model
    model = Network(input_channel=3,
                    pose_feature_channel=args.pose_feature_channel,
                    geometry_feature_channel=args.geo_feature_channel,
                    posmap_size=args.meanshape_posmap_size,
                    uv_feature_dimension=2,
                    pq_feature_dimension=2,
                    hidden_size=args.hidden_size)
    # print(model)

    # geometric feature tensor
    if args.punet == False:
        geometry_feature_map = torch.ones(num_outfits_seen, 
                                        args.geo_feature_channel,
                                        args.meanshape_posmap_size,
                                        args.meanshape_posmap_size)
        geometry_feature_map.normal_(mean=0., std=0.01).to(DEVICE)
        geometry_feature_map.requires_grad = True
    else:
        geometry_feature_map = torch.ones(num_outfits_seen,
                                        args.geo_feature_channel,
                                        args.num_body_verts)
        geometry_feature_map.normal_(mean=0., std=0.01).to(DEVICE)
        geometry_feature_map.requires_grad = True

    subpixel_sampler = SampleSquarePoints(npoints=1, device=DEVICE)
    
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": geometry_feature_map, "lr": args.lr_geometry}
        ]
    )

    n_epochs = args.epochs 
    epoch_now = 0

    dataset_config = {
        "data_root": data_root,
        "query_posmap_size": args.query_posmap_size,
        "meanshape_posmap_size": args.meanshape_posmap_size,
        "sample_spacing": args.data_spacing,
        "dataset_subset_portion": args.dataset_subset_portion,
    }

    model_config = {
        "device": DEVICE,
        "face_list_uv": face_list_uv,
        "uv_coords_map": uv_coords_map,
        "bary_coords_map": bary_coords,
        "valid_idx": valid_index,
        # "mean_valid_idx": mean_valid_index,
        "transf_scaling": 0.02,
        "repeat": 6
    }

    # ----------- load checkpoints -------------
    if args.mode.lower() in ["resume", "test", "test_unseen", "test_seen"]:
        checkpoints = [fn for fn in os.listdir(ckpt_dir) if fn.endswith("_model.pt")]
        checkpoints = sorted(checkpoints)

        latest_path = os.path.join(ckpt_dir, checkpoints[-1])
        print("\n--------------------- Loading checkpoint {}".format(checkpoints[-1]))

        ckpt_loaded = torch.load(latest_path)
        model.load_state_dict(ckpt_loaded["model_state"])

        geo_checkpoints = [fn for fn in os.listdir(ckpt_dir) if fn.endswith("_geom_featmap.pt")]
        geo_checkpoints = sorted(geo_checkpoints)

        geo_latest_path = os.path.join(ckpt_dir, geo_checkpoints[-1])
        # print(geometry_feature_map.shape)
        load_latent_features(geo_latest_path, geometry_feature_map)

        if args.mode.lower() == "resume":
            optimizer.load_state_dict(ckpt_loaded['optimizer_state'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(DEVICE)
            epoch_now = ckpt_loaded["epoch"] + 1
            print("\n--------------------- Resume training from {}".format(epoch_now))
        elif "test" in args.mode.lower():
            epoch_idx = ckpt_loaded["epoch"]
            model = model.to(DEVICE)
            print("\n--------------------- Test with checkpoint at epoch {}".format(epoch_idx))

    '''
    ------------ Train Model ------------
    '''
    print("mode: {}".format(args.mode.lower()))
    if args.mode.lower() in ["train", "resume"]:
        train_dataset = Dataset(split="train", outfits=outfits["seen"], **dataset_config)

        val_outfit = dict()
        for i in range(1):
            val_outfit_name, val_outfit_idx = list(outfits["seen"].items())[i]
            val_outfit[val_outfit_name] = val_outfit_idx

        val_dataset = Dataset(split="test", outfits=val_outfit, **dataset_config)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
        # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=True)
        # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=True)

        writer = SummaryWriter(log_dir=log_dir)

        print("Total {} training examples, {} validating examples, training starts...".format(len(train_dataset), len(val_dataset)))

        model = model.to(DEVICE)
        start = time.time()
        pbar = range(epoch_now, n_epochs)

        for epoch_idx in pbar:
            # print("epoch index is {}".format(epoch_idx))
            w_decay_lrd = adjust_loss_weights(args.w_lrd, epoch_idx, mode="decay", start=args.decay_start, every=args.decay_every)
            w_rise_normal = adjust_loss_weights(args.w_normal, epoch_idx, mode="normal", start=args.rise_start, every=args.rise_every)
            # w_pcd_dense = adjust_loss_weights(args.w_ldense, epoch_idx, mode="pcd", start=args.pcd_start)
            w_s2m = adjust_loss_weights(args.w_s2m, epoch_idx, mode="s2m", start=args.s2m_start)
            w_m2s = adjust_loss_weights(args.w_m2s, epoch_idx, mode="m2s", start=args.m2s_start)

            loss_weights = torch.tensor([w_s2m, w_m2s, w_rise_normal, w_decay_lrd, args.w_lrg, args.w_ldense])

            train_stats = train(model,
                                train_loader,
                                geometry_feature_map,
                                optimizer,
                                epoch_idx=epoch_idx,
                                summary_writer=writer,
                                loss_weights=loss_weights,
                                subpixel_sampler=subpixel_sampler,
                                **model_config)

            if (epoch_idx + 1) % 50 == 0 or epoch_idx == n_epochs - 1:
                ckpt_path = os.path.join(ckpt_dir, "{}_epoch{}_model.pt".format(exp_name, str(epoch_idx).zfill(5)))
                save_model(ckpt_path, model, epoch_idx, optimizer)
                geo_ckpt_path = os.path.join(ckpt_dir, "{}_epoch{}_geom_featmap.pt".format(exp_name, str(epoch_idx).zfill(5)))
                save_latent_features(geo_ckpt_path, geometry_feature_map, epoch_idx)
            
            # test on val dataset every N epochs
            print("current epoch: {}".format(epoch_idx))
            if (epoch_idx + 1) % args.val_every == 0:
                print("val for epoch {}".format(epoch_idx))
                dur = (time.time() - start) / (60 * (epoch_idx - epoch_now + 1))
                now = datetime.now()
                datetime_str = now.strftime("%d/%m/%Y %H:%M:%S")
                print("\n{}, Epoch {}, average {:.2f} min / epoch".format(datetime_str, epoch_idx, dur))
                print("weights s2m: {:.1e}, m2s: {:.1e}, normal: {:.1e}, lrd: {:.1e}".format(w_s2m, w_m2s, w_rise_normal, w_decay_lrd))

                val_stats = test_seen_clothing(model,
                                               geometry_feature_map,
                                               val_loader,
                                               samples_dir_val,
                                               epoch_idx,
                                               model_name=exp_name,
                                               subpixel_sampler=subpixel_sampler,
                                               mode="val",
                                               **model_config)
                val_total_loss = np.stack(val_stats).dot(loss_weights)
                val_stats.append(val_total_loss)
                
                tensorboard_tabs = ["scan2model", "model2scan", "normal_loss", "residual_square", "latent_rgl", "pcd_loss"]
                stats = {"train": train_stats, "val": val_stats}

                for split in ["train", "val"]:
                    for (tab, stat) in zip(tensorboard_tabs, stats[split]):
                        writer.add_scalar("{}/{}".format(tab, split), stat, epoch_idx)                
        end = time.time()
        t_total = (end - start) / 60
        print("training finished, duration {:.2f} minutes\n".format(t_total))
        writer.close()

    '''
    ------------ Test model, seen outfits ------------
    '''
    if args.mode.lower() in ['train', 'test', 'test_seen']:

        test_rst_msg = []
        test_rst_msg.append('\n\n{}, epoch={}, test query resolution={} \n'.format(exp_name, epoch_idx, args.query_posmap_size))

        print('\n------------------------Eval on test data, seen outfits, unseen poses...')

        per_outfit_dataset = [{k:v} for k, v in outfits['seen'].items()]

        sum_chamfer_all_outfits, sum_normal_all_outfts, num_ex_all_outfits = 0, 0, 0

        test_rst_msg.append('\tEval on test set, seen clo:\n')

        for outfit in per_outfit_dataset: # outfit is a dict that contains a single key:val pair (a clothing type)

            test_set = Dataset(split='test', outfits=outfit, **dataset_config)
            test_loader = DataLoader(test_set, batch_size=args.batch_size*2, shuffle=False, num_workers=4)

            samples_dir_outfit = os.path.join(samples_dir_test_seen_base, "upsample_beatrice", list(outfit.keys())[0])
            os.makedirs(samples_dir_outfit, exist_ok=True)
            
            start = time.time()
            print("samples dir outfit: {}".format(samples_dir_outfit))
            test_stats = test_seen_clothing( 
                                        model, geometry_feature_map, 
                                        test_loader,
                                        samples_dir_outfit,
                                        epoch_idx,
                                        mode='test_seen',
                                        subpixel_sampler=subpixel_sampler,
                                        model_name=exp_name,
                                        save_all_results=bool(args.save_all_results),
                                        **model_config
                                    )
            test_m2s, test_s2m, test_lnormal, _, _ = test_stats

            # accumulate errors across all outfits
            sum_chamfer_outfit = (test_m2s+test_s2m) * len(test_set) 
            sum_normal_outfit = test_lnormal * len(test_set)

            sum_chamfer_all_outfits += sum_chamfer_outfit
            sum_normal_all_outfts += sum_normal_outfit
            num_ex_all_outfits += len(test_set)

            outfit_info = '{:<18}, {} examples.'.format(list(outfit.keys())[0], len(test_set))
            test_seen_result = "{:<34} m2s dist: {:.3e}, s2m dist: {:.3e}. Chamfer total: {:.3e}, normal loss: {:.3e}.\n"\
                            .format(outfit_info, test_m2s, test_s2m, test_m2s+test_s2m, test_lnormal)
            print(test_seen_result)
            test_rst_msg.append('\t\t{}'.format(test_seen_result))
        
        # calculate the average error across all outfits
        avg_chamfer_all = sum_chamfer_all_outfits / num_ex_all_outfits
        avg_normal_all = sum_normal_all_outfts / num_ex_all_outfits
        test_seen_full_stats = '\t\tOn all seen data, {} exmaples, average Chamfer: {:.3e}, average normal loss: {:.3e}\n'\
            .format(num_ex_all_outfits, avg_chamfer_all, avg_normal_all)
        test_rst_msg.append(test_seen_full_stats)

    '''
    ------------ Test model, unseen outfits ------------
    '''
    if args.mode.lower() in ['test', 'test_unseen']:
        test_rst_msg = []
        test_rst_msg.append('\n\n{}, epoch={}, test query resolution={} \n'.format(exp_name, epoch_idx, args.query_posmap_size))

        print('\n------------------------Eval on test data, unseen outfit, unseen poses...')

        per_outfit_dataset = [{k:v} for k, v in outfits['unseen'].items()]

        test_rst_msg.append('\tEval on test set, unseen clo:')

        for outfit in per_outfit_dataset:
            assert args.num_unseen_frames ==1, "Currently only supports single scan optimization."
            
            print('------Sequence test data for animation:')
            test_set = Dataset(split='test', outfits=outfit, **dataset_config)
            test_loader = DataLoader(test_set, batch_size=args.batch_size*2, shuffle=False, num_workers=4)

            print('------Single frame scan data for optimization:')
            data_spacing_for_optim = len(test_set) // args.num_unseen_frames
            dataset_config['sample_spacing'] = data_spacing_for_optim
            test_set_for_optim = Dataset(split='test', outfits=outfit, **dataset_config)
            test_loader_for_optim = DataLoader(test_set_for_optim, batch_size=args.batch_size, shuffle=False, num_workers=4)

            samples_dir_outfit = os.path.join(samples_dir_test_unseen_base, 'query_resolution{}'.format(args.query_posmap_size))
            
            # loss weights for the optimization w.r.t. the unseen scan
            print("epoch index is {}".format(epoch_idx))
            w_decay_lrd = adjust_loss_weights(args.w_lrd, epoch_idx, mode="decay", start=args.decay_start, every=args.decay_every)
            w_rise_normal = adjust_loss_weights(args.w_normal, epoch_idx, mode="normal", start=args.rise_start, every=args.rise_every)
            # w_pcd_dense = adjust_loss_weights(args.w_ldense, epoch_idx, mode="pcd", start=args.pcd_start)
            w_s2m = adjust_loss_weights(args.w_s2m, epoch_idx, mode="s2m", start=args.s2m_start)
            w_m2s = adjust_loss_weights(args.w_m2s, epoch_idx, mode="m2s", start=args.m2s_start)
            loss_weights = torch.tensor([w_s2m, w_m2s, w_rise_normal, w_decay_lrd, args.w_lrg, args.w_ldense])


            test_stats = test_unseen_clothing(
                                        model,
                                        geometry_feature_map,
                                        test_loader, 
                                        test_loader_for_optim, 
                                        epoch_idx,
                                        samples_dir_outfit,
                                        mode='test_unseen',
                                        model_name=exp_name,
                                        subpixel_sampler=subpixel_sampler,
                                        loss_weights=loss_weights,
                                        num_optim_iterations=args.num_optim_iterations,
                                        random_subsample_scan=bool(args.random_subsample_scan),
                                        save_all_results=bool(args.save_all_results),
                                        **model_config
                                        )

            test_m2s, test_s2m, test_lnormal, _, _ = test_stats

            outfit_info = '{:<18}, {} examples.'.format(list(outfit.keys())[0], len(test_set))
            test_unseen_result = "{:<34} m2s dist: {:.3e}, s2m dist: {:.3e}. Chamfer total: {:.3e}, normal loss: {:.3e}.\n"\
                                            .format(outfit_info, test_m2s, test_s2m, test_m2s+test_s2m, test_lnormal)
            print(test_unseen_result)
            test_rst_msg.append('\t\t{}'.format(test_unseen_result))

if __name__ == '__main__':
    main()