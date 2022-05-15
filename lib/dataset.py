import torch
from torch.utils.data import Dataset
import numpy as np

import os
import glob
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

class Dataset(Dataset):
    def __init__(self, data_root=None, 
                 split="train", 
                 outfits={},
                 query_posmap_size=256, 
                 meanshape_posmap_size=128,
                 sample_spacing=1,
                 dataset_subset_portion=1.0):
        self.data_root = data_root
        self.data_dirs = {}
        for outfit in outfits.keys():
            self.data_dirs[outfit] = os.path.join(self.data_root, outfit, split)
            print(self.data_dirs[outfit])

        self.query_posmap_size = query_posmap_size
        self.meanshape_posmap_size = meanshape_posmap_size

        self.spacing = sample_spacing
        self.dataset_subset_portion = dataset_subset_portion
        self.split = split

        assets_path = os.path.join(SCRIPT_DIR, "..", "assets")
        self.body_model = os.path.join(assets_path, "smplx_faces.npy") # use smplx model
        self.clothing_label_def = outfits

        # input position maps
        self.posmap, self.posmap_meanshape = [], []
        # ground truth point cloud and normals
        self.scan_pc, self.scan_normals, self.scan_names = [], [], []

        self.clothing_labels = []
        self.body_verts = []
        self.vtransf = []

        print("Start to initialize the dataset...")
        self._init_dataset()

        self.data_size = int(len(self.posmap))
        print("Data loaded, in total {} {} examples".format(self.data_size, self.split))

    def _init_dataset(self):
        
        file_list_all = []
        subject_id_all = []

        print(self.dataset_subset_portion)

        for outfit_id, (outfit, outfit_datadir) in enumerate(self.data_dirs.items()):
            flist = sorted(glob.glob(os.path.join(outfit_datadir, '*.npz')))[::self.spacing]
            print('Loading {}, {} examples..'.format(outfit, len(flist)))
            file_list_all = file_list_all + flist
            subject_id_all = subject_id_all + [outfit.split('_')[0]] * len(flist)

        # file_list_all = file_list_all[:16]

        if self.dataset_subset_portion < 1.0:
            import random
            random.shuffle(file_list_all)
            num_total = len(file_list_all)
            num_chosen = int(self.dataset_subset_portion*num_total)
            file_list_all = file_list_all[:num_chosen]
            print('Total examples: {}, now only randomly sample {} from them...'.format(num_total, num_chosen))

        for idx, file_name in enumerate(tqdm(file_list_all)):
            # if idx > 100: 
            #     break
            data = np.load(file_name)
            clothing_type = os.path.dirname(file_name).split('/')[-2]  # e.g. rp_aaron_posed_002
            clothing_label = self.clothing_label_def[clothing_type]
            self.clothing_labels.append(clothing_label)

            self.posmap.append(torch.tensor(data["posmap{}".format(self.query_posmap_size)]).float().permute([2, 0, 1]))
            
            if "posmap_canonical{}".format(self.meanshape_posmap_size) not in data.files:
                self.posmap_meanshape.append(
                    torch.tensor(data["posmap{}".format(self.meanshape_posmap_size)]).float().permute([2, 0, 1])
                )
            else:
                self.posmap_meanshape.append(
                    torch.tensor(data["posmap_canonical{}".format(self.meanshape_posmap_size)]).float().permute([2, 0, 1])
                )

            scan_name = str(data["scan_name"])
            self.scan_names.append(scan_name)

            self.body_verts.append(torch.tensor(data["body_verts"]).float())
            self.scan_pc.append(torch.tensor(data["scan_pc"]).float())
            self.scan_normals.append(torch.tensor(data["scan_n"]).float())

            vtransf = torch.tensor(data["vtransf"]).float()
            if vtransf.shape[-1] == 4:
                vtransf = vtransf[:, :3, :3]
            self.vtransf.append(vtransf)

    def __getitem__(self, index):
        posmap = self.posmap[index]
        posmap_meanshape = self.posmap_meanshape[index]
        body_verts = self.body_verts[index]
        scan_pc = self.scan_pc[index]
        scan_normals = self.scan_normals[index]
        scan_name = self.scan_names[index]

        clothing_label = self.clothing_labels[index]
        vtransf = self.vtransf[index]

        return posmap, posmap_meanshape, scan_pc, scan_normals, body_verts, scan_name, vtransf, clothing_label

    def __len__(self):
        return self.data_size