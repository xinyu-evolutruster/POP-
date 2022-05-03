import os

import torch
import numpy as np

def get_face_per_pixel(mask, face_list):
    '''
    :param mask: the uv_mask returned from posmap renderer, where -1 stands for background
                 pixels in the uv map, where other value (int) is the face index that this
                 pixel point corresponds to.
    :param flist: the face list of the body model,
        - smpl, it should be an [13776, 3] array
        - smplx, it should be an [20908,3] array
    :return:
        flist_uv: an [img_size, img_size, 3] array, each pixel is the index of the 3 verts that belong to the triangle
    Note: we set all background (-1) pixels to be 0 to make it easy to parralelize, but later we
        will just mask out these pixels, so it's fine that they are wrong.
    '''
    mask2 = mask.clone()
    mask2[mask == -1] = 0
    face_list_uv = face_list[mask2]
    return face_list_uv

def get_index_map_torch(img, offset=False):
    C, H, W = img.shape
    idx = torch.where(~torch.isnan(img[0]))
    idx = torch.stack(idx)
    if offset:
        idx = idx.float() + 0.5
    idx = idx.view(2, H * W).float().contiguous()
    idx = idx.transpose(0, 1)

    idx = (idx / (H - 1)) if offset == False else (idx / H)
    return idx

def load_masks(PROJECT_DIR, posmap_size, body_model="smpl", device=None):
    if device == None:
        device = torch.device("cpu")
    uv_mask_faceid = os.path.join(PROJECT_DIR, "assets/uv_masks", "uv_mask{}_with_faceid_{}.npy".format(posmap_size, body_model))
    uv_mask_faceid = np.load(uv_mask_faceid)
    uv_mask_faceid = torch.from_numpy(uv_mask_faceid).long().to(device)

    # [256, 256]
    # print(uv_mask_faceid.shape)

    smpl_faces = os.path.join(PROJECT_DIR, "assets", "{}_faces.npy".format(body_model))
    smpl_faces = np.load(smpl_faces)
    face_list = torch.tensor(smpl_faces.astype(np.int32)).long()
    face_list_uv = get_face_per_pixel(uv_mask_faceid, face_list).to(device)

    points_index_from_posmap = (uv_mask_faceid != -1).reshape(-1)

    uv_coord_map = get_index_map_torch(torch.randn(3, posmap_size, posmap_size)).to(device)
    uv_coord_map.requires_grad = True

    return face_list_uv, points_index_from_posmap, uv_coord_map

def load_barycentric_coords(PROJECT_DIR, posmap_size, body_model="smpl", device=None):
    '''
    load the barycentric coordinates (pre-computed and saved) of each pixel 
    on the positional map. Each pixel on the positional map corresponds to 
    a point on the SMPL / SMPL-X body (mesh) which falls into a triangle 
    in the mesh. This function loads the barycentric coordinate of the point 
    in that triangle.
    '''
    if device == None:
        device = torch.device("cpu")
    barycentric = os.path.join(PROJECT_DIR, "assets", "bary_coords_uv_map", 
                               "bary_coords_{}_uv{}.npy".format(body_model, posmap_size))
    barycentric = np.load(barycentric)
    barycentric = barycentric.reshape(posmap_size, posmap_size, 3)
    return torch.from_numpy(barycentric).to(device)

def load_latent_features(file_path, latent_features):
    full_filename = file_path

    if not os.path.isfile(full_filename):
        raise Exception("the file {} does not exist".format(full_filename))
    
    data = torch.load(file_path)
    if isinstance(data["latent_codes"], torch.Tensor):
        latent_features.data[...] = data["latent_codes"].data[...]
    else:
        raise NotImplementedError

    return data["epoch"]

def save_model(path, model, epoch, optimizer=None):
    model_dict = {
        "epoch": epoch,
        "model_state": model.state_dict()
    }
    if optimizer is not None:
        model_dict["optimizer_state"] = optimizer.state_dict()
    torch.save(model_dict, path)

def save_latent_features(path, latent_vectors, epoch):
    if not isinstance(latent_vectors, torch.Tensor):
        all_latents = latent_vectors.state_dict()
    else:
        all_latents = latent_vectors
    
    latent_dict = {
        "epoch": epoch,
        "latent_codes": all_latents
    }
    torch.save(latent_dict, path)

def tensor2numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()

def vertex_normal_2_vertex_color(normals):
    # normalize vertex normal
    if torch.is_tensor(normals):
        normals = normals.detach().cpu().numpy()
    normal_length = (normals ** 2).sum(1)
    normal_length = normal_length ** 0.5
    normal_length = normal_length.reshape(-1, 1)
    normals = normals / normal_length
    # convert normals to colors
    color = normals * 255 / 2.0 + 128
    return color.astype(np.ubyte)

def customized_export_ply(outfile_name, v, f = None, v_n = None, v_c = None, f_c = None, e = None):
    '''
    Author: Jinlong Yang, jyang@tue.mpg.de

    Exports a point cloud / mesh to a .ply file
    supports vertex normal and color export
    such that the saved file will be correctly displayed in MeshLab

    # v: Vertex position, N_v x 3 float numpy array
    # f: Face, N_f x 3 int numpy array
    # v_n: Vertex normal, N_v x 3 float numpy array
    # v_c: Vertex color, N_v x (3 or 4) uchar numpy array
    # f_n: Face normal, N_f x 3 float numpy array
    # f_c: Face color, N_f x (3 or 4) uchar numpy array
    # e: Edge, N_e x 2 int numpy array
    # mode: ascii or binary ply file. Value is {'ascii', 'binary'}
    '''

    v_n_flag=False
    v_c_flag=False
    f_c_flag=False

    N_v = v.shape[0]
    assert(v.shape[1] == 3)
    if not type(v_n) == type(None):
        assert(v_n.shape[0] == N_v)
        if type(v_n) == 'torch.Tensor':
            v_n = v_n.detach().cpu().numpy()
        v_n_flag = True
    if not type(v_c) == type(None):
        assert(v_c.shape[0] == N_v)
        v_c_flag = True
        if v_c.shape[1] == 3:
            # warnings.warn("Vertex color does not provide alpha channel, use default alpha = 255")
            alpha_channel = np.zeros((N_v, 1), dtype = np.ubyte)+255
            v_c = np.hstack((v_c, alpha_channel))

    N_f = 0
    if not type(f) == type(None):
        N_f = f.shape[0]
        assert(f.shape[1] == 3)
        if not type(f_c) == type(None):
            assert(f_c.shape[0] == f.shape[0])
            f_c_flag = True
            if f_c.shape[1] == 3:
                # warnings.warn("Face color does not provide alpha channel, use default alpha = 255")
                alpha_channel = np.zeros((N_f, 1), dtype = np.ubyte)+255
                f_c = np.hstack((f_c, alpha_channel))
    N_e = 0
    if not type(e) == type(None):
        N_e = e.shape[0]

    with open(outfile_name, 'w') as file:
        # Header
        file.write('ply\n')
        file.write('format ascii 1.0\n')
        file.write('element vertex %d\n'%(N_v))
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')

        if v_n_flag:
            file.write('property float nx\n')
            file.write('property float ny\n')
            file.write('property float nz\n')
        if v_c_flag:
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')
            file.write('property uchar alpha\n')

        file.write('element face %d\n'%(N_f))
        file.write('property list uchar int vertex_indices\n')
        if f_c_flag:
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')
            file.write('property uchar alpha\n')

        if not N_e == 0:
            file.write('element edge %d\n'%(N_e))
            file.write('property int vertex1\n')
            file.write('property int vertex2\n')

        file.write('end_header\n')

        # Main body:
        # Vertex
        if v_n_flag and v_c_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %f %f %f %d %d %d %d\n'%\
                    (v[i,0], v[i,1], v[i,2],\
                    v_n[i,0], v_n[i,1], v_n[i,2], \
                    v_c[i,0], v_c[i,1], v_c[i,2], v_c[i,3]))
        elif v_n_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %f %f %f\n'%\
                    (v[i,0], v[i,1], v[i,2],\
                    v_n[i,0], v_n[i,1], v_n[i,2]))
        elif v_c_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %d %d %d %d\n'%\
                    (v[i,0], v[i,1], v[i,2],\
                    v_c[i,0], v_c[i,1], v_c[i,2], v_c[i,3]))
        else:
            for i in range(0, N_v):
                file.write('%f %f %f\n'%\
                    (v[i,0], v[i,1], v[i,2]))
        # Face
        if f_c_flag:
            for i in range(0, N_f):
                file.write('3 %d %d %d %d %d %d %d\n'%\
                    (f[i,0], f[i,1], f[i,2],\
                    f_c[i,0], f_c[i,1], f_c[i,2], f_c[i,3]))
        else:
            for i in range(0, N_f):
                file.write('3 %d %d %d\n'%\
                    (f[i,0], f[i,1], f[i,2]))

        # Edge
        if not N_e == 0:
            for i in range(0, N_e):
                file.write('%d %d\n'%(e[i,0], e[i,1]))

def save_result_examples(save_dir, model_name, result_name, points,
                         normals=None, patch_color=None, texture=None, 
                         coarse_pts=None, gt=None, epoch=None):

    if epoch == None:
        normals_file_name = "{}_{}_pred.ply".format(model_name, result_name)
    else:
        print("normals are not None")
        normals_file_name = "{}_epoch{}_{}_pred.ply".format(model_name, str(epoch).zfill(5), result_name)
    normals_file_name = os.path.join(save_dir, normals_file_name)

    points = tensor2numpy(points)

    if normals is not None:
        print("normals are not None 2")
        print("normals_file_name is {}".format(normals_file_name))
        normals = tensor2numpy(normals)
        color_normals = vertex_normal_2_vertex_color(normals)
        customized_export_ply(normals_file_name, v=points, v_n=normals, v_c=color_normals)

    if patch_color is not None:
        patch_color = tensor2numpy(patch_color)
        if patch_color.max() < 1.1:
            patch_color = (patch_color * 255.).astype(np.ubyte)
        pcolor_file_name = normals_file_name.replace("pred.ply", "pred_patchcolor.ply")
        customized_export_ply(pcolor_file_name, v=points, v_c=patch_color)
    
    if texture is not None:
        texture = tensor2numpy(texture)
        if texture.max() < 1.1:
            texture = (texture * 255.).astype(np.ubyte)
        texture_file_name = normals_file_name.replace("pred.ply", "pred_texture.ply")
        customized_export_ply(texture_file_name, v=points, v_c=texture)
    
    if coarse_pts is not None:
        coarse_pts = tensor2numpy(coarse_pts)
        coarse_file_name = normals_file_name.replace("pred.ply", "interm.ply")
        customized_export_ply(coarse_file_name, v=coarse_pts)
    
    if gt is not None:
        gt = tensor2numpy(gt)
        gt_file_name = normals_file_name.replace("pred.ply", "gt.ply")
        customized_export_ply(gt_file_name, v=gt)