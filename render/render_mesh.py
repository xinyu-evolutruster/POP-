import numpy as np
import open3d as o3d

# print(o3d.__file__)

import os

def main():
    result_dir = "/root/myPOP/results/rendered_imgs/saved_samples/upsample_beatrice/test_seen/query_resolution256"
    # pcd_file_path = os.path.join(result_dir, "epoch00331", "epoch00600_01_pred.ply")
    # ply_file_path = os.path.join(result_dir, "epoch00331", "epoch00600_01_pred_mesh.ply")
    pcd_file_path = os.path.join(result_dir, "epoch00243_01_pred.ply")
    ply_file_path = os.path.join(result_dir, "epoch00243_01_pred_mesh.ply")

    # ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(pcd_file_path)
    
    print('run Poisson surface reconstruction')
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    print(mesh)
    o3d.io.write_triangle_mesh(ply_file_path, mesh, write_vertex_normals=False)

if __name__ == '__main__':
    main()