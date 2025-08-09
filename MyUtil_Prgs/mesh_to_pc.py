import open3d as o3d
import os

def stl_to_ply(stl_path, output_dir=None, n_points=10000):
    # Load the STL mesh
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.compute_vertex_normals()

    # Convert mesh to point cloud
    pcd = mesh.sample_points_poisson_disk(number_of_points=n_points)

    print(len(pcd.points), " points sampled from the mesh.")
    # Generate output path
    filename = os.path.splitext(os.path.basename(stl_path))[0] + "_pointcloud.xyz"
    if output_dir is None:
        output_path = os.path.join(os.path.dirname(stl_path), filename)
    else:
        output_path = os.path.join(output_dir, filename)

    # Save point cloud
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"✅ Saved point cloud to: {output_path}")
    o3d.visualization.draw_geometries([pcd])


# Replace this with any of your STL filenames
file_path = r"C:\Users\samee\OneDrive\Documents\INTERN_IITB\Armadillo.ply"

stl_to_ply(file_path, output_dir=None, n_points=30000)