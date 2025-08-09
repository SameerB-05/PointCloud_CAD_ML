import open3d as o3d
import h5py
import numpy as np

# Load the HDF5 file
h5_path = r"C:\Users\samee\Documents\IITB_INTERN\HPNet\INPUTS\Part_01_pointcloud_normal.h5"
with h5py.File(h5_path, 'r') as f:
    xyz = f['xyz'][:]
    normal = f['normal'][:]

# Create Open3D PointCloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.normals = o3d.utility.Vector3dVector(normal)

print(f"Loaded {len(pcd.points)} points with normals")
# (Optional) Estimate color based on normals (for visualization)
colors = (normal + 1.0) / 2.0  # Normalize normals to [0,1]
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize
o3d.visualization.draw_geometries([pcd], point_show_normal=True)
