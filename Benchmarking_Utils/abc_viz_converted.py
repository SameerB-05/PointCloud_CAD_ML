import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def visualize_xyz(file_path, color=[0.1, 0.6, 1.0]):

    # Load XYZ point cloud
    points = np.loadtxt(file_path)
    if points.shape[1] > 3:
        points = points[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    print(f"Loaded {len(pcd.points)} points from {file_path}")

    # Set uniform color
    pcd.paint_uniform_color(color)

    # Visualize
    o3d.visualization.draw_geometries([pcd], window_name=file_path.split('\\')[-1])

def visualize_xyzc(file_path):
    # Load XYZC point cloud
    data = np.loadtxt(file_path)
    points = data[:, :3]
    labels = data[:, 3].astype(int)

    # Assign color per label (up to 20 unique colors)
    unique_labels = np.unique(labels)
    color_map = plt.get_cmap('tab20')
    colors = color_map(labels % 20)[:, :3]  # RGB only

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    o3d.visualization.draw_geometries([pcd], window_name=file_path.split('\\')[-1])

# Paths
unlabeled_path = r"C:\Users\samee\Documents\IITB_INTERN\ABC_Dataset_Samples\Objects\00000004\pointcloud_unlabeled2.xyz"
semantic_path = r"C:\Users\samee\Documents\IITB_INTERN\ABC_Dataset_Samples\Objects\00000003\pointcloud_semantic_labelled.xyzc"

# Call visualizers
visualize_xyz(unlabeled_path)
#visualize_xyzc(semantic_path)
