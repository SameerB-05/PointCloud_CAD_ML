import open3d as o3d

# Path to the output PLY file with color-coded clusters
ply_file_path = r"C:\Users\samee\Documents\IITB_INTERN\HPNet\OUTPUTS\abc_00470_normal_output8.ply"

# Load the point cloud
pcd = o3d.io.read_point_cloud(ply_file_path)

# Sanity check: print point and color info
print(f"Loaded {len(pcd.points)} points")
print(f"Has colors? {'Yes' if pcd.has_colors() else 'No'}")




# Visualize the point cloud with colors
o3d.visualization.draw_geometries(
    [pcd],
    window_name="Segmented Output - Cluster Colors",
    width=900,
    height=700,
    point_show_normal=False
)
