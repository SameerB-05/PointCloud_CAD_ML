import open3d as o3d

# Path to your point cloud file
file_path = r"C:\Users\samee\Documents\IITB_INTERN\point2cad\BenchmarkingInputs\abc_00949.xyz"
file_name = file_path.split("\\")[-1].replace(".xyz", "")
# Load the point cloud
pcd = o3d.io.read_point_cloud(file_path)

# Print some info about the point cloud
print(pcd)
print("Number of points:", len(pcd.points))

# Visualize
o3d.visualization.draw_geometries([pcd])


"""# Downsample to 10,000 points
pcd_down = pcd.farthest_point_down_sample(10000)

print("Number of points in downsampled:", len(pcd_down.points))
print
# Save or visualize
o3d.io.write_point_cloud(f"point2cad/assets/{file_name}_downsampled.xyz", pcd_down, write_ascii=True)
o3d.visualization.draw_geometries([pcd_down])"""