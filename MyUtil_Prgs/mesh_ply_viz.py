import open3d as o3d

# Change to 'clipped' if you want clipped mesh instead
mesh_path = r"C:\Users\samee\Documents\IITB_INTERN\point2cad\out\unclipped\mesh_abc_00470.ply"

# Read the mesh
mesh = o3d.io.read_triangle_mesh(mesh_path)

# Check if mesh is loaded
if not mesh.has_triangles():
    print("Error: No triangles found in the mesh.")
else:
    print("Mesh loaded successfully!")
    print(mesh)

    # Visualize it
    o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True, mesh_show_back_face=True)
