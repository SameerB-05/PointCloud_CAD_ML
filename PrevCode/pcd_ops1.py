# Streamlined Point Cloud Processing and Analysis
# By Sameer


import open3d as o3d
import numpy as np
import pyvista as pv
import trimesh
import os
import sys
import matplotlib.pyplot as plt
import pymeshfix as mf
from prettytable import PrettyTable

def preprocess_point_cloud(pcd):

    # denoise the point cloud
    nn = 40 # nearest neighboors
    nsigma = 10 # std dev multiplier, lower the number the more aggressive the denoising

    filtered_pcd = pcd.remove_statistical_outlier(nn, nsigma)
    outliers = pcd.select_by_index(filtered_pcd[1], invert = True)
    outliers.paint_uniform_color([1,0,0])
    filtered_pcd = filtered_pcd[0]
    print("Number of points after denoising:", len(filtered_pcd.points))

    # downsample the point cloud
    vox_size = 0.005
    downsampled_pcd = filtered_pcd.voxel_down_sample(vox_size)
    print("Number of points after downsampling:", len(downsampled_pcd.points))

    # estimate normals
    avg_dist = np.mean(downsampled_pcd.compute_nearest_neighbor_distance())
    radius_normals = avg_dist * 4

    downsampled_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normals, max_nn=30))
    downsampled_pcd.orient_normals_consistent_tangent_plane(200)

    return downsampled_pcd




def ransac(downsampled_pcd):
    pt_to_plane = 0.02

    plane_model, inliers = downsampled_pcd.segment_plane(distance_threshold = pt_to_plane, ransac_n = 3, num_iterations = 1000)

    [a, b, c, d] = plane_model

    print(f"Plane Equation: {a:.2f}x + {b:.2f}y +{c:.2f}z + {d:.2f} = 0")

    inliercloud = downsampled_pcd.select_by_index(inliers)
    outliercloud = downsampled_pcd.select_by_index(inliers, invert = True)

    inliercloud.paint_uniform_color([1,0,0])
    outliercloud.paint_uniform_color([0.6,0.6,0.6])
    o3d.visualization.draw_geometries([inliercloud, outliercloud])


def automated_ransac(downsampled_pcd):
    max_plane_ind = 6
    pt_to_plane_dist = 0.01

    segment_models = {}
    segment = {}
    rest = downsampled_pcd

    for i in range(max_plane_ind):
        if len(rest.points) < 3:
            print(f"RANSAC stopped at iteration {i}: not enough points to fit a plane.")
            break

        colors = plt.get_cmap("tab20")(i)
        segment_models[i], inliers = rest.segment_plane(
            distance_threshold=pt_to_plane_dist,
            ransac_n=3,
            num_iterations=1000
        )
        segment[i] = rest.select_by_index(inliers)
        segment[i].paint_uniform_color(list(colors[:3]))
        rest = rest.select_by_index(inliers, invert=True)

    o3d.visualization.draw_geometries([segment[i] for i in segment] + [rest])


def dbscan(downsampled_pcd):
    rest = downsampled_pcd
    print(np.asarray(rest.points).shape)

    labels = np.array(rest.cluster_dbscan(eps = 1, min_points = 5))

    max_labels = labels.max()
    print(f"The point cloud has {max_labels +1} clusters")

    colors = plt.get_cmap("tab10")(labels/(max_labels if max_labels>0 else 1))
    colors[labels < 0]

    rest.colors = o3d.utility.Vector3dVector(colors[:,:3])
    #o3d.visualization.draw_geometries([segment[i] for i in range (max_plane_ind)] + [rest])

    o3d.visualization.draw_geometries([rest])




def surface_reconstruction(pcd):

    # surface reconstruction using Ball Pivoting Algorithm (BPA)
    avg_dist = np.mean(pcd.compute_nearest_neighbor_distance())
    radius = 3*avg_dist
    mesh_in = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,
                                        o3d.utility.DoubleVector([radius, radius*2]))
    
    # draw_geometries can be heavy; wrap it in try/except
    try:
        o3d.visualization.draw_geometries([mesh_in])
    except Exception as e:
        print("Draw_geometries failed:", e)

    # surface smoothing
    mesh = mesh_in.filter_smooth_simple(number_of_iterations=1)

    # converting open3d mesh into trimesh object
    # disable process so .show() won’t trigger an internal repair pass
    objmesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_normals=np.asarray(mesh.vertex_normals),
        process=False
    )

    objmesh.units = trimesh.units.units_from_metadata(objmesh, guess=True)
    print("The mesh currently has the unit: ", objmesh.units)

    if objmesh.units is None:
        objmesh.units = "mm"
        print("The mesh unit has been changed to: ", objmesh.units)

    brkn = trimesh.repair.broken_faces(objmesh)
    print(brkn)
    print("Number of faces that break the mesh are (non manifold, degenerate triangles): ", len(brkn))

    # REPLACED trimesh.show() with pyvista show
    # because trimesh.show() was not working properly
    # and trimesh_obj can be easily wrapped in pyvista, while not in open3d
    try:
        pv_mesh = pv.wrap(objmesh)
        p = pv.Plotter(title="Surface Reconstructed Mesh (PyVista Visualization)")
        p.add_mesh(pv_mesh, color='lightgrey', show_edges=False)
        p.enable_eye_dome_lighting()
        p.camera_position = 'iso'
        p.show()
    except Exception as e:
        print("PyVista show failed:", e)
    
    #trimesh.scene.lighting.SpotLight(objmesh, intensity = 50)
    #objmesh.show()

    # displaying broken facets/edges of the model
    if not objmesh.is_watertight:
        print("Mesh is not continuous. Holes in the mesh are highlighted in red:")
        objpv = pv.wrap(objmesh)
        meshfix = mf.MeshFix(objpv)

        # wrap hole extraction & show in try/except
        try:
            holes = meshfix.extract_holes()
            p = pv.Plotter(title="Holes in Mesh (PyVista Visualization)")
            p.add_mesh(objpv, color=True)
            p.add_mesh(holes, color="r", line_width=8)
            p.enable_eye_dome_lighting()  # helps depth perception
            p.camera_position = 'iso'
            p.show()  # default backend instead of 'trame'
        except Exception as e:
            print("Hole-extraction visualization failed:", e)
    else:
        print("Mesh is already watertight")

    return objmesh, meshfix



def fix_mesh(objmesh, meshfix):

    if not objmesh.is_watertight:
        meshfix.repair(verbose=False)
        repaired = meshfix.mesh
        print(repaired)
        faces_as_array = repaired.faces.reshape((repaired.n_faces, 4))[:, 1:] 
        acl = trimesh.Trimesh(repaired.points, faces_as_array)
        acl.units = trimesh.units.units_from_metadata(acl, guess = True)
    else:
        acl = objmesh
    
    try:
        pv_fixed = pv.wrap(acl)
        plotter = pv.Plotter(title="Fixed Mesh (PyVista Visualization)")
        plotter.add_mesh(pv_fixed, show_edges=False)
        plotter.enable_eye_dome_lighting()
        plotter.camera_position = 'iso'
        plotter.show()
    except Exception as e:
        print("PyVista show() failed on fixed mesh:", e)

    # acl.show(smooth = False)
    print("Mesh is watertight: ",acl.is_watertight)
    print("Volume of the object is: ",acl.volume, acl.units + "³")
    print("Area of the object is: ",acl.area, acl.units + "²")
    return acl



def get_closureSet_params(acl):

    vertacl = np.array(acl.vertices)

    covmat_acl = np.cov(vertacl, rowvar=False)

    eigvalacl, eigvecacl = np.linalg.eigh(covmat_acl)

    sortindacl = np.argsort(eigvalacl)[::-1]
    eigvecacl_sorted = eigvecacl[:, sortindacl]
    eigvalacl_sorted = eigvalacl[sortindacl]

    majaxis_acl = np.abs(eigvecacl_sorted[:, 0])
    majlen_acl = np.sqrt(eigvalacl_sorted[0])

    minaxis_acl = np.abs(eigvecacl_sorted[:, 2]) 
    minlen_acl = np.sqrt(eigvalacl_sorted[2])

    print("Major Axis:", majaxis_acl)
    print("Major Length:", majlen_acl, acl.units)
    print("Minor Axis:", minaxis_acl)
    print("Minor Length:", minlen_acl, acl.units)

    origin = acl.centroid
    print("Centroid: ", acl.centroid)

    #code for table of contents for closure set (acl)
    acl_table = PrettyTable(["Parameter", "Value", "Unit"])
    acl_table.add_row(["No open surfaces", acl.is_watertight, "-"])
    # acl_table.add_row(["Feret max", bmax_acl, acl.units])
    # acl_table.add_row(["Feret min", bmin_acl, acl.units])
    acl_table.add_row(["Major axis", majlen_acl, acl.units])
    acl_table.add_row(["Minor axis", minlen_acl, acl.units])
    acl_table.add_row(["Lebesgue Measure", acl.volume, acl.units + "³"])
    acl_table.add_row(["Minkowski Functional", acl.area, acl.units + "²"])
    print(acl_table)

    return {"volume": acl.volume, "area": acl.area, "units": acl.units,
            "majaxis": majaxis_acl, "majlen": majlen_acl,
            "minaxis": minaxis_acl, "minlen": minlen_acl}



def get_boundingHyperrect_params(acl):
    # Compute the minimum oriented bounding box
    abh = acl.bounding_box_oriented

    # Access the vertices and rotation matrix of the bounding box
    abh_vertices = abh.vertices
    abh_rotation_matrix = abh.primitive.transform
    abh.units = trimesh.units.units_from_metadata(abh, guess = True)

    """
    # Visualize the mesh and the bounding box
    scene = trimesh.Scene([acl, abh.as_outline()])
    scene.show()
    #current the bounding box outline is in white colour, looking for a function that can change the colour, rotating the component can help in seeing the fine white edges
    """

    # Use pv.wrap to convert acl to PyVista
    pv_acl = pv.wrap(acl)

    # Convert the outline of the bounding box to PyVista line segments
    outline = abh.as_outline()
    outline_lines = []
    for entity in outline.entities:
        indices = entity.points
        points = outline.vertices[indices]
        for i in range(len(points) - 1):
            line = pv.Line(points[i], points[i + 1])
            outline_lines.append(line)

    # Setup plotter
    plotter = pv.Plotter(title="Bounding Hyperrectangle (PyVista Visualization)")
    plotter.add_mesh(pv_acl, color="lightgray", show_edges=False, opacity=0.8)
    for line in outline_lines:
        plotter.add_mesh(line, color="red", line_width=2)

    plotter.enable_eye_dome_lighting()
    plotter.camera_position = 'iso'
    plotter.show()
    
    print("Bounding hyperrectangle is watertight: ",abh.is_watertight)
    print("Volume of the bounding hyperrectangle is: ", abh.volume, abh.units + "³")
    print("Area of the bounding hyperrectangle is: ",abh.area, abh.units + "²")

    vertabh = np.array(abh.vertices)
    # Initialize variables to store the minimum and maximum Feret values
    bmin_abh = float('inf')
    bmax_abh = 0.0

    # Iterate through all edges of the mesh
    for edge in abh.edges_unique:
        edge_length = np.linalg.norm(vertabh[edge[0]] - vertabh[edge[1]])
        
        bmin_abh = min(bmin_abh, edge_length)
        bmax_abh = max(bmax_abh, edge_length)
    print("Feret Min:", bmin_abh, abh.units)
    print("Feret Max:", bmax_abh, abh.units)

    covmat_abh = np.cov(vertabh, rowvar=False)

    eigvalabh, eigvecabh = np.linalg.eigh(covmat_abh)

    sortindabh = np.argsort(eigvalabh)[::-1]
    eigvecabh_sorted = eigvecabh[:, sortindabh]
    eigvalabh_sorted = eigvalabh[sortindabh]

    majaxis_abh = np.abs(eigvecabh_sorted[:, 0])
    majlen_abh = np.sqrt(eigvalabh_sorted[0])

    minaxis_abh = np.abs(eigvecabh_sorted[:, 2]) 
    minlen_abh = np.sqrt(eigvalabh_sorted[2])

    print("Major Axis:", majaxis_abh)
    print("Major Length:", majlen_abh, abh.units)
    print("Minor Axis:", minaxis_abh)
    print("Minor Length:", minlen_abh, abh.units)

    #code for table of contents for bounding hyperrectangle (abh)
    abh_table = PrettyTable(["Parameter", "Value", "Unit"])
    abh_table.add_row(["No open surfaces", abh.is_watertight, "-"])
    # abh_table.add_row(["Feret max", bmax_abh, abh.units])
    # abh_table.add_row(["Feret min", bmin_abh, abh.units])
    abh_table.add_row(["Major axis", majlen_abh, abh.units])
    abh_table.add_row(["Minor axis", minlen_abh, abh.units])
    abh_table.add_row(["Lebesgue Measure", abh.volume, abh.units + "³"])
    abh_table.add_row(["Minkowski Functional", abh.area, abh.units + "²"])
    print(abh_table)

    return {"volume": abh.volume, "area": abh.area, "units": abh.units,
            "majaxis": majaxis_abh, "majlen": majlen_abh,
            "minaxis": minaxis_abh, "minlen": minlen_abh}




def get_convexHull_params(acl):
    ach = trimesh.convex.convex_hull(acl)
    ach.units = trimesh.units.units_from_metadata(ach, guess = True)
    pv_fixed = pv.wrap(ach)
    plotter = pv.Plotter(title="Convex Hull (PyVista Visualization)")
    plotter.add_mesh(pv_fixed, show_edges=False)
    plotter.enable_eye_dome_lighting()
    plotter.camera_position = 'iso'
    plotter.show()

    print("Convex hull is watertight: ",ach.is_watertight)
    print("Volume of the convex hull is: ",ach.volume, ach.units + "³")
    print("Area of the convex hull is: ",ach.area, ach.units + "²")

    vertach = np.array(ach.vertices)

    covmat_ach = np.cov(vertach, rowvar=False)

    eigvalach, eigvecach = np.linalg.eigh(covmat_ach)

    sortindach = np.argsort(eigvalach)[::-1]
    eigvecach_sorted = eigvecach[:, sortindach]
    eigvalach_sorted = eigvalach[sortindach]

    majaxis_ach = np.abs(eigvecach_sorted[:, 0])
    majlen_ach = np.sqrt(eigvalach_sorted[0])

    minaxis_ach = np.abs(eigvecach_sorted[:, 2])
    minlen_ach = np.sqrt(eigvalach_sorted[2])

    print("Major Axis:", majaxis_ach)
    print("Major Length:", majlen_ach, ach.units)
    print("Minor Axis:", minaxis_ach)
    print("Minor Length:", minlen_ach, ach.units)

    #code for table of contents for convex hull(ach)
    ach_table = PrettyTable(["Parameter", "Value", "Unit"])
    ach_table.add_row(["No open surfaces", ach.is_watertight, "-"])
    # ach_table.add_row(["Feret max", bmax_ach, ach.units])
    # ach_table.add_row(["Feret min", bmin_ach, ach.units])
    ach_table.add_row(["Major axis", majlen_ach, ach.units])
    ach_table.add_row(["Minor axis", minlen_ach, ach.units])
    ach_table.add_row(["Lebesgue Measure", ach.volume, ach.units + "³"])
    ach_table.add_row(["Minkowski Functional", ach.area, ach.units + "²"])
    print(ach_table)

    return {"volume": ach.volume, "area": ach.area, "units": ach.units,
            "majaxis": majaxis_ach, "majlen": majlen_ach,
            "minaxis": minaxis_ach, "minlen": minlen_ach}



def get_invariant_params(acl_params, abh_params, ach_params):
    solidity = acl_params["volume"] / ach_params["volume"]
    anisotropy = acl_params["majlen"] / acl_params["minlen"]
    rectangularity = abh_params["minlen"] / abh_params["majlen"]
    compactness = acl_params["volume"] / (acl_params["area"] ** 1.5)
    roundness = acl_params["volume"] / (ach_params["area"] ** 1.5)
    convexity = ach_params["area"] / acl_params["area"]

    diminv_table = PrettyTable(["Dimensionless Invariant Stereometric features", "Value"])
    diminv_table.add_row(["Compactness", compactness])
    diminv_table.add_row(["Roundness", roundness])
    diminv_table.add_row(["Convexity", convexity])
    #diminv_table.add_row(["Sphericity", ""])
    diminv_table.add_row(["Solidity", solidity])
    diminv_table.add_row(["Anisotropy", anisotropy])
    diminv_table.add_row(["Rectangularity", rectangularity])
    #diminv_table.add_row(["Second order moment invariant (J1)", ""])
    #diminv_table.add_row(["Second order moment invariant (J2)", ""])
    #diminv_table.add_row(["Second order moment invariant (J3)", ""])

    print(diminv_table)



def create_gif(acl, majaxis_acl):
    #Creating a GIF where the object rotates about its major axis
    mesh = pv.wrap(acl)
    major_axis = majaxis_acl

    plotter = pv.Plotter(window_size = [1180, 1180])
    plotter.add_mesh(mesh, show_edges = True)
    plotter.camera_position = 'xy'
    #plotter.camera.elevation = 30
    plotter.enable_eye_dome_lighting()

    n_frames = 36

    rotations = [i * 360 / n_frames for i in range(n_frames)]

    plotter.open_gif('meshrotate_majaxis.gif')

    for angle in rotations:
        plotter.clear()
        rotated_mesh = mesh.copy().rotate_vector(major_axis, angle, point = mesh.center)
        plotter.add_mesh(rotated_mesh)
        plotter.write_frame()

    plotter.close()



if __name__ == "__main__":

    file_path = r"C:\Users\samee\OneDrive\Documents\INTERN_IITB\Armadillo_PC.ply"
    pcd = o3d.io.read_point_cloud(file_path)
    print("Number of points:", len(pcd.points))

    # Input point cloud visualization
    o3d.visualization.draw_geometries([pcd])

    
    downsampled_pcd = preprocess_point_cloud(pcd)
    o3d.visualization.draw_geometries([downsampled_pcd]) # visualize the preprocessed point cloud
    print("--------Preprocessing complete--------\n")

    objmesh, meshfix = surface_reconstruction(downsampled_pcd)
    print("--------Surface reconstruction complete--------\n")

    fixed_mesh = fix_mesh(objmesh, meshfix)
    print("--------Mesh fixing complete--------\n")

    acl_params = get_closureSet_params(fixed_mesh)
    print("--------Closure set parameters obtained--------\n")

    abh_params = get_boundingHyperrect_params(fixed_mesh)
    print("--------Bounding hyperrectangle parameters obtained--------\n")

    ach_params = get_convexHull_params(fixed_mesh)
    print("--------Convex hull parameters obtained--------\n")

    get_invariant_params(acl_params, abh_params, ach_params)
    print("--------Dimensionless invariant stereometric features obtained--------\n")


    