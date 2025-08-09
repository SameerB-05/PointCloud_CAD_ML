try:
    import pyvista as pv
    print("✅ PyVista installed:", pv.__version__)
except ImportError as e:
    print("❌ PyVista not installed:", e)

try:
    import trimesh
    print("✅ Trimesh installed:", trimesh.__version__)
except ImportError as e:
    print("❌ Trimesh not installed:", e)

try:
    import pymeshfix
    print("✅ PyMeshFix installed")
except ImportError as e:
    print("❌ PyMeshFix not installed:", e)

try:
    from prettytable import PrettyTable
    print("✅ PrettyTable installed")
except ImportError as e:
    print("❌ PrettyTable not installed:", e)
