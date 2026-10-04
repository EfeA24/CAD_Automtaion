
import sys
from collections import Counter
from pathlib import Path

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCC.Core.GProp import GProp_GProps
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Extend.DataExchange import read_step_file

step_path, png_path = sys.argv[1], sys.argv[2]
shape = read_step_file(step_path)

kinds = Counter()
exp = TopExp_Explorer(shape, TopAbs_FACE)
while exp.More():
    t = BRepAdaptor_Surface(topods.Face(exp.Current())).GetType()
    kinds["plane" if t == GeomAbs_Plane else "cylinder" if t == GeomAbs_Cylinder else str(t)] += 1
    exp.Next()

solids = 0
exp = TopExp_Explorer(shape, TopAbs_SOLID)
while exp.More():
    solids += 1
    exp.Next()

box = Bnd_Box()
brepbndlib.Add(shape, box)
xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
props = GProp_GProps()
brepgprop.VolumeProperties(shape, props)
size = (xmax - xmin, ymax - ymin, zmax - zmin)

print("solids:", solids)
print("faces:", dict(kinds))
print("bbox size: %.3f x %.3f x %.3f" % size)
print("volume: %.4f  (bbox volume %.4f, fill %.1f%%)" % (props.Mass(), size[0] * size[1] * size[2], 100 * props.Mass() / (size[0] * size[1] * size[2])))

BRepMesh_IncrementalMesh(shape, 0.002, False, 0.3, True)
stl_path = str(Path(step_path).with_suffix(".stl"))
StlAPI_Writer().Write(shape, stl_path)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

mesh = trimesh.load(stl_path)
fig = plt.figure(figsize=(10, 5))
for i, (elev, azim) in enumerate([(30, 45), (90, -90)]):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    ax.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces], facecolor="#8fb3d9", edgecolor="#2d4a6b", linewidth=0.1))
    ax.auto_scale_xyz(*[(mesh.bounds[0][k], mesh.bounds[1][k]) for k in range(3)])
    ax.set_box_aspect(list(mesh.extents))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title("isometric" if i == 0 else "top view")
fig.suptitle(Path(step_path).name)
plt.tight_layout()
plt.savefig(png_path, dpi=110)
print("preview:", png_path)
