from collections import Counter
from dataclasses import dataclass

import pytest
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Extend.DataExchange import read_step_file

from pipeline.text_to_cad import TextToCAD


@dataclass
class StepInfo:
    valid: bool
    solids: int
    faces: Counter
    size: tuple[float, float, float]
    volume: float


def inspect_step(path) -> StepInfo:
    shape = read_step_file(str(path))

    faces = Counter()
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        kind = BRepAdaptor_Surface(topods.Face(exp.Current())).GetType()
        faces["plane" if kind == GeomAbs_Plane else "cylinder" if kind == GeomAbs_Cylinder else "other"] += 1
        exp.Next()

    solids = 0
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        solids += 1
        exp.Next()

    box = Bnd_Box()
    brepbndlib.Add(shape, box, False)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)

    return StepInfo(
        valid=BRepCheck_Analyzer(shape).IsValid(),
        solids=solids,
        faces=faces,
        size=(xmax - xmin, ymax - ymin, zmax - zmin),
        volume=props.Mass(),
    )


@pytest.fixture(scope="session")
def cad():
    return TextToCAD()
