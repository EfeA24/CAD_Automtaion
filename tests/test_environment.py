import pytest
import torch

from pipeline.text_to_cad import DEFAULT_CHECKPOINT


def test_occ_static_method_api_available():
    """The pythonocc 7.9 API that the Text2CAD OCC patch relies on."""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopoDS import topods
    from OCC.Display.SimpleGui import init_display  # noqa: F401
    from OCC.Extend.DataExchange import write_step_file  # noqa: F401

    for fn in (
        breptools.Write,
        breptools.UVBounds,
        brepgprop.VolumeProperties,
        brepgprop.SurfaceProperties,
        brepgprop.LinearProperties,
        brepbndlib.Add,
        topexp.MapShapesAndAncestors,
        topexp.CommonVertex,
        topods.Edge,
        topods.Vertex,
        BRep_Tool.Surface,
        BRep_Tool.Continuity,
    ):
        assert callable(fn)


def test_text2cad_import_chain():
    from CadSeqProc.cad_sequence import CADSequence  # noqa: F401
    from Cad_VLM.models.text2cad import Text2CAD  # noqa: F401


def test_cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device; TextToCAD will fall back to CPU")
    assert torch.cuda.get_device_properties(0).total_memory > 4 * 1024**3


def test_checkpoint_present():
    assert DEFAULT_CHECKPOINT.exists(), f"download Text2CAD_1.0.pth to {DEFAULT_CHECKPOINT}"
