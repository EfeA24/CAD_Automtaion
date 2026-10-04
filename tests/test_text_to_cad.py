import time

import numpy as np
import pytest

from pipeline.text_to_cad import InvalidCADError, TextToCAD
from tests.conftest import inspect_step

PRISM_WITH_HOLE = "A rectangular prism with a hole in the middle."


def test_generate_sequence_returns_token_array(cad):
    vec = cad.generate_sequence(PRISM_WITH_HOLE)
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 2 and vec.shape[1] == 2


def test_greedy_decoding_is_deterministic(cad):
    np.testing.assert_array_equal(cad.generate_sequence(PRISM_WITH_HOLE), cad.generate_sequence(PRISM_WITH_HOLE))


def test_prism_with_hole_is_valid_solid(cad, tmp_path):
    path = cad.generate(PRISM_WITH_HOLE, output=tmp_path / "part.step")
    assert path == tmp_path / "part.step" and path.exists()

    info = inspect_step(path)
    assert info.valid
    assert info.solids == 1
    assert info.faces["cylinder"] >= 1, f"expected a cylindrical hole, got {dict(info.faces)}"
    bbox_volume = info.size[0] * info.size[1] * info.size[2]
    assert 0.3 < info.volume / bbox_volume < 0.99, "hole should remove part of the material"


def test_scale_multiplies_dimensions(cad, tmp_path):
    base = inspect_step(cad.generate(PRISM_WITH_HOLE, output=tmp_path / "base.step"))
    scaled = inspect_step(cad.generate(PRISM_WITH_HOLE, output=tmp_path / "scaled.step", scale=100))
    for a, b in zip(base.size, scaled.size):
        assert b == pytest.approx(a * 100, rel=1e-3)
    assert scaled.volume == pytest.approx(base.volume * 100**3, rel=1e-3)


def test_stp_extension_is_kept(cad, tmp_path):
    path = cad.generate(PRISM_WITH_HOLE, output=tmp_path / "part.stp")
    assert path.suffix == ".stp" and path.exists()
    assert not (tmp_path / "part.step").exists()
    assert inspect_step(path).valid


def test_stl_output(cad, tmp_path):
    import trimesh

    path = cad.generate(PRISM_WITH_HOLE, output=tmp_path / "part.stl", scale=10)
    mesh = trimesh.load(path)
    assert len(mesh.faces) > 0
    assert mesh.is_watertight


def test_output_directory_is_created(cad, tmp_path):
    path = cad.generate(PRISM_WITH_HOLE, output=tmp_path / "nested" / "dir" / "part.step")
    assert path.exists()


def test_unsupported_extension_raises(cad, tmp_path):
    with pytest.raises(ValueError, match="Unsupported output format"):
        cad.generate(PRISM_WITH_HOLE, output=tmp_path / "part.obj")


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TextToCAD(checkpoint=tmp_path / "missing.pth")


def test_invalid_sequence_raises_invalid_cad_error(cad, tmp_path, monkeypatch):
    garbage = np.zeros((10, 2), dtype=np.int64)
    monkeypatch.setattr(cad, "generate_sequence", lambda prompt: garbage)
    with pytest.raises(InvalidCADError):
        cad.generate("anything", output=tmp_path / "bad.step")


def test_model_is_reused_between_calls(cad, tmp_path):
    start = time.perf_counter()
    cad.generate(PRISM_WITH_HOLE, output=tmp_path / "a.step")
    assert time.perf_counter() - start < 30, "generate() should not reload the model"


DIVERSE_PROMPTS = [
    "A cylinder.",
    "A rectangular plate with four holes at the corners.",
    "A circular disc with a hole in the center.",
    "An L-shaped bracket.",
    "A cube.",
    "A hexagonal prism.",
    "A ring with an outer diameter twice the inner diameter.",
    "A rectangular block with a cylindrical boss on top.",
]


@pytest.mark.slow
def test_diverse_prompts_mostly_produce_valid_solids(cad, tmp_path):
    results = {}
    for i, prompt in enumerate(DIVERSE_PROMPTS):
        try:
            info = inspect_step(cad.generate(prompt, output=tmp_path / f"p{i}.step"))
            results[prompt] = "ok" if info.valid and info.solids >= 1 else f"invalid ({info.solids} solids)"
        except InvalidCADError as exc:
            results[prompt] = f"InvalidCADError: {exc.__cause__!r}"
    print("\n".join(f"  {status:<12} {prompt}" for prompt, status in results.items()))
    ok = sum(status == "ok" for status in results.values())
    assert ok >= len(DIVERSE_PROMPTS) * 0.75, results


@pytest.mark.slow
def test_cpu_fallback(tmp_path):
    cpu_cad = TextToCAD(device="cpu")
    info = inspect_step(cpu_cad.generate(PRISM_WITH_HOLE, output=tmp_path / "cpu.step"))
    assert info.valid and info.solids == 1
