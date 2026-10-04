"""Thin wrapper that hides the Text2CAD research code behind a single class.

    cad = TextToCAD()
    cad.generate("A rectangular prism with a hole in the middle.", output="output/part.step")
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT2CAD_DIR = REPO_ROOT / "Text2CAD"
CAD_VLM_DIR = TEXT2CAD_DIR / "Cad_VLM"
DEFAULT_CONFIG = CAD_VLM_DIR / "config" / "inference_local.yaml"
DEFAULT_CHECKPOINT = TEXT2CAD_DIR / "checkpoints" / "Text2CAD_1.0.pth"

# OCCUtils imports itself as a top-level package, so CadSeqProc must be on sys.path too.
for _path in (TEXT2CAD_DIR, TEXT2CAD_DIR / "CadSeqProc"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform  # noqa: E402
from OCC.Core.gp import gp_Pnt, gp_Trsf  # noqa: E402

from CadSeqProc.cad_sequence import CADSequence  # noqa: E402
from CadSeqProc.utility.macro import MAX_CAD_SEQUENCE_LENGTH, N_BIT  # noqa: E402
from Cad_VLM.models.text2cad import Text2CAD  # noqa: E402


class InvalidCADError(RuntimeError):
    """The model produced a CAD sequence that OpenCASCADE could not turn into a solid."""


class TextToCAD:
    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        config: str | Path = DEFAULT_CONFIG,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = self._load_config(Path(config))

        cad_config = dict(self.config["cad_decoder"])
        cad_config["cad_seq_len"] = MAX_CAD_SEQUENCE_LENGTH
        self.model = Text2CAD(text_config=self.config["text_encoder"], cad_config=cad_config).to(self.device)
        # Some Text2CAD layers store their device as a plain attribute picked at
        # construction time (cuda if available), which .to() does not update.
        for module in self.model.modules():
            if type(module).__module__.startswith("Cad_VLM.") and isinstance(
                getattr(module, "device", None), torch.device
            ):
                module.device = torch.device(self.device)
        self._load_checkpoint(Path(checkpoint))
        self.model.eval()

        self.topk = 1 if self.config["test"]["sampling_type"] == "max" else 5
        self.nucleus_prob = self.config["test"].get("nucleus_prob", 0)

    @staticmethod
    def _load_config(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # Paths in the YAML are relative to Cad_VLM, where the upstream scripts are run from.
        embedder = config["text_encoder"]["text_embedder"]
        embedder["cache_dir"] = str((CAD_VLM_DIR / embedder["cache_dir"]).resolve())
        return config

    def _load_checkpoint(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Text2CAD checkpoint not found: {path}")
        # The official checkpoint stores more than tensors, so weights_only must be off.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = {
            key.removeprefix("module."): value for key, value in checkpoint["model_state_dict"].items()
        }
        result = self.model.load_state_dict(state_dict, strict=False)
        # The checkpoint intentionally omits the frozen BERT weights (loaded from Hugging Face).
        missing = [k for k in result.missing_keys if not k.startswith("base_text_embedder.")]
        if missing or result.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint does not match the model. Missing: {missing[:5]}, "
                f"unexpected: {result.unexpected_keys[:5]}"
            )

    @torch.no_grad()
    def generate_sequence(self, prompt: str) -> np.ndarray:
        """Return the raw CAD vector (token sequence) predicted for a prompt."""
        output = self.model.test_decode(
            texts=[prompt],
            maxlen=MAX_CAD_SEQUENCE_LENGTH,
            nucleus_prob=self.nucleus_prob,
            topk_index=self.topk,
            device=self.device,
        )
        return output["cad_vec"][0].cpu().numpy()

    def generate(self, prompt: str, output: str | Path, scale: float = 1.0) -> Path:
        """Generate a CAD model for `prompt` and write it to `output` (.step or .stl).

        Text2CAD works in a normalized space (parts fit in roughly a unit cube);
        `scale` multiplies the result uniformly, e.g. scale=100 for ~100 mm parts.
        """
        output = Path(output)
        file_type = {".step": "step", ".stp": "step", ".stl": "stl"}.get(output.suffix.lower())
        if file_type is None:
            raise ValueError(f"Unsupported output format '{output.suffix}', use .step, .stp or .stl")

        cad_vec = self.generate_sequence(prompt)
        try:
            sequence = CADSequence.from_vec(cad_vec, bit=N_BIT, post_processing=True)
            sequence.create_cad_model()
            if scale != 1.0:
                trsf = gp_Trsf()
                trsf.SetScale(gp_Pnt(0, 0, 0), scale)
                sequence.cad_model = BRepBuilderAPI_Transform(sequence.cad_model, trsf, True).Shape()
            output.parent.mkdir(parents=True, exist_ok=True)
            sequence.save_stp(filename=output.stem, output_folder=str(output.parent), type=file_type)
        except Exception as exc:
            raise InvalidCADError(f"Could not build a valid CAD model for prompt: {prompt!r}") from exc

        # save_stp always appends .step/.stl, so normalize .stp to the actual file name.
        written = output.with_suffix(".stl" if file_type == "stl" else ".step")
        if written != output:
            written.replace(output)
        return output
