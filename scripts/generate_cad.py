"""Generate a STEP/STL file from an English text prompt with Text2CAD.

Run from the repository root with the Text2CAD venv:
    Text2CAD\\.venv\\Scripts\\python scripts\\generate_cad.py --prompt "A rectangular prism with a hole in the middle." --output output\\part.step
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Text2CAD logs contain emoji that the default Windows console code page cannot encode.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pipeline.text_to_cad import InvalidCADError, TextToCAD  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True, help="English description of the part")
    parser.add_argument("--output", default="output/part.step", help=".step, .stp or .stl path")
    parser.add_argument("--scale", type=float, default=1.0, help="uniform scale (model output is normalized)")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    args = parser.parse_args()

    cad = TextToCAD(device=args.device)
    try:
        path = cad.generate(args.prompt, output=args.output, scale=args.scale)
    except InvalidCADError as exc:
        print(f"ERROR: {exc}\n  cause: {exc.__cause__!r}", file=sys.stderr)
        return 1
    print(f"Saved {path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
