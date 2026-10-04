"""End-to-end pipeline: prompt -> (translation) -> Text2CAD -> STEP/STL.

Run from the repository root with the Text2CAD venv:
    Text2CAD\\.venv\\Scripts\\python -m pipeline.run --lang tr --prompt "Ortasinda delik olan dikdortgen bir prizma." --output output\\part.step --scale 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.text_to_cad import InvalidCADError, TextToCAD
from pipeline.translator import Translator


def run(
    prompt: str,
    output: str | Path,
    lang: str = "en",
    scale: float = 1.0,
    translator: Translator | None = None,
    cad_model: TextToCAD | None = None,
) -> tuple[str, Path]:
    """Return the English prompt that was sent to Text2CAD and the written file path."""
    if lang != "en":
        translator = translator or Translator(source_lang=lang)
        prompt = translator.generate(prompt)
    cad_model = cad_model or TextToCAD()
    return prompt, cad_model.generate(prompt, output=output, scale=scale)


def main() -> int:
    # Text2CAD logs contain emoji that the default Windows console code page cannot encode.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--lang", default="en", help="prompt language: en (no translation) or tr")
    parser.add_argument("--output", default="output/part.step", help=".step, .stp or .stl path")
    parser.add_argument("--scale", type=float, default=1.0, help="uniform scale (model output is normalized)")
    args = parser.parse_args()

    try:
        english_prompt, path = run(args.prompt, args.output, lang=args.lang, scale=args.scale)
    except InvalidCADError as exc:
        print(f"ERROR: {exc}\n  cause: {exc.__cause__!r}", file=sys.stderr)
        return 1
    if args.lang != "en":
        print(f"Translated prompt: {english_prompt}")
    print(f"Saved {path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
