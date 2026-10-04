import pytest

from pipeline.run import run
from pipeline.translator import TRANSLATION_MODELS, Translator
from tests.conftest import inspect_step

pytestmark = pytest.mark.skipif(
    not TRANSLATION_MODELS["tr"].exists(), reason="TR->EN model weights are not checked in"
)


@pytest.fixture(scope="module")
def translator():
    return Translator("tr")


def test_translates_basic_cad_prompt(translator):
    english = translator.generate("Ortasında delik olan dikdörtgen bir prizma oluştur.").lower()
    assert "prism" in english and "hole" in english


def test_multi_sentence_prompt_is_split(translator):
    english = translator.generate("Bir küp oluştur. Ortasına delik aç.")
    assert english.count(".") >= 2


def test_empty_prompt(translator):
    assert translator.generate("   ") == ""


def test_unknown_language_raises():
    with pytest.raises(ValueError, match="No translation model"):
        Translator("xx")


def test_english_prompt_skips_translation(cad, tmp_path):
    prompt, path = run("A cube.", tmp_path / "en.step", lang="en", cad_model=cad)
    assert prompt == "A cube."
    assert path.exists()


def test_turkish_prompt_end_to_end(cad, translator, tmp_path):
    prompt, path = run(
        "Ortasında delik olan dikdörtgen bir prizma oluştur.",
        tmp_path / "tr.step",
        lang="tr",
        scale=100,
        translator=translator,
        cad_model=cad,
    )
    assert "hole" in prompt.lower()
    info = inspect_step(path)
    assert info.valid and info.solids == 1 and info.faces["cylinder"] >= 1
