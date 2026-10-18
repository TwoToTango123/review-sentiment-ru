from src.data import has_letters, normalize
from src.text import lemmatize, mark_negation


def test_normalize():
    assert normalize("  Товар   НЕ пришёл ") == "товар не пришел"


def test_has_letters():
    assert not has_letters("...")
    assert has_letters("ок")


def test_negation():
    assert mark_negation("мне не понравилось") == "мне не_понравилось"


def test_lemmatize_keeps_emoticons():
    out = lemmatize("платья отличные )))")
    assert "платье" in out and "отличный" in out and ")))" in out
