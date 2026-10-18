"""Препроцессинг текста: токенизация и лемматизация."""
import re
from functools import lru_cache

import pymorphy3

_morph = pymorphy3.MorphAnalyzer()
TOKEN_RE = re.compile(r"[а-яa-z]+(?:-[а-яa-z]+)?|[!?)(]+")

# стоп-слова НЕ выкидываю: "не", "очень", "но" для тональности критичны
# ("не понравилось" vs "понравилось"). Проверял - со стоп-словами из nltk F1 падает.


@lru_cache(maxsize=200_000)
def lemma(word: str) -> str:
    return _morph.parse(word)[0].normal_form


def tokenize(text: str) -> list:
    return TOKEN_RE.findall(text)


def lemmatize(text: str) -> str:
    out = []
    for tok in tokenize(text):
        out.append(lemma(tok) if tok[0].isalpha() else tok)
    return " ".join(out)


def mark_negation(text: str) -> str:
    """'не понравилось' -> 'не_понравилось'. Для bag-of-words это единственный
    способ различить отрицание, биграммы частично решают то же самое."""
    return re.sub(r"\b(не|нет|ни)\s+([а-я]+)", r"\1_\2", text)
