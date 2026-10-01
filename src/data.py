"""Загрузка, чистка и разбиение данных.

python -m src.data
"""
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "women-clothing-accessories.3-class.balanced.csv"
PROCESSED = ROOT / "data" / "processed"
RAW_URL = ("https://raw.githubusercontent.com/sismetanin/rureviews/master/"
           "women-clothing-accessories.3-class.balanced.csv")

# в исходнике опечатка "neautral", исправляем сразу
LABEL_FIX = {"neautral": "neutral"}
LABELS = ["negative", "neutral", "positive"]


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def has_letters(text: str) -> bool:
    return re.search(r"[а-яa-z]", text) is not None


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(RAW, sep="\t")
    df["label"] = df["sentiment"].replace(LABEL_FIX)
    df["text"] = df["review"].astype(str)
    df["norm"] = df["text"].map(normalize)
    n0 = len(df)

    # отзывы вида ".", "-", "..." - метку по ним не угадать, это шум
    df = df[df["norm"].map(has_letters)]
    print(f"убрал без букв: {n0 - len(df)}")

    # одинаковые тексты с разными метками ("товар не пришел" есть во всех 3 классах).
    # оставляем текст один раз с самой частой меткой; если ничья - выкидываем
    stats = df.groupby("norm")["label"].agg(
        n="size", n_labels="nunique",
        top=lambda s: s.value_counts().index[0],
        top_share=lambda s: s.value_counts(normalize=True).iloc[0],
    )
    conflicts = stats[stats.n_labels > 1]
    print(f"уникальных текстов: {len(stats)}, из них с противоречивыми метками: {len(conflicts)}")
    stats = stats[stats.top_share > 0.5]
    out = df.drop_duplicates("norm").set_index("norm")
    out = out.loc[stats.index]
    out["label"] = stats["top"]
    out = out.reset_index()[["text", "norm", "label"]]
    print(f"после дедупликации: {len(out)} ({n0 - len(out)} строк убрано)")
    return out


def split(df: pd.DataFrame, seed: int = 42):
    # дедупликация ДО разбиения - иначе одинаковые тексты попадают и в train, и в test
    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)
    valid, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=seed)
    return train, valid, test


def main():
    df = load_and_clean()
    train, valid, test = split(df)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("valid", valid), ("test", test)]:
        part.to_parquet(PROCESSED / f"{name}.parquet", index=False)
        print(name, len(part), part.label.value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()
