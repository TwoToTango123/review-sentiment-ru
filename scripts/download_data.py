"""Скачивает RuReviews (90 тыс. отзывов, ~21 МБ) с GitHub авторов датасета."""
import sys
import urllib.request
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data import RAW, RAW_URL  # noqa: E402

RAW.parent.mkdir(parents=True, exist_ok=True)
if RAW.exists():
    print("уже скачано")
else:
    urllib.request.urlretrieve(RAW_URL, RAW)
    print("готово:", RAW)
