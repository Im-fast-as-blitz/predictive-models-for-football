#!/usr/bin/env python3
"""Скачивание весов DL-модели и данных с публичной ссылки Яндекс.Диска.

Веса (`saved/best_model.pth`) и датасет (`data/<name>.csv`) слишком большие для
GitHub. Мы выкладываем их в одну публичную папку на Яндекс.Диске и качаем оттуда
через публичный API (auth не требуется).

Подготовка (делается один раз автором):
  1. Создать на Яндекс.Диске папку, положить в неё:
       - best_model.pth
       - selected_leagues_one_line.csv
  2. Включить «Поделиться» у папки и скопировать публичную ссылку.
  3. Вставить ссылку ниже в DEFAULT_PUBLIC_LINK или передать её через
     переменную окружения YDISK_PUBLIC_LINK / флаг --link.

Использование:
  python scripts/download_artifacts.py
  python scripts/download_artifacts.py --link https://disk.yandex.ru/d/XXXXXXXX
  YDISK_PUBLIC_LINK=https://disk.yandex.ru/d/XXXX python scripts/download_artifacts.py

Использует только стандартную библиотеку — можно запускать до установки зависимостей.
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

# 👇 вставьте сюда публичную ссылку на папку Яндекс.Диска с артефактами
DEFAULT_PUBLIC_LINK = ""

API_URL = "https://cloud-api.yandex.net/v1/disk/public/resources/download"

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (путь внутри публичной папки, локальный путь назначения относительно корня проекта)
ARTIFACTS = [
    ("best_model.pth", "saved/best_model.pth"),
    ("selected_leagues_one_line.csv", "data/selected_leagues_one_line.csv"),
]


def get_direct_url(public_link: str, path_in_folder: str | None) -> str:
    """Запрашивает у API Яндекс.Диска прямую ссылку на скачивание."""
    params = {"public_key": public_link}
    if path_in_folder:
        params["path"] = "/" + path_in_folder.lstrip("/")
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if "href" not in payload:
        raise RuntimeError(f"API не вернул ссылку для '{path_in_folder}': {payload}")
    return payload["href"]


def download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        chunk = 1 << 20  # 1 MiB
        with open(tmp, "wb") as f:
            while True:
                data = resp.read(chunk)
                if not data:
                    break
                f.write(data)
                read += len(data)
                if total:
                    pct = read * 100 // total
                    print(f"\r  {dest}: {pct}% ({read >> 20}/{total >> 20} MiB)", end="")
                else:
                    print(f"\r  {dest}: {read >> 20} MiB", end="")
    os.replace(tmp, dest)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--link",
        default=os.getenv("YDISK_PUBLIC_LINK", DEFAULT_PUBLIC_LINK),
        help="Публичная ссылка на папку Яндекс.Диска с артефактами",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перекачать, даже если файл уже существует",
    )
    args = parser.parse_args()

    if not args.link:
        print(
            "Ошибка: не указана публичная ссылка Яндекс.Диска.\n"
            "Передайте её через --link, переменную YDISK_PUBLIC_LINK "
            "или впишите в DEFAULT_PUBLIC_LINK в этом скрипте.",
            file=sys.stderr,
        )
        return 1

    for path_in_folder, rel_dest in ARTIFACTS:
        dest = os.path.join(ROOT_DIR, rel_dest)
        if os.path.exists(dest) and not args.force:
            print(f"Пропускаю (уже есть): {rel_dest}")
            continue
        print(f"Скачиваю {path_in_folder} -> {rel_dest}")
        direct_url = get_direct_url(args.link, path_in_folder)
        download(direct_url, dest)

    print("Готово.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
