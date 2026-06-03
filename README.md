## Чекпоинты проекта

### Чекпоинт 1, 2, 3

Мы сделали:
* EDA
* 2 основных алгоритма сбора данных для baseline модели
* модель (catboost) в качестве baseline

Показали все нашему куратору в тг (на гитхаб большой такой файл не получится загрузить)

### Чекпоинт 4: FastAPI приложение
FastAPI приложение для предсказания результатов футбольных матчей.

**Что сделано:**
* Endpoint `/forward` для предсказаний
* Endpoint `/teams` - список доступных команд
* Endpoint `/seasons` - список доступных сезонов
* Docker контейнеризация

### Инференс DL-модели

Приложение обслуживает обученную DL-модель (`ITransformer`, конфиг
`full_ts_approach`). Веса (`saved/best_model.pth`) и датасет
(`data/selected_leagues_one_line.csv`) слишком большие для GitHub, поэтому
выложены в публичную папку на Яндекс.Диске и скачиваются скриптом.

Размерности модели восстанавливаются из датасета так же, как в `train.py`
(тот же `cat_encoder` и `feature_cols`), а признаки одиночного матча строятся
через `PandasDataset.build_sample`.

## Запуск обучения

### Установка

0. (Optional) Создайте и активируйте новое окружение `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Установите все необходимые пакеты:

   ```bash
   pip install -r requirements.txt
   ```


### Как использовать

Для обучения модели в терминале введите команду:

```bash
python3 train.py --config CONFIG_NAME
```

Где `CONFIG_NAME` это имя файла конфига из папки `src/configs`.

Например для обчения base модели:
```bash
python3 train.py --config base
```


## Скачивание весов и данных

Перед запуском инференса нужно скачать веса модели и датасет с Яндекс.Диска.
Публичную ссылку на папку с артефактами укажите через флаг или переменную
окружения (или впишите в `DEFAULT_PUBLIC_LINK` в скрипте):

```bash
python3 scripts/download_artifacts.py --link https://disk.yandex.ru/d/XXXXXXXX
# или
YDISK_PUBLIC_LINK=https://disk.yandex.ru/d/XXXXXXXX python3 scripts/download_artifacts.py
```

Скрипт скачает:
* `best_model.pth` → `saved/best_model.pth`
* `selected_leagues_one_line.csv` → `data/selected_leagues_one_line.csv`

Скрипт использует только стандартную библиотеку, поэтому его можно запускать
до установки зависимостей.

> Папка на Яндекс.Диске должна содержать файлы с именами `best_model.pth`
> и `selected_leagues_one_line.csv` в корне публичной папки.

## Развертывание

### Docker Compose
```bash
# 1. скачать артефакты в ./saved и ./data
python3 scripts/download_artifacts.py --link <ваша_ссылка>
# 2. поднять сервис (./saved и ./data монтируются в контейнер)
cd docker
docker compose up --build
```

### Локально
```bash
pip install -r requirements_minimal.txt
python3 scripts/download_artifacts.py --link <ваша_ссылка>
python3 api/app.py
```

## Использование API

**Базовый URL:** `http://localhost:8000` или `http://<IP>:8000`

### GET /teams
```bash
curl http://localhost:8000/teams
```

### GET /seasons
```bash
curl http://localhost:8000/seasons
```

### POST /forward
```bash
curl -X POST http://localhost:8000/forward \
  -H "Content-Type: application/json" \
  -d '{"HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "season": "2001-2002"}'
```

**Ответ:**
```json
{
  "prediction": 0,
  "prediction_label": "home win",
  "probabilities": {
    "home_win": 0.47,
    "away_win": 0.18,
    "draw": 0.35
  }
}
```

**Параметры:**
- `HomeTeam` (обязательный) - домашняя команда
- `AwayTeam` (обязательный) - гостевая команда
- `season` (опциональный, default: "2024-2025") - сезон

**Коды ответов:**
- 200 - успех
- 400 - неверный формат
- 403 - модель не смогла обработать данные
