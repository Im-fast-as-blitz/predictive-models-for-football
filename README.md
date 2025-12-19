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
* Docker контейнеризация

## Развертывание

### Docker Compose
```bash
docker compose up --build
```

### Локально
```bash
pip install -r requirements_minimal.txt
python3 app.py
```

## Использование API

**Базовый URL:** `http://localhost:8000` или `http://<IP>:8000`

### GET /teams
```bash
curl http://localhost:8000/teams
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
