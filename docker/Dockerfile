FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements_minimal.txt .
RUN pip install --no-cache-dir -r requirements_minimal.txt

COPY src ./src
COPY api ./api

RUN mkdir -p /app/data /app/saved

EXPOSE 8000

CMD ["python", "api/app.py"]
