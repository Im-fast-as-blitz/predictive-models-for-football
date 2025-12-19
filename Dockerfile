FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y && rm -rf /var/lib/apt/lists/*

COPY requirements_minimal.txt .
RUN pip install --no-cache-dir -r requirements_minimal.txt

COPY app.py .

RUN mkdir -p /app/content

EXPOSE 8000

CMD ["python", "app.py"]

