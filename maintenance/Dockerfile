FROM python:3.7-slim


WORKDIR /app

# Install system dependencies for lightgbm
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --progress-bar=off -r requirements.txt


COPY . .

CMD ["python", "pred.py", "--mode", "server"]

