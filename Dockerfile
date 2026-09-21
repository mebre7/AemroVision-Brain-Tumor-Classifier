FROM python:3.12-slim-bookworm

RUN apt update && apt install awscli -y

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.in

CMD ["python3", "app.py"]