FROM python:3.7-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./