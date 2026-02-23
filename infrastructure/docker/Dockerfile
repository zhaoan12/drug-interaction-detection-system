FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY configs /app/configs
COPY data /app/data

RUN pip install --no-cache-dir .

EXPOSE 8080

CMD ["python", "-m", "drug_interaction_detection.cli.main", "--config", "configs/default.toml", "serve", "--port", "8080"]

