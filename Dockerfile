FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY configs /app/configs
COPY scripts /app/scripts

RUN pip install --upgrade pip && pip install -e .

ENV PYTHONPATH=/app/src

CMD ["python", "-m", "turbine_project.cli", "run-all", "--config", "configs/default.yaml", "--max-input-chunks", "1", "--max-assets", "2"]
