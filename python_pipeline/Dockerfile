FROM python:3.13-slim

WORKDIR /app

COPY python_pipeline/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY python_pipeline /app/python_pipeline

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "python_pipeline.api.main:app", "--host", "0.0.0.0", "--port", "8000"]