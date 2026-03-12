FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY models/v1/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn

# Copy model artifact and API
COPY models/v1/ /app/models/v1/
COPY src/api.py /app/src/api.py

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
