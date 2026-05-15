FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY anvil_pcam/ anvil_pcam/

RUN pip install --no-cache-dir -e .

EXPOSE 8420

CMD ["anvil-pcam", "serve", "--host", "0.0.0.0", "--port", "8420"]
