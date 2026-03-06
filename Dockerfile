FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir ".[web]"

EXPOSE 8000

CMD ["spectra", "serve", "--host", "0.0.0.0", "--no-open"]
