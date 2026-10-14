FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY requirements.txt ./requirements.txt
# Retries/timeout tolerate slow registries; the installed set is unchanged.
RUN python -m pip install --upgrade pip && pip install --retries 10 --timeout 120 -r requirements.txt

COPY . .
RUN mkdir -p data/raw data/processed reports/generated artifacts
# Editable install exposes the relocated `audit` and `features` packages.
RUN pip install --no-build-isolation -e .

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
