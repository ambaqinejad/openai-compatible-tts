ARG BASE_IMAGE=omnivoice-base:latest

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ============================================================
# Application
# ============================================================

WORKDIR /app

# ============================================================
# Python dependencies
# ============================================================

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install \
        --break-system-packages \
        -r /tmp/requirements.txt && \
    rm -f /tmp/requirements.txt

# ============================================================
# Application source
# ============================================================

COPY app ./app

# ============================================================
# Runtime directories
# ============================================================

RUN mkdir -p \
        /app/voices \
        /app/output

# ============================================================
# Runtime
# ============================================================

EXPOSE 8000

ENTRYPOINT ["tini", "--"]

CMD ["python3","-m","uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]