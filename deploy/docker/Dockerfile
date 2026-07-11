# syntax=docker/dockerfile:1.7
FROM openvino/ubuntu22_runtime:2024.4.0

USER root
WORKDIR /app

# System deps for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps first (cache-friendly)
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Source
COPY pyproject.toml README.md ./
COPY src ./src
COPY dashboard ./dashboard
COPY scripts ./scripts
RUN pip3 install --no-cache-dir -e .

# Non-root runtime user
RUN useradd -m -u 10001 gaze && chown -R gaze:gaze /app
USER gaze

EXPOSE 8501

# Analytics headless by default; override in docker-compose to run dashboard.
ENV GAZE_DEVICE=AUTO \
    GAZE_HEADLESS=true \
    PYTHONUNBUFFERED=1

CMD ["python3", "-m", "gaze_analytics", "--headless", "--device", "AUTO"]
