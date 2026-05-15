FROM ultralytics/ultralytics:latest

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY app /workspace/app
COPY config /workspace/config
COPY scripts /workspace/scripts

ENV PYTHONPATH=/workspace/app \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

ENTRYPOINT ["python", "-m", "scripts.cli"]
CMD ["--help"]
