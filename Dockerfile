FROM ultralytics/ultralytics:latest

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN git clone https://github.com/noahcao/OC_SORT.git /opt/OC_SORT
ENV PYTHONPATH=/opt/OC_SORT:${PYTHONPATH}

COPY scripts /workspace/scripts
COPY trackers /workspace/trackers
COPY configs /workspace/configs
COPY README.md /workspace/README.md
COPY Makefile /workspace/Makefile

ENV PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

ENTRYPOINT ["python", "/workspace/scripts/cli.py"]
CMD ["--help"]
