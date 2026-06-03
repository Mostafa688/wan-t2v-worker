FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install --upgrade --force-reinstall diffusers>=0.32.0 transformers>=4.45.0 accelerate>=0.33.0 && \
    pip install runpod boto3 imageio imageio-ffmpeg sentencepiece protobuf huggingface_hub

COPY handler.py .

CMD ["python", "-u", "handler.py"]
