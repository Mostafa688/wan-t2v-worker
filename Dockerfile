FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install diffusers==0.32.2 transformers==4.47.0 accelerate==1.2.1 && \
    pip install runpod boto3 imageio imageio-ffmpeg sentencepiece protobuf huggingface_hub

COPY handler.py .

CMD ["python", "-u", "handler.py"]
