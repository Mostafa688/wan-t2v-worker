FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

RUN pip install "diffusers>=0.33.0" "transformers>=4.45.0,<4.50.0" "accelerate>=0.33.0"

RUN pip install runpod boto3 sentencepiece protobuf huggingface_hub ftfy imageio imageio-ffmpeg

# تحميل الموديل وقت البناء
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Lightricks/LTX-Video')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
