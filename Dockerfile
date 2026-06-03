FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121

RUN pip install git+https://github.com/huggingface/diffusers.git@v0.33.0

RUN pip install "transformers>=4.45.0" "accelerate>=0.33.0"

RUN pip install runpod boto3 "imageio[ffmpeg]" sentencepiece protobuf huggingface_hub

COPY handler.py .

CMD ["python", "-u", "handler.py"]
