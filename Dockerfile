FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

RUN pip install "diffusers==0.31.0" "transformers==4.46.0" "accelerate==1.1.0"

RUN pip install runpod boto3 "imageio[ffmpeg]" sentencepiece protobuf huggingface_hub

COPY handler.py .

CMD ["python", "-u", "handler.py"]
