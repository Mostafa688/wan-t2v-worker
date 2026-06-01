import runpod
import torch
import os
import uuid
import boto3
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from transformers import UMT5EncoderModel

print("Loading WAN 2.1 T2V model...")

dtype = torch.bfloat16
model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

text_encoder = UMT5EncoderModel.from_pretrained(
    model_id, subfolder="text_encoder", torch_dtype=dtype
)
vae = AutoencoderKLWan.from_pretrained(
    model_id, subfolder="vae", torch_dtype=torch.float32
)
pipe = WanPipeline.from_pretrained(
    model_id, text_encoder=text_encoder, vae=vae, torch_dtype=dtype
)
pipe.to("cuda")
print("Model loaded!")

def upload_to_s3(file_path, bucket, key):
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
    )
    s3.upload_file(file_path, bucket, key, ExtraArgs={"ContentType": "video/mp4"})
    return f"{os.environ.get('S3_ENDPOINT_URL')}/{bucket}/{key}"

def handler(job):
    input = job["input"]
    prompt = input.get("prompt", "")
    negative_prompt = input.get("negative_prompt", "")
    num_frames = input.get("num_frames", 33)
    guidance_scale = input.get("guidance_scale", 5.0)
    num_inference_steps = input.get("num_inference_steps", 50)
    width = input.get("width", 832)
    height = input.get("height", 480)

    try:
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )

        frames = output.frames[0]
        filename = f"{uuid.uuid4()}.mp4"
        local_path = f"/tmp/{filename}"
        export_to_video(frames, local_path, fps=16)

        video_url = upload_to_s3(
            local_path,
            os.environ.get("S3_BUCKET"),
            f"wan-videos/{filename}"
        )

        return {"status": "success", "video_url": video_url}

    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
