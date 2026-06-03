import runpod
import torch
import os
import uuid
import boto3

def upload_to_s3(file_path, bucket, key):
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
    )
    s3.upload_file(file_path, bucket, key, ExtraArgs={"ContentType": "video/mp4"})
    public_url = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")
    return f"{public_url}/{key}"

def handler(job):
    input_data = job["input"]
    prompt = input_data.get("prompt", "")
    negative_prompt = input_data.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted")
    num_frames = input_data.get("num_frames", 121)
    width = input_data.get("width", 832)
    height = input_data.get("height", 480)
    fps = input_data.get("fps", 24)

    try:
        from diffusers import LTXPipeline
        from diffusers.utils import export_to_video

        dtype = torch.bfloat16
        model_id = "Lightricks/LTX-Video"

        print("Loading model from cache...")
        pipe = LTXPipeline.from_pretrained(model_id, torch_dtype=dtype, local_files_only=True)
        pipe.to("cuda")
        print("Model loaded!")

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=3.0,
            num_inference_steps=50,
        )

        frames = output.frames[0]
        filename = f"{uuid.uuid4()}.mp4"
        local_path = f"/tmp/{filename}"

        export_to_video(frames, local_path, fps=fps)

        video_url = upload_to_s3(
            local_path,
            os.environ.get("S3_BUCKET", "erivion-videos"),
            f"wan-videos/{filename}"
        )

        return {"status": "success", "video_url": video_url}

    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}

runpod.serverless.start({"handler": handler})
