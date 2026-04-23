import boto3
import os

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

BUCKET = os.getenv("AWS_S3_BUCKET")

def upload_pdf(local_path: str, s3_key: str) -> str:
    s3.upload_file(
        local_path,
        BUCKET,
        s3_key,
        ExtraArgs={
            "ContentType": "application/pdf",
            
        }
    )

    signed_url = s3.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": BUCKET,
            "Key": s3_key
        },
        ExpiresIn=600  # 10 minutes
    )
    return signed_url
