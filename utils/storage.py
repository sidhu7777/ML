import boto3
from botocore.exceptions import ClientError
from flask import current_app
import os

class S3Storage:
    """AWS S3 storage handler"""
    
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=current_app.config['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=current_app.config['AWS_SECRET_ACCESS_KEY'],
            region_name=current_app.config['S3_REGION']
        )
        self.bucket = current_app.config['S3_BUCKET_NAME']
    
    def upload_file(self, local_path, s3_key):
        """Upload a file to S3"""
        try:
            self.s3_client.upload_file(local_path, self.bucket, s3_key)
            current_app.logger.info(f"Uploaded to S3: {s3_key}")
            return s3_key
        except ClientError as e:
            current_app.logger.error(f"S3 upload error: {e}")
            raise
    
    def upload_directory(self, local_dir, s3_prefix):
        """Upload entire directory to S3"""
        uploaded_files = {}
        
        for root, dirs, files in os.walk(local_dir):
            for filename in files:
                local_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_path, local_dir)
                s3_key = f"{s3_prefix}/{relative_path}"
                
                self.upload_file(local_path, s3_key)
                uploaded_files[filename] = s3_key
        
        return uploaded_files
    
    def get_download_url(self, output_dir, filename, expiration=3600):
        """Generate presigned download URL"""
        s3_key = f"{output_dir}/{filename}"
        
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            current_app.logger.error(f"Error generating URL: {e}")
            raise
    
    def list_files(self, prefix):
        """List files in S3 prefix"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                return []
            
            return [obj['Key'].split('/')[-1] for obj in response['Contents']]
            
        except ClientError as e:
            current_app.logger.error(f"Error listing files: {e}")
            return []