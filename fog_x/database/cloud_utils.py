import os
import smart_open
import logging
import boto3
from google.cloud import storage
logger = logging.getLogger(__name__)

def check_file_exists_s3(bucket_name, file_key):
    """Check if a file exists in Amazon S3."""
    s3_client = boto3.client('s3')
    try:
        s3_client.head_object(Bucket=bucket_name, Key=file_key)
        return True
    except Exception as e:
        return False

def check_file_exists_gcs(bucket_name, file_name):
    """Check if a file exists in gcs."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    return blob.exists()

def check_path_exists_smart_open(path: str):
    """Check if a file exists with smart-open."""
    path = path.strip("/")
    path = f"{path}/{filename}"
    path = os.path.expanduser(path)
    try:
        file = smart_open.open(path)
    except Exception as e:
        return False
    return True
    
    
if __name__ == "__main__":
    bucket = "fog-rtx-test-east-1"
    filename = "demo_ds.parquet"
    print(f"checking file: {bucket}/{filename}, Result: {check_file_exists_s3(bucket, filename)}")