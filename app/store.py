from app.constants import *

import boto3
import os

class S3Store:
    def __init__(self, local_dir=LOCAL_DATA_PATH):
        self.bucket_name = 'sae-classification'
        self.local_dir = local_dir
        self.s3 = boto3.client('s3', 
                               aws_access_key_id=CRED["AWS_ACCESS_KEY"], 
                               aws_secret_access_key=CRED["AWS_SECRET_KEY"])
        self.create_bucket_if_not_exists()

    def download(self, file_path):
        local_file = os.path.join(self.local_dir, file_path)
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        self.s3.download_file(self.bucket_name, file_path, local_file)

    def upload(self, file_path):
        remote_file = file_path.replace(self.local_dir + '/', "").lstrip('/')
        self.s3.upload_file(file_path, self.bucket_name, remote_file)

    def overwrite_local(self, file_path):
        remote_file = file_path.replace(self.local_dir + '/', "").lstrip('/')
        self.s3.download_file(self.bucket_name, remote_file, file_path)

    def overwrite_remote(self, file_path):
        local_file = os.path.join(self.local_dir, file_path)
        self.s3.upload_file(local_file, self.bucket_name, file_path)

    def delete_all_remote(self):
        paginator = self.s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket_name):
            for obj in page.get('Contents', []):
                self.s3.delete_object(Bucket=self.bucket_name, Key=obj['Key'])

    def show_remote_files(self):
        response = self.s3.list_objects_v2(Bucket=self.bucket_name)
        n_remote_objects = response.get('KeyCount', 0)
        print(f"Total remote objects: {n_remote_objects}")
        for obj in response.get('Contents', []):
            print(obj['Key'])

    def create_bucket_if_not_exists(self):
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
        except self.s3.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                print(f"Bucket '{self.bucket_name}' does not exist. Creating bucket.")
                self.s3.create_bucket(
                    Bucket=self.bucket_name,
                )
            else:
                print(f"Error checking if bucket exists: {e}")

    # make it so that the remote matches the local by deleting remote files that don't exist locally
    # and uploading local files that don't exist remotely
    def sync_remote(self):
        for root, dirs, files in os.walk(self.local_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                remote_file_path = os.path.relpath(local_file_path, self.local_dir)

                if not remote_file_path.startswith('.'):
                    try:
                        self.s3.head_object(Bucket=self.bucket_name, Key=remote_file_path)
                        print(file, '<----- already exists remotely')
                    except self.s3.exceptions.ClientError as e:
                        if e.response['Error']['Code'] == '404':
                            self.upload(local_file_path)
                            print(file, '<----- uploaded')

        paginator = self.s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket_name):
            for obj in page.get('Contents', []):
                remote_file_path = obj['Key']

                if not remote_file_path.startswith('.'):
                    local_file_path = os.path.join(self.local_dir, remote_file_path)
                    if not os.path.exists(local_file_path):
                        self.s3.delete_object(Bucket=self.bucket_name, Key=remote_file_path)
                        print(remote_file_path, '<----- deleted from remote')

    def sync(self):
        for root, dirs, files in os.walk(self.local_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                remote_file_path = os.path.relpath(local_file_path, self.local_dir)

                if not remote_file_path.startswith('.'):
                    try:
                        self.s3.head_object(Bucket=self.bucket_name, Key=remote_file_path)
                        print(file, '<----- already exists remotely')
                    except self.s3.exceptions.ClientError as e:
                        if e.response['Error']['Code'] == '404':
                            self.upload(local_file_path)
                            print(file, '<----- uploaded')

        paginator = self.s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket_name):
            for obj in page.get('Contents', []):
                remote_file_path = obj['Key']

                if not remote_file_path.startswith('.'):
                    local_file_path = os.path.join(self.local_dir, remote_file_path)
                    if not os.path.exists(local_file_path):
                        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                        self.download(remote_file_path)
                        print(remote_file_path, '<----- downloaded')
                    else:
                        print(remote_file_path, '<----- already exists locally')
