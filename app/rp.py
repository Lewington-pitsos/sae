import runpod
import uuid

from app.constants import *


def create_pod(gpu_type_id="NVIDIA A100 80GB PCIe"):
    runpod.api_key = CREDS['RUNPOD_API_KEY']

    env = {**CREDS}

    pod = runpod.create_pod(
        cloud_type="SECURE", # or else someone might snoop your session and steal your AWS/CDS credentials
        name=str(uuid.uuid4())[:32], 
        image_name="unpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04", 
        gpu_type_id=gpu_type_id,
        container_disk_in_gb=400,
        env=env
    )

    print(f"Pod created: {pod}")

def show_pods():
    runpod.api_key = CREDS['RUNPOD_API_KEY']

    pods = runpod.get_pods()

    print(f"Total pods: {len(pods)}")
    for pod in pods:
        print(pod)

def destroy_all_pods():
    runpod.api_key = CREDS['RUNPOD_API_KEY']

    pods = runpod.get_pods()

    print('currently running pods:', pods)

    for pod in pods:
        print(f"Deleting pod: {pod['id']}")
        runpod.delete_pod(pod['id'])