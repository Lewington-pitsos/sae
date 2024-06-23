import runpod
import uuid

from app.constants import *

def deploy_pod(gpu_type_id="NVIDIA A100 80GB PCIe"):
    runpod.api_key = CRED['RUNPOD_API_KEY']

    env = {**CRED}

    pod = runpod.create_pod(
        cloud_type="SECURE", # or else someone might snoop your session and steal your AWS/CDS credentials
        name=str(uuid.uuid4())[:32], 
        image_name="runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04", 
        gpu_type_id=gpu_type_id,
        container_disk_in_gb=400,
        env=env,
        ports="22/tcp"

    )

    print(f"Pod created: {pod}")

def show_pods():
    runpod.api_key = CRED['RUNPOD_API_KEY']

    pods = runpod.get_pods()

    print(f"Total pods: {len(pods)}")
    for pod in pods:
        print(pod)
        port_data = pod['runtime']['ports'][-1]
        print(port_data)
        print(f"""\nHost runpod.io
    HostName {port_data['ip']}
    User root
    Port {port_data['publicPort']}
    IdentityFile ~/.ssh/id_louka
    IdentitiesOnly yes
""")

def destroy_all_pods():
    runpod.api_key = CRED['RUNPOD_API_KEY']

    pods = runpod.get_pods()

    print('currently running pods:', pods)

    for pod in pods:
        print(f"Deleting pod: {pod['id']}")
        runpod.terminate_pod(pod['id'])