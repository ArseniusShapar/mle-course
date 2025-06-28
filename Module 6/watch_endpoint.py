import platform
import time

import boto3

endpoint_name = "my-endpoint"
region = "eu-north-1"
check_interval = 30

client = boto3.client("sagemaker", region_name=region)


def beep():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 500)
    else:
        print("\a")


while True:
    response = client.describe_endpoint(EndpointName=endpoint_name)
    status = response["EndpointStatus"]

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Current status: {status}")

    if status == "InService":
        print("✅ Endpoint is ready!")
        beep()
        break

    if status in ("Failed", "OutOfService"):
        print("❌ Endpoint is failed")
        beep()
        break

    time.sleep(check_interval)
