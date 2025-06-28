import boto3

runtime = boto3.client("sagemaker-runtime", region_name="eu-north-1")

with open("sample.csv", "rb") as f:
    payload = f.read()

response = runtime.invoke_endpoint(
    EndpointName="my-endpoint",
    ContentType="text/csv",
    Body=payload
)

result = response["Body"].read()
print(result.decode())
