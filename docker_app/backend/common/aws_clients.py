import os
import json
import boto3
from botocore.exceptions import ClientError
region = os.getenv("AWS_REGION", "us-east-1")

def s3_client():
    return boto3.client('s3', region_name=region)

def bedrock_client():
    return boto3.client('bedrock-runtime', region_name=region)

def get_secret(secret_name, region_name, key=None):

    
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e
    secret_dict = json.loads(get_secret_value_response["SecretString"])
    if key:
        return secret_dict[key]
    else:
        return secret_dict
