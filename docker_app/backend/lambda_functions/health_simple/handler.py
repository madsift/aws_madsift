"""
Health check Lambda function
"""
import json
import os

def handler(event, context):
    # Check if it's SQS event (DLQ) or API Gateway (health check)
    if 'Records' in event:
        return handle_dlq_messages(event)
    else:
        return health_check(event, context)

def handle_dlq_messages(event):
    """Process failed KG jobs from DLQ"""
    for record in event['Records']:
        failed_message = json.loads(record['body'])
        job_id = failed_message.get('job_id', 'unknown')

        # Log the failure
        print(f"❌ KG Job {job_id} failed permanently: {failed_message}")

        # Optional: Update job status to FAILED_PERMANENTLY
        # Optional: Send alert/notification

    return {"statusCode": 200}

def health_check(event, context):
    """
    Simple health check endpoint
    """
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'status': 'healthy',
            'environment': os.environ.get('ENVIRONMENT', 'unknown'),
            'service': 'madsift-rumour-verification'
        })
    }
