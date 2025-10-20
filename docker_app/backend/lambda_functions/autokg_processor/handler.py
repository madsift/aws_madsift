# Auto KG Processor Handler with AgentCore (Natural Language Interface)
from botocore.exceptions import ClientError
import json
import logging
import os
from pythonjsonlogger import jsonlogger
import boto3
from datetime import datetime, timedelta, time, timezone 
import io
import requests
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent, tool
from common.cognito_auth import lambda_handler_with_auth
import tempfile
import uuid


# Setup structured logging
logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Initialize AgentCore app
app = BedrockAgentCoreApp()

# AWS Clients
s3_client = boto3.client('s3')
sqs_client = boto3.client('sqs')
scheduler_client = boto3.client('scheduler')

# Environment variables
bucket = os.getenv("KG_BUCKET")
jobs_table_name = os.getenv("JOBS_TABLE_NAME")
kg_queue_url = os.getenv("KG_PROCESSING_QUEUE_URL")

# DynamoDB setup
DYNAMODB_ENDPOINT = os.getenv('DYNAMODB_ENDPOINT')
if DYNAMODB_ENDPOINT:
    dynamodb = boto3.resource('dynamodb', endpoint_url=DYNAMODB_ENDPOINT)
else:
    dynamodb = boto3.resource('dynamodb')
jobs_table = dynamodb.Table(jobs_table_name) if jobs_table_name else None

BATCH_SIZE_CLAIMS = 32


def get_event_source(event):
    """Helper to identify event source for logging"""
    if 'Records' in event and event['Records'][0].get('eventSource') == 'aws:sqs':
        return f"sqs:{event['Records'][0]['eventSource']}"
    elif 'source' in event and event['source'] == 'aws.events':
        return f"eventbridge:{event['source']}"
    elif 'httpMethod' in event:
        return f"api_gateway:{event['httpMethod']} {event.get('path', '')}"
    # FIX: Add a check for the custom payload from EventBridge Scheduler
    elif 'job_id' in event and 'payload' in event and 'detail-type' not in event:
        return "eventbridge:scheduler_custom"
    else:
        return "unknown"
        
@app.entrypoint
def lambda_handler(event, context):
    """Main entry point - routes based on event source"""
    
    # Check for natural language prompt
    if isinstance(event, dict) and 'prompt' in event and 'httpMethod' not in event:
        return handle_natural_language(event, context)
    
    # FIX: Get the event source once and use it for routing
    event_source_val = get_event_source(event)
    logger.info({
        "event_type": "request",
        "lambda_name": context.function_name,
        "aws_request_id": context.aws_request_id,
        "environment": os.getenv("ENVIRONMENT"),
        "event_source": event_source_val
    })
    
    try:
        if 'Records' in event and event['Records'][0].get('eventSource') == 'aws:sqs':
            return process_sqs_message(event, context)
        # FIX: Check if the source contains 'eventbridge'
        elif 'eventbridge' in event_source_val:
            return handle_eventbridge_tick(event, context)
        elif 'httpMethod' in event:
            return handle_api_gateway(event, context)
        else:
            return {"error": "Unknown event source", "statusCode": 400}
            
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {"error": "Internal server error", "statusCode": 500}

# ============================================================================
# NATURAL LANGUAGE HANDLING (AgentCore)
# ============================================================================

def handle_natural_language(event, context):
    """Handle natural language prompts via AgentCore"""
    prompt = event.get('prompt')
    username = event.get('username', 'system')
    
    logger.info(f"Natural language request: {prompt}")
    
    # Create agent with tools
    agent = Agent(
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        system_prompt="""You are a knowledge graph builder assistant.

You can help users:
1. Start monitoring jobs - "Monitor r/Nepal every 6 hours for 2 days"
2. Stop monitoring jobs - "Stop job kg-123"
3. Check job status - "What's the status of job kg-123?"

When user asks to monitor:
- Extract subreddit, interval, and duration
- Convert to proper format (interval in seconds, calculate end_time)
- Call start_monitoring_job tool with proper payload

Be conversational and helpful. Explain what you're doing.""",
        tools=[start_monitoring_job, stop_monitoring_job, get_job_status]
    )
    
    # Execute agent
    result = agent(prompt)
    
    # Extract response text
    response_text = str(result)
    if hasattr(result, 'message') and isinstance(result.message, dict):
        content = result.message.get('content', [])
        if content and isinstance(content, list) and len(content) > 0:
            response_text = content[0].get('text', str(result))
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'response': response_text,
            'timestamp': datetime.utcnow().isoformat()
        })
    }

# ============================================================================
# API GATEWAY HANDLING (Queue jobs and check status)
# ============================================================================

@lambda_handler_with_auth
def handle_api_gateway(event, context, user_info):
    """Handle API Gateway requests with Cognito auth"""
    username = user_info["username"]
    path = event.get('path', '')
    method = event.get('httpMethod', '')
    
    try:
        # GET /kg-processor/status/{job_id}
        if method == 'GET' and '/status/' in path:
            job_id = event.get('pathParameters', {}).get('job_id')
            return get_job_status(job_id)
        
        # POST /kg-processor
        elif method == 'POST':
            body = json.loads(event.get('body', '{}'))
            action = body.get('action')
            payload = body.get('payload', {})
            
            if action in ('start_monitoring', 'start_job'):
                return start_monitoring_job(payload, username, context)
            elif action in ('stop_monitoring', 'stop_job'):
                return stop_monitoring_job(payload)
            else:
                return queue_one_time_job(body, username)
        
        else:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({"success": False, "error": "Invalid request"})
            }
            
    except Exception as e:
        logger.error(f"API Gateway handler error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({"success": False, "error": str(e)})
        }

def get_job_status(job_id):
    """Get job status from DynamoDB"""
    if not job_id:
        return {
            'statusCode': 400,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({"success": False, "error": "job_id is required"})
        }
    
    job_details = get_job_details(job_id)
    
    if not job_details:
        return {
            'statusCode': 404,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({"success": False, "error": "Job not found"})
        }
    
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
        'body': json.dumps({
            "success": True,
            "job_id": job_id,
            "status": job_details.get('status'),
            "created_at": job_details.get('created_at'),
            "updated_at": job_details.get('updated_at'),
            "result": job_details.get('result'),
            "error_message": job_details.get('error_message')
        })
    }

def queue_one_time_job(body, username):
    """Queue a one-time KG building job"""
    try:
        # Create job record
        job_id = create_job_record('one_time', body, username)
        
        # Send to SQS
        sqs_message = {
            'job_id': job_id,
            'job_type': 'one_time',
            'payload': body,
            'username': username
        }
        
        sqs_client.send_message(
            QueueUrl=kg_queue_url,
            MessageBody=json.dumps(sqs_message)
        )
        
        logger.info(f"Queued one-time job {job_id}")
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({
                "success": True,
                "job_id": job_id,
                "status": "QUEUED",
                "message": "Job queued successfully"
            })
        }
        
    except Exception as e:
        logger.error(f"Failed to queue job: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({"success": False, "error": str(e)})
        }




# ============================================================================
# SQS MESSAGE PROCESSING (Actual KG Building)
# ============================================================================
# In backend/lambda_functions/kg_processor/handler.py

def process_sqs_message(event, context):
    """Process SQS message - do actual KG building with job tracking"""
    
   
    for record in event['Records']:
        try:
            # Parse SQS message
            message_body = json.loads(record['body'])
            job_id = message_body['job_id']
            job_type = message_body['job_type']
            payload = message_body['payload'] # This is the outer payload with "action"
            username = message_body.get('username', 'system')
            
            logger.info(f"Processing {job_type} KG job {job_id}")
            
            # Update job status to PROCESSING
            update_job_status(job_id, 'PROCESSING')
            
            # Extract the inner payload which contains the actual arguments
            action = payload.get('action')
            actual_payload = payload
            # Route to appropriate KG building function using the inner payload
            if action == 'build_reddit_kg':
                result = _build_and_load_reddit_kg(actual_payload, username, job_id)
            elif action == 'build_reddit_kg_dummy':
                logger.info(f"Executing dummy action for job {job_id}. Real KG build is skipped.")
                
                # 2. Create a successful result dictionary to send back
                result = {
                    "success": True,
                    "message": "Dummy action successful. KG build will be executed in a real run.",
                    "job_id": job_id,
                    "action_executed": "build_reddit_kg_dummy"
                }
            elif action == 'load_static_kg':
                result = _load_static_kg(actual_payload, username, job_id)
            else:
                result = {"success": False, "error": f"Unknown action: {payload.get('action')}"}
            
            # --- END OF CORRECTED LOGIC ---
            
            # Update job status based on result
            if result.get('success'):
                summary_result = {
                    "message": result.get("message"),
                    "s3_uri": result.get("s3_uri"),
                    "post_count": result.get("post_count"),
                    "total_triples": result.get("metadata", {}).get("total_triples"),
                    "embedding_table": result.get("embedding_result", {}).get("table_name")
                }
                update_job_status(job_id, 'COMPLETED', summary_result)
                
                logger.info(f"Job {job_id} completed successfully")
            else:
                update_job_status(job_id, 'FAILED', error_message=result.get('error'))
                logger.error(f"Job {job_id} failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Error processing SQS message: {str(e)}")
            if 'job_id' in locals():
                update_job_status(job_id, 'FAILED', error_message=str(e))
    
    return {"statusCode": 200, "message": "SQS messages processed"}

# ============================================================================
# EVENTBRIDGE TICK HANDLING (Scheduled Jobs)
# ============================================================================

def handle_eventbridge_tick(event, context):
    """Handle EventBridge scheduled tick - queue work to SQS"""
    
    try:
        # Extract job details from EventBridge event
        job_id = event.get('job_id')
        payload = event.get('payload', {})
        
        if not job_id:
            return {"success": False, "error": "No job_id in EventBridge event ever"}
        
        logger.info(f"EventBridge tick for job {job_id}")
        
        # Check if job should continue (not past end time)
        job_details = get_job_details(job_id)
        if not job_details:
            logger.info(f"Job {job_id} not found, stopping schedule.")
            # --- CORRECTED LOGIC TO STOP THE JOB ---
            if "AWS_SAM_LOCAL" not in os.environ:
                schedule_name = job_id
                try:
                    scheduler_client.delete_schedule(Name=schedule_name, GroupName='default')
                except scheduler_client.exceptions.ResourceNotFoundException:
                    logger.warning(f"Schedule {schedule_name} not found, may have already been deleted.")
            return {"success": True, "message": "Job not found, schedule deleted."}
        
        # Check end time
        end_time_str = job_details.get('schedule', {}).get('end_time_iso')
        if end_time_str:
            # Parse the ISO string into a naive datetime object
            naive_end_time = datetime.fromisoformat(end_time_str)
        
            # Make the naive time "aware" by explicitly telling Python it represents UTC time
            aware_end_time = naive_end_time.replace(tzinfo=timezone.utc)
            
            # Get the current time as a timezone-aware UTC object for a correct comparison
            current_utc_time = datetime.now(timezone.utc)
            
            if current_utc_time >= aware_end_time:
                logger.info(f"Job {job_id} has reached its end time. Stopping the schedule.")
                return stop_monitoring_job({'job_id': job_id})
        
        # Queue the actual work to SQS
        sqs_message = {
            'job_id': job_id,
            'job_type': 'scheduled',
            'payload': payload,
            'username': job_details.get('created_by', 'system')
        }
        
        sqs_client.send_message(
            QueueUrl=kg_queue_url,
            MessageBody=json.dumps(sqs_message)
        )
        
        logger.info(f"Queued scheduled work for job {job_id}")
        
        # Periodic summarization: Generate summary every N iterations
        try:
            iteration_count = job_details.get('iteration_count', 0) + 1
            
            # Update iteration count
            jobs_table.update_item(
                Key={'job_id': job_id},
                UpdateExpression="SET iteration_count = :count",
                ExpressionAttributeValues={':count': iteration_count}
            )
            
            # Summarize every 1 iterations (configurable)
            SUMMARIZATION_INTERVAL = 1
            if iteration_count % SUMMARIZATION_INTERVAL == 0:
                logger.info(f"Generating summary for job {job_id} (iteration {iteration_count})")
                
                from kg_merger import merge_kgs_for_job
                #from kg_summarizer import summarize_merged_kg
                from common.graph_summary import summarize_graph 
                
                username = job_details.get('created_by', 'system')
                original_query = job_details.get('payload', {}).get('query')

                # Merge all KGs from this job
                merged_ttl = merge_kgs_for_job(job_id, username)
                
                if merged_ttl:
                    summary_data = None
                    tmp_file_path = None
                    try:
                        # --- FIX #1: Write content to a temp file ---
                        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ttl', encoding='utf-8') as tmp_file:
                            tmp_file.write(merged_ttl)
                            tmp_file_path = tmp_file.name

                        # Pass the FILE PATH to the summarizer
                        summary_data = summarize_graph(
                            ttl_path=tmp_file_path,
                            query_text=original_query
                        )
                    finally:
                        # Always clean up the temp file
                        if tmp_file_path and os.path.exists(tmp_file_path):
                            os.remove(tmp_file_path)

                    # --- FIX #2: Use the correct variable 'summary_data' ---
                    if summary_data and "error" not in summary_data:
                        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                        summary_key = f"knowledge_graph/{username}/summaries/{job_id}_summary_{timestamp}.json"
                        s3_client.put_object(
                            Bucket=bucket,
                            Key=summary_key,
                            Body=json.dumps(summary_data), # <-- Use the correct variable
                            ContentType='application/json'
                        )
                    
                        # Update job with summary reference
                        jobs_table.update_item(
                            Key={'job_id': job_id},
                            UpdateExpression="SET last_summary_key = :key, last_summary_time = :time",
                            ExpressionAttributeValues={
                                ':key': summary_key,
                                ':time': datetime.utcnow().isoformat()
                            }
                        )
                    
                        logger.info(f"Summary saved to {summary_key}")
                else:
                    logger.warning(f"No KGs found to merge for job {job_id}")
                    
        except Exception as e:
            logger.error(f"Summarization failed for job {job_id}: {str(e)}")
            # Don't fail the tick, just log the error
        
        return {"success": True, "message": f"Queued work for job {job_id}"}
        
    except Exception as e:
        logger.error(f"EventBridge tick error: {str(e)}")
        return {"success": False, "error": str(e)}

# ============================================================================
# JOB MANAGEMENT FUNCTIONS
# ============================================================================

def update_job_status(job_id, status, result=None, error_message=None):
    if not jobs_table:
        logger.warning("Jobs table not configured")
        return
    
    try:
        update_expression = "SET #status = :status, updated_at = :updated_at"
        expression_values = {
            ':status': status,
            ':updated_at': datetime.utcnow().isoformat()
        }
        # Start with the placeholder for 'status'
        expression_names = {'#status': 'status'}
        
        if result:
            # Add a placeholder for 'result' to the expression
            update_expression += ", #result = :result" # <-- FIX #1
            # Add the mapping for the placeholder
            expression_names['#result'] = 'result'      # <-- FIX #2
            expression_values[':result'] = result
            
        if error_message:
            # It's good practice to do this for all attributes
            update_expression += ", #error_message = :error"
            expression_names['#error_message'] = 'error_message'
            expression_values[':error'] = error_message
        
        jobs_table.update_item(
            Key={'job_id': job_id},
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expression_names,
            ExpressionAttributeValues=expression_values
        )
        
    except Exception as e:
        logger.error(f"Failed to update job status: {str(e)}")
        
        
def get_job_details(job_id):
    """Get job details from DynamoDB"""
    if not jobs_table:
        return None
    
    try:
        response = jobs_table.get_item(Key={'job_id': job_id})
        return response.get('Item')
    except Exception as e:
        logger.error(f"Failed to get job details: {str(e)}")
        return None

def create_job_record(job_type, payload, username, schedule=None):
    """Create initial job record in DynamoDB"""
    job_id = f"kg-{int(datetime.now().timestamp())}-{str(uuid.uuid4())[:8]}"
    
    job_record = {
        'job_id': job_id,
        'job_type': job_type,
        'status': 'QUEUED',
        'payload': payload,
        'created_by': username,
        'created_at': datetime.utcnow().isoformat(),
        'updated_at': datetime.utcnow().isoformat()
    }
    
    if schedule:
        job_record['schedule'] = schedule
    
    if jobs_table:
        try:
            jobs_table.put_item(Item=job_record)
        except Exception as e:
            logger.error(f"Failed to create job record: {str(e)}")
    
    return job_id

@tool
def start_monitoring_job(payload, username, context):
    """Start a scheduled monitoring job with EventBridge Scheduler."""
    try:
        query = payload.get('query')
        interval_field = payload.get('interval', 900)

        interval_seconds = interval_field.get('seconds') if isinstance(interval_field, dict) else interval_field
        end_time = None

        schedule = payload.get('schedule', {})
        if 'end_time_iso' in schedule:
            try:
                end_time = datetime.fromisoformat(schedule['end_time_iso'])
            except Exception:
                pass

        if not end_time:
            duration_hours = payload.get('duration', 2)
            end_time = datetime.utcnow() + timedelta(hours=duration_hours)

        if not query:
            return _error("query is required")

        # --- Create EventBridge rate expression ---
        if interval_seconds < 3600:
            minutes = int(interval_seconds / 60)
            rate = f"rate({minutes} {'minute' if minutes == 1 else 'minutes'})"
        else:
            hours = int(interval_seconds / 3600)
            rate = f"rate({hours} {'hour' if hours == 1 else 'hours'})"

        # --- Create job record ---
        schedule_info = {
            'interval_seconds': interval_seconds,
            'end_time_iso': end_time.isoformat(),
            'rate_expression': rate
        }
        job_id = create_job_record('scheduled', payload, username, schedule_info)

        if "AWS_SAM_LOCAL" not in os.environ:
            schedule_name = job_id  # simpler: match directly to CLI commands
            tick_payload = {"job_id": job_id, "payload": payload}

            scheduler_client.create_schedule(
                Name=schedule_name,
                GroupName='default',
                ScheduleExpression=rate,
                FlexibleTimeWindow={'Mode': 'OFF'},
                Target={
                    'Arn': context.invoked_function_arn,
                    'RoleArn': os.getenv('EVENTBRIDGE_ROLE_ARN'),
                    'Input': json.dumps(tick_payload)
                }
            )

        logger.info(f"Started monitoring job {job_id}")
        return _ok({
            "job_id": job_id,
            "status": "RUNNING",
            "message": "Monitoring job started",
            "schedule": schedule_info
        })

    except Exception as e:
        logger.error(f"Failed to start monitoring job: {str(e)}")
        return _error(str(e))


@tool
def stop_monitoring_job(payload):
    """Stop a monitoring job by deleting EventBridge schedule"""
    job_id = payload.get('job_id')
    if not job_id:
        return _error("job_id is required")

    try:
        if "AWS_SAM_LOCAL" not in os.environ:
            scheduler_client.delete_schedule(Name=job_id, GroupName='default')

        update_job_status(job_id, 'STOPPED')
        return _ok({"job_id": job_id, "message": "Job stopped"})

    except scheduler_client.exceptions.ResourceNotFoundException:
        logger.warning(f"Schedule {job_id} not found in Scheduler.")
        return _ok({"job_id": job_id, "message": "Already stopped or missing"})

    except Exception as e:
        logger.error(f"Failed to stop job: {str(e)}")
        return _error(f"Failed to stop job: {str(e)}")


@tool
def get_job_status(job_id: str):
    """
    Get the status of a KG building job.
    
    Args:
        job_id: Job identifier (e.g., 'kg-1234567890-abc')
    
    Returns:
        dict: Job details including status, timestamps, results
    """
    try:
        response = jobs_table.get_item(Key={'job_id': job_id})
        if 'Item' not in response:
            return {"success": False, "error": f"Job {job_id} not found"}
        
        job = response['Item']
        return {
            "success": True,
            "job_id": job_id,
            "status": job.get('status'),
            "created_at": job.get('created_at'),
            "updated_at": job.get('updated_at'),
            "result": job.get('result'),
            "error_message": job.get('error_message')
        }
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        return {"success": False, "error": str(e)}


def _ok(data):
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
        'body': json.dumps({"success": True, **data})
    }


def _error(message):
    return {
        'statusCode': 400,
        'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
        'body': json.dumps({"success": False, "error": message})
    }

# ============================================================================
# KG BUILDING FUNCTIONS (Adapted from original)
# ============================================================================
def _verify_ttl_content(unverified_ttl_content: str, job_id: str) -> tuple[str, list]:
    """
    Takes a string of TTL content, runs it through the verifier,
    and returns the verified TTL content string and the results.
    """
    temp_ttl_path = f"/tmp/{job_id}_initial.ttl"
    verified_ttl_path = None
    from agents.rumor_verifier_tavilly import RumorVerifierBatchLLM

    try:
        # Write the unverified content to a temporary file
        with open(temp_ttl_path, "w", encoding="utf-8") as f:
            f.write(unverified_ttl_content)
        
        logger.info(f"Starting verification for job {job_id} using file {temp_ttl_path}")
        
        # Run the verifier on the temporary file
        verifier = RumorVerifierBatchLLM(ttl_path=temp_ttl_path)
        verification_results, verified_ttl_path = verifier.verify_clusters()
        
        if not verified_ttl_path:
            raise Exception("Rumor verification process failed to produce an output file.")

        # Read the final, verified TTL content back into a string
        with open(verified_ttl_path, "r", encoding="utf-8") as f:
            verified_ttl_content = f.read()
        
        logger.info(f"Verification complete. Verified TTL content size: {len(verified_ttl_content)} bytes")
        
        return verified_ttl_content, verification_results

    finally:
        # Ensure temporary files are always cleaned up
        if os.path.exists(temp_ttl_path):
            os.remove(temp_ttl_path)
        if verified_ttl_path and os.path.exists(verified_ttl_path):
            os.remove(verified_ttl_path)
            
def _build_and_load_reddit_kg(payload, username, job_id):
    """Build Reddit KG - adapted from original with job tracking"""
    from agents.kg_building_agent import create_kg_builder_components 
    from common.embed import run_kg_embedding

    subreddit = payload.get("subreddit")
    
    if not subreddit:
        subreddit = payload.get("query")
    limit = payload.get("limit", 10)
    extract_claims = payload.get("extract_claims", True)
    
    if not subreddit:
        return {"success": False, "error": "Subreddit must be specified"}

    try:
        # 1. Fetch Reddit Posts
        from tools.discovery.reddit_mcp_tool import RedditMCPTool
        from tools.discovery.reddit_praw_tool import create_reddit_praw_tool
        
        logger.info(f"Fetching ss {limit} posts from {subreddit}")
        reddit_tool = RedditMCPTool.create_instance()
        fetch_result = reddit_tool.execute(subreddit=subreddit, limit=limit)
        if len(fetch_result.get("posts", []))<limit:
            pub_tool = create_reddit_praw_tool()
            fetch_result = pub_tool.execute(subreddit=subreddit, limit=limit)
        logger.info(f"Reddit tool fetch_result: {len(fetch_result['posts'])}")
        if not fetch_result.get("success"):
            return fetch_result
            
        posts = fetch_result.get("posts", [])
        if isinstance(posts, str):
            posts = json.loads(posts)
        
        logger.info(f"Fetched {len(posts)} posts successfully")
        
        # 2. Build Knowledge Graph
        logger.info("Building Knowledge Graph from posts")
        claims_agent, kg_builder = create_kg_builder_components()
        
        build_result = kg_builder.build_from_posts(
            posts=posts, 
            agent=claims_agent if extract_claims else None,
            extract_claims=extract_claims,
            platform="Reddit", 
            batch_size=BATCH_SIZE_CLAIMS
        )
        
        if not build_result.get("success"):
            return build_result
            
        unverified_ttl_content = build_result.get("kg_turtle")
        logger.info(f"KG built successfully. Size: {len(unverified_ttl_content)} bytes")
        kg_turtle_content, verification_results = _verify_ttl_content(unverified_ttl_content, job_id)
        # 3. Save to S3
        query = payload.get("query", f"Posts from {subreddit}")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        s3_key = save_reddit_kg_to_s3(username, kg_turtle_content, subreddit, query, len(posts), timestamp, job_id)
        
        # 4. Run embedding
        embedding_result = {}
        try:
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".ttl", encoding='utf-8') as tmp_file:
                tmp_file.write(kg_turtle_content)
                tmp_file_path = tmp_file.name
            
            logger.info(f"Running KG embedding process for user {username}")
            embedding_result = run_kg_embedding(tmp_file_path, username, source_name=subreddit, subreddit=subreddit, timestamp=timestamp, job_id=job_id)
            os.remove(tmp_file_path)
            
        except Exception as e:
            logger.error(f"KG embedding failed: {e}")
            embedding_result = {"success": False, "error": str(e)}
       
        return {
            "success": True,
            "message": f"KG built successfully for '{subreddit}'",
            "s3_uri": f"s3://{bucket}/{s3_key}",
            "embedding_result": embedding_result,
            "metadata": build_result.get("metadata"),
            "verification_results_len": len(verification_results),
            "post_count": len(posts), 
            
        }
        
    except Exception as e:
        logger.error(f"Failed during Reddit KG building: {e}")
        return {"success": False, "error": str(e)}

def _load_static_kg(payload, username, job_id):
    """Load static KG - adapted from original with job tracking"""
    from common.embed import run_kg_embedding
    kg_name = payload.get("name", "putinmissing")
    
    if kg_name != "putinmissing":
        return {"success": False, "error": "Only 'putinmissing' is supported"}

    try:
        is_local = os.getenv("ENVIRONMENT") == "dev"
        kg_content = ""

        if is_local:
            local_path = "data/putinmissing-all-rnr-threads_kg.ttl"
            logger.info(f"Reading static KG from local path: {local_path}")
            with open(local_path, "r", encoding="utf-8") as f:
                kg_content = f.read()
        else:
            # Production: download from S3
            s3_bucket = "madsiftpublic"
            s3_key = "putinmissing-all-rnr-threads_kg.ttl"
            logger.info(f"Downloading static KG from s3://{s3_bucket}/{s3_key}")
            s3_object = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            kg_content = s3_object['Body'].read().decode('utf-8')

        # Run embedding
        embedding_result = {}
        try:
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".ttl", encoding='utf-8') as tmp_file:
                tmp_file.write(kg_content)
                tmp_file_path = tmp_file.name
            
            logger.info(f"Running KG embedding process on static KG for user {username}")
            embedding_result = run_kg_embedding(tmp_file_path, username, source_name=kg_name)
            os.remove(tmp_file_path)
            
        except Exception as e:
            logger.error(f"Static KG embedding failed: {e}")
            embedding_result = {"success": False, "error": str(e)}
        
        return {
            "success": True,
            "message": f"Static KG '{kg_name}' loaded successfully",
            "embedding_result": embedding_result
        }
        
    except Exception as e:
        logger.error(f"Failed to load static KG: {e}")
        return {"success": False, "error": str(e)}

# ============================================================================
# HELPER FUNCTIONS (From original)
# ============================================================================

def save_reddit_kg_to_s3(username, ttl_data, subreddit, query, post_count, timestamp, job_id):
    """Save KG to S3 with job tracking"""
    prefix = f"knowledge_graph/{username}/"
    # Include job_id in filename for efficient merging and summarization
    key = f"{prefix}reddit_{subreddit}_{job_id}_{timestamp}.ttl"
    keywords = extract_keywords(query)
    
    metadata = {
        'Type': 'knowledge_graph',
        'Source': 'reddit',
        'Subreddit': subreddit,
        'Query': query,
        'CreatedBy': username,
        'CreatedAt': timestamp,
        'PostCount': str(post_count),
        'Keywords': ','.join(keywords),
        'JobId': job_id,
        'PairedWith': f"{prefix}reddit_{subreddit}_{job_id}_{timestamp}_vectorstore"
    }
    
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=ttl_data.encode("utf-8"),
        ContentType="text/turtle",
        Metadata=metadata
    )
    
    logger.info(f"Saved KG to s3://{bucket}/{key}")
    return key

def extract_keywords(query, max_keywords=4):
    """Extract meaningful keywords from query text"""
    import re
    
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
        'by', 'about', 'find', 'posts', 'get', 'show', 'tell', 'me', 'what', 
        'how', 'when', 'where', 'why', 'is', 'are', 'was', 'were', 'be', 'been'
    }
    
    words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
    keywords = []
    
    for word in words:
        if word not in stop_words and word not in keywords:
            keywords.append(word.capitalize())
            if len(keywords) >= max_keywords:
                break
    
    return keywords
