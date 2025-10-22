# In: lambda_functions/chat_agent/handler.py
import json
import logging
import os
from pythonjsonlogger import jsonlogger


from common.cognito_auth import lambda_handler_with_auth, get_s3_user_prefix
from common.graph_summary import summarize_graph
from enhanced_chat_agent import EnhancedChatAgent # Import your refactored class

bucket = os.getenv("KG_BUCKET")
env = os.getenv("ENVIRONMENT", "dev")
region = os.getenv("AWS_REGION", "us-east-1")

print("Initializing EnhancedChatAgent...")
# We pass a placeholder session_id; it will be updated per request.
_enhanced_agent = EnhancedChatAgent(session_id="global_agent")
print("EnhancedChatAgent initialized.")
# Setup structured logging
logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)


@lambda_handler_with_auth
def lambda_handler(event, context, user_info):
    """
    API Gateway handler for the enhanced chat agent.
    """
   
    logger.info({
        "event_type": "request",
        "bucket":  bucket,
        "lambda_name": context.function_name,
        "aws_request_id": context.aws_request_id,
        "environment": os.getenv("ENVIRONMENT"),
        "input_summary": str(event)[:500]
    })
    username = user_info["username"]

    try:
        # 1. Parse the incoming request body from the API Gateway event
        body = json.loads(event.get('body', '{}'))
        kg_path = body.get('kg_path')  # ✅ new field from frontend
        ldb_path = body.get('ldb_path')
        ldb_table_name = body.get('ldb_table_name')  # ✅ new field for table name
        user_message = body.get('message')
        session_id = body.get('session_id')
        chat_options = body.get('chat_options', {})

        if not user_message or not session_id:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing message or session_id in request body'})
            }
        print ("Session_id", session_id)
        # 2. Update the global agent's session_id for this specific request
        # This is a safe way to handle state in a "warm" Lambda environment
        if body.get('type')=='chat':
            # Check if we need to reinitialize (session change OR dataset change)
            current_kg_path = getattr(_enhanced_agent.kg_operations, 'kg_path', None)
            current_ldb_path = getattr(_enhanced_agent.kg_operations, 'ldb_path', None)

            needs_reinit = (
                _enhanced_agent.session_id != session_id or
                not _enhanced_agent.initialized or
                current_kg_path != kg_path or
                current_ldb_path != ldb_path
            )

            if needs_reinit:
                print(f"Reinitializing agent - Session: {_enhanced_agent.session_id} -> {session_id}, KG: {current_kg_path} -> {kg_path}")
                _enhanced_agent.session_id = session_id
                _enhanced_agent._initialize_agent(bucket, username, kg_path, ldb_path, region, ldb_table_name)

            # 3. Call your core agent logic to get a response
            agent_response = _enhanced_agent.get_response(user_message, chat_options)

            # 4. Return a successful API response
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*' # Important for web UIs
                },
                'body': json.dumps(agent_response)
            }
        if body.get('type')=='summarize':
            output = summarize_graph(kg_path, query_text='', ldb_path=ldb_path)
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*' # Important for web UIs
                },
                'body': json.dumps(output)
            }
    except Exception as e:
        print(f"Error in lambda_handler: {e}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Internal server error', 'message': str(e)})
        }

def health_check(event, context):
    """
    A simple health check function that returns a 200 OK response.
    """
    print("Health check endpoint invoked.")

    response_body = {
        "status": "healthy",
        "message": "Backend services are running.",
        "environment": os.getenv("ENVIRONMENT", "unknown")
    }

    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(response_body)
    }
