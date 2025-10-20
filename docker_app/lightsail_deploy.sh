#!/bin/bash
# lightsail_deploy.sh
# Builds the Streamlit UI, pushes it to ECR, and deploys to Lightsail with correct environment variables.

#set -euo pipefail  # Fail fast, treat unset vars as errors

# -------------------------------------------------------------
# 🔧 Global Configuration Variables
# -------------------------------------------------------------
AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""
AWS_DEFAULT_REGION=""

# ⚠️ Ensure this is your actual Cognito domain (NOT user pool ID)
COGNITO_DOMAIN=""
COGNITO_CLIENT_ID=""
COGNITO_REDIRECT_URI=""

PRIVATE_BUCKET="madsift"
PRIVATE_PREFIX="knowledge_graph"

# Stack and service configuration
SERVICE_NAME="madsift-ui"
ECR_LOGICAL_ID="StreamlitECR"
STACK_NAME="madsift-dev"
LOCAL_IMAGE_NAME="madsift-ui"
LOCAL_IMAGE_TAG="latest"

# -------------------------------------------------------------
# 🧩 Step 1: Get CloudFormation Outputs
# -------------------------------------------------------------
echo "--- Retrieving CloudFormation outputs for stack: $STACK_NAME ---"

ECR_REPO_URI=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --query "Stacks[0].Outputs[?OutputKey=='${ECR_LOGICAL_ID}Uri'].OutputValue" \
    --output text)

if [ -z "$ECR_REPO_URI" ]; then
    echo "❌ Error: Could not find ECR repository URI for '$ECR_LOGICAL_ID' in stack '$STACK_NAME'."
    exit 1
fi
echo "✅ Found ECR Repository: $ECR_REPO_URI"

API_BASE_URL=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --query "Stacks[0].Outputs[?OutputKey=='ApiGatewayUrl'].OutputValue" \
    --output text)

if [ -z "$API_BASE_URL" ]; then
    echo "❌ Error: Could not find API Gateway URL in stack '$STACK_NAME'."
    exit 1
fi
echo "✅ Found API Gateway URL: $API_BASE_URL"

# -------------------------------------------------------------
# 🔐 Step 2: Log in to ECR
# -------------------------------------------------------------
echo "--- Logging into Amazon ECR ---"
aws ecr get-login-password --region "$AWS_DEFAULT_REGION" \
    | docker login --username AWS --password-stdin "$ECR_REPO_URI"

# -------------------------------------------------------------
# 🏗️ Step 3: Build and Push Docker Image
# -------------------------------------------------------------
echo "--- Building local Docker image ---"
docker build -t "${LOCAL_IMAGE_NAME}:${LOCAL_IMAGE_TAG}" ./ui

echo "--- Tagging image for ECR ---"
docker tag "${LOCAL_IMAGE_NAME}:${LOCAL_IMAGE_TAG}" "${ECR_REPO_URI}:${LOCAL_IMAGE_TAG}"

echo "--- Pushing image to ECR ---"
docker push "${ECR_REPO_URI}:${LOCAL_IMAGE_TAG}"
echo "✅ Successfully pushed image to ECR."

# -------------------------------------------------------------
# 🚀 Step 4: Deploy to Lightsail
# -------------------------------------------------------------
IMAGE_REF="${ECR_REPO_URI}:${LOCAL_IMAGE_TAG}"

echo "--- Preparing Lightsail deployment ---"
echo "Environment variables being passed:"
echo "  API_BASE_URL:        $API_BASE_URL"
echo "  AWS_DEFAULT_REGION:  $AWS_DEFAULT_REGION"
echo "  COGNITO_DOMAIN:      $COGNITO_DOMAIN"
echo "  COGNITO_CLIENT_ID:   $COGNITO_CLIENT_ID"
echo "  COGNITO_REDIRECT_URI:$COGNITO_REDIRECT_URI"

aws lightsail create-container-service-deployment \
  --service-name "$SERVICE_NAME" \
  --containers "{
    \"streamlit-ui\": {
      \"image\": \"$IMAGE_REF\",
      \"environment\": {
        \"API_BASE_URL\": \"$API_BASE_URL\",
        \"AWS_ACCESS_KEY_ID\": \"$AWS_ACCESS_KEY_ID\",
        \"AWS_SECRET_ACCESS_KEY\": \"$AWS_SECRET_ACCESS_KEY\",
        \"AWS_DEFAULT_REGION\": \"$AWS_DEFAULT_REGION\",
        \"COGNITO_DOMAIN\": \"$COGNITO_DOMAIN\",
        \"COGNITO_CLIENT_ID\": \"$COGNITO_CLIENT_ID\",
        \"COGNITO_REDIRECT_URI\": \"$COGNITO_REDIRECT_URI\",
        \"PRIVATE_BUCKET\": \"$PRIVATE_BUCKET\",
        \"PRIVATE_PREFIX\": \"$PRIVATE_PREFIX\"
      },
      \"ports\": { \"8501\": \"HTTP\" }
    }
  }" \
  --public-endpoint "{
    \"containerName\": \"streamlit-ui\",
    \"containerPort\": 8501
  }"

echo "✅ Deployment to Lightsail initiated successfully!"
