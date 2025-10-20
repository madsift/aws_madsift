# Use AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.11
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy common modules first (for layer caching)
COPY common/ ${LAMBDA_TASK_ROOT}/common/
COPY agents/ ${LAMBDA_TASK_ROOT}/agents/
COPY tools/ ${LAMBDA_TASK_ROOT}/tools/
COPY data/ ${LAMBDA_TASK_ROOT}/data/ 

COPY lambda_functions/autokg_processor/requirements.txt ${LAMBDA_TASK_ROOT}/

# Install the dependencies from the requirements file
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY lambda_functions/autokg_processor/ ${LAMBDA_TASK_ROOT}/



# Set the CMD to your handler
CMD ["handler.lambda_handler"]
