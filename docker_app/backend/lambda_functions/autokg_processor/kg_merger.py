"""
KG Merger - Merge multiple KG files from the same monitoring job
"""
import boto3
from rdflib import Graph
import logging
import os

logger = logging.getLogger(__name__)
s3_client = boto3.client('s3')

# Environment variables
bucket = os.getenv("KG_BUCKET", "madsift")


def merge_kgs_for_job(job_id: str, username: str) -> str:
    """
    Merge all KG files from a monitoring job.
    
    Args:
        job_id: Job identifier (e.g., 'kg-1705501822-abc')
        username: User who owns the KGs
    
    Returns:
        str: Merged TTL content
    """
    logger.info(f"Merging KGs for job {job_id}, user {username}")
    
    # List all objects in user's KG directory
    prefix = f"knowledge_graph/{username}/"
    
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        if 'Contents' not in response:
            logger.warning(f"No KGs found for user {username}")
            return None
        
        # Filter for this job's KGs (must have job_id in key and end with .ttl)
        job_kgs = [
            obj['Key'] for obj in response['Contents']
            if job_id in obj['Key'] and obj['Key'].endswith('.ttl')
        ]
        
        if not job_kgs:
            logger.warning(f"No KGs found for job {job_id}")
            return None
        
        logger.info(f"Found {len(job_kgs)} KGs for job {job_id}: {job_kgs}")
        
        # Merge using rdflib
        merged_graph = Graph()
        
        for key in job_kgs:
            try:
                logger.info(f"Merging {key}")
                obj = s3_client.get_object(Bucket=bucket, Key=key)
                ttl_content = obj['Body'].read().decode('utf-8')
                merged_graph.parse(data=ttl_content, format='turtle')
            except Exception as e:
                logger.error(f"Failed to merge {key}: {str(e)}")
                # Continue with other files
        
        # Serialize merged graph
        merged_ttl = merged_graph.serialize(format='turtle')
        logger.info(f"Merged graph: {len(merged_graph)} triples, {len(merged_ttl)} bytes")
        
        return merged_ttl
        
    except Exception as e:
        logger.error(f"Error merging KGs for job {job_id}: {str(e)}", exc_info=True)
        return None
