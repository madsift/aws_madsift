#!/usr/bin/env python3
"""
Path utilities for KG and LanceDB path resolution
"""

def get_corresponding_ldb_path(kg_path: str) -> str:
    """Convert KG path to corresponding LanceDB path"""
    if not kg_path:
        return None
        
    # Handle public files
    if kg_path == "s3://madsiftpublic/putinmissing-all-rnr-threads_kg.ttl":
        return "s3://madsiftpublic/vector_store"
    
    # For user files: reddit_worldnews_20241210_143022.ttl -> reddit_worldnews_20241210_143022_vectorstore
    if kg_path.endswith('.ttl'):
        base_path = kg_path[:-4]  # Remove .ttl
        return f"{base_path}_vectorstore"
    
    return None

def extract_metadata_from_kg_path(kg_path: str) -> dict:
    """Extract metadata from KG path for display purposes"""
    if not kg_path:
        return {}
        
    # Handle public files
    if kg_path == "s3://madsiftpublic/putinmissing-all-rnr-threads_kg.ttl":
        return {
            'display_name': 'Public: Putin Missing - All Threads (Static)',
            'source': 'public',
            'subreddit': 'multiple',
            'query': 'Putin missing rumors verification',
            'created_at': 'Static Dataset',
            'post_count': 'N/A'
        }
    
    # Extract from filename pattern: reddit_{subreddit}_{timestamp}.ttl
    try:
        filename = kg_path.split('/')[-1]  # Get filename
        if filename.startswith('reddit_') and filename.endswith('.ttl'):
            parts = filename[7:-4].split('_')  # Remove 'reddit_' prefix and '.ttl' suffix
            if len(parts) >= 3:
                subreddit = parts[0]
                # Last two parts are date and time
                date_part = parts[-2]  # YYYYMMDD
                time_part = parts[-1]  # HHMMSS
                
                # Format timestamp for display
                try:
                    from datetime import datetime
                    dt = datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
                    formatted_date = dt.strftime("%b %d, %Y %H:%M")
                except:
                    formatted_date = f"{date_part}_{time_part}"
                
                return {
                    'display_name': f"Reddit: {subreddit} ({formatted_date})",
                    'source': 'reddit',
                    'subreddit': subreddit,
                    'created_at': formatted_date,
                    'filename': filename
                }
    except:
        pass
    
    # Fallback for unrecognized patterns
    return {
        'display_name': kg_path.split('/')[-1],
        'source': 'unknown',
        'filename': kg_path.split('/')[-1]
    }