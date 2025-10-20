"""
KG Summarizer - Generate summaries of merged knowledge graphs
"""
import logging
from datetime import datetime
from rdflib import Graph, Namespace, RDF, RDFS

logger = logging.getLogger(__name__)

# Define common namespaces
EX = Namespace("http://example.org/")


def count_claims(graph: Graph) -> int:
    """Count claims in the graph"""
    try:
        # Adjust this based on your actual schema
        # Assuming claims have a specific type or predicate
        claims = set(graph.subjects(RDF.type, EX.Claim))
        return len(claims)
    except Exception as e:
        logger.warning(f"Could not count claims: {str(e)}")
        return 0


def count_entities(graph: Graph) -> int:
    """Count unique entities in the graph"""
    try:
        # Count unique subjects
        entities = set(graph.subjects())
        return len(entities)
    except Exception as e:
        logger.warning(f"Could not count entities: {str(e)}")
        return 0


def summarize_merged_kg(merged_ttl: str, job_id: str, ttl_path: str = None, ldb_path: str = None) -> dict:
    """
    Generate summary of merged KG using graph_summary.py for intelligent analysis.
    
    Args:
        merged_ttl: Merged TTL content (the actual data)
        job_id: Job identifier
        ttl_path: Optional S3 path to TTL file (for graph_summary)
        ldb_path: Optional LanceDB path for semantic context
    
    Returns:
        dict: Summary with key metrics and narrative
    """
    import tempfile
    import os
    
    logger.info(f"Summarizing merged KG for job {job_id}")
    
    try:
        # Parse graph for basic metrics
        graph = Graph()
        graph.parse(data=merged_ttl, format='turtle')
        
        # Count metrics
        total_triples = len(graph)
        total_claims = count_claims(graph)
        total_entities = count_entities(graph)
        
        logger.info(f"Basic metrics: {total_triples} triples, {total_claims} claims, {total_entities} entities")
        
        # Generate intelligent narrative summary using graph_summary
        narrative_summary = None
        
        # Save merged_ttl to temp file and use that for summarization
        if merged_ttl:
            tmp_file_path = None
            try:
                from common.graph_summary import summarize_graph
                
                # Write TTL data to temp file
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ttl', encoding='utf-8') as tmp_file:
                    tmp_file.write(merged_ttl)
                    tmp_file_path = tmp_file.name
                
                logger.info(f"Saved TTL to temp file: {tmp_file_path}")
                logger.info(f"Generating intelligent summary using graph_summary")
                
                # Use the intelligent summarization with local temp file
                graph_summary_result = summarize_graph(
                    ttl_path=tmp_file_path,
                    query_text=None, #ACTUALLY NEED TO GET THE query from UI (NOT IMPLEMENTED YET)
                    ldb_path=ldb_path,
                    limit=200,
                    top_k=5
                )
                
                if "error" not in graph_summary_result:
                    narrative_summary = graph_summary_result.get("summary", "")
                    logger.info("Intelligent summary generated successfully")
                else:
                    logger.warning(f"graph_summary returned error: {graph_summary_result.get('error')}")
                    
            except Exception as e:
                logger.warning(f"Intelligent summarization failed, using basic summary: {str(e)}")
            finally:
                # Clean up temp file
                if tmp_file_path and os.path.exists(tmp_file_path):
                    try:
                        os.remove(tmp_file_path)
                        logger.info(f"Cleaned up temp file: {tmp_file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up temp file: {e}")
        
        # Fallback to basic summary if intelligent summary failed
        if not narrative_summary:
            narrative_summary = f"Knowledge graph contains {total_triples} triples describing {total_entities} entities"
            if total_claims > 0:
                narrative_summary += f" with {total_claims} extracted claims"
        
        summary = {
            "job_id": job_id,
            "total_triples": total_triples,
            "total_claims": total_claims,
            "total_entities": total_entities,
            "summary": narrative_summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Summary completed for job {job_id}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error summarizing KG for job {job_id}: {str(e)}", exc_info=True)
        return {
            "job_id": job_id,
            "total_triples": 0,
            "total_claims": 0,
            "total_entities": 0,
            "summary": {"error": str(e)},
            "timestamp": datetime.utcnow().isoformat()
        }
