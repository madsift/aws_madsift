# In: backend/agents/kg_building_agent.py
import sys
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional

import rdflib
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, XSD
from pydantic import BaseModel, Field
from strands import Agent

from common.llm_models import get_default_model

# Original model for a single claim
class ClaimModel(BaseModel):
    claim_summary: str = Field(description="A short, factual, self-contained claim extracted from the text.")
    language: str = Field(description="The language code of the claim, e.g., 'en'.")


class SingleTextClaimResult(BaseModel):
    text_id: int = Field(description="The original numeric ID of the text this result corresponds to.")
    claims: List[ClaimModel]

class BatchClaimExtractionResult(BaseModel):
    results: List[SingleTextClaimResult]

class ClaimsExtractionAgent:
    """A simple, stateless agent focused ONLY on extracting claims from text in batches."""
    def __init__(self):
        self.agent = None
        self._initialize_agent()

    def _initialize_agent(self):
        try:
            model = get_default_model()
            if not model:
                raise Exception("Could not get default model (Bedrock or Ollama).")
            
            self.agent = Agent(
                model=model,
                system_prompt="You are an AI assistant that extracts factual claims from a given list of texts. Respond ONLY with the JSON structure requested."
            )
            print("✅ ClaimsExtractionAgent initialized successfully.")
        except Exception as e:
            print(f"⚠️ ClaimsExtractionAgent initialization failed: {e}")

    def extract_claims_from_texts(self, texts: List[str], max_claims_per_text: int = 3) -> Dict[int, List[Dict[str, str]]]:
        """
        Extracts claims from a LIST of texts in a single batch call.
        Returns a dictionary mapping the original text index to its list of claims.
        """
        if not self.agent or not texts:
            return {}
            
        # Build a numbered list of texts for the prompt
        prompt_texts = "\n".join([f"Text {i+1}: \"{text}\"" for i, text in enumerate(texts)])
        
        prompt = (
            f"For each of the following texts, extract up to {max_claims_per_text} concise factual claims.\n"
            "Match the output JSON schema exactly, providing a result for each text_id.\n\n"
            f"{prompt_texts}"
        )
        
        try:
            # Use the agent's structured_output with our BATCH Pydantic model
            batch_result = self.agent.structured_output(BatchClaimExtractionResult, prompt)
            
            # Convert the Pydantic result into a simple dictionary for the caller
            final_claims_map = {}
            for res in batch_result.results:
                original_index = res.text_id - 1 # Convert from 1-based ID to 0-based index
                claims_for_text = []
                for c in res.claims[:max_claims_per_text]:
                    claim_text = (c.claim_summary or "").strip()
                    lang = (c.language or "und")
                    if claim_text:
                        claims_for_text.append({"claim": claim_text, "language": lang})
                final_claims_map[original_index] = claims_for_text
            return final_claims_map
        except Exception as e:
            print(f"⚠️ Batch claim extraction failed: {e}")
            return {}

# --- The Simplified Knowledge Graph Tool ---
class SimplifiedKGT:
    """
    A simplified tool for building the ontology-based knowledge graph.
    It is no longer a 'Tool' in the agent sense, just a builder class.
    """
    # --- Ontology Namespaces (Preserved as requested) ---
    EX = Namespace("http://example.org/")
    SIOC = Namespace("http://rdfs.org/sioc/ns#")
    SCHEMA = Namespace("http://schema.org/")
    PROV = Namespace("http://www.w3.org/ns/prov#")

    def build_from_posts(self, posts: List[Dict], agent: Optional[ClaimsExtractionAgent], extract_claims: bool, platform: str, batch_size: int = 64) -> Dict[str, Any]:
        """The main execution method to build the graph, now with chunked batch processing."""
        try:
            g = Graph()
            g.bind("ex", self.EX)
            g.bind("sioc", self.SIOC)
            g.bind("schema", self.SCHEMA)
            g.bind("prov", self.PROV)

            platform_uri = self.EX[platform.replace(" ", "")]
            g.add((platform_uri, RDF.type, self.EX.Platform))
            g.add((platform_uri, self.SCHEMA.name, Literal(platform)))

 
            all_extracted_claims = {} # This will store claims for ALL posts
            if extract_claims and agent:
                print(f"Processing {len(posts)} posts in chunks of size {batch_size}...")
                
                # Break the posts into smaller chunks
                post_chunks = [posts[i:i + batch_size] for i in range(0, len(posts), batch_size)]

                for chunk_index, chunk in enumerate(post_chunks):
                    print(f"  - Processing chunk {chunk_index + 1}/{len(post_chunks)}...")
                    
                    # 1. Prepare the batch of texts for THIS CHUNK
                    texts_to_process = []
                    post_indices_in_chunk = []
                    for i, post in enumerate(chunk):
                        text = post.get("text") or post.get("title") or post.get("selftext")
                        if text:
                            texts_to_process.append(text)
                            post_indices_in_chunk.append(i)
                    
                    if not texts_to_process:
                        continue

                    # 2. Call the agent ONCE for this chunk
                    chunk_claims_map = agent.extract_claims_from_texts(texts_to_process)

                    # 3. Map the results from the chunk back to the original post index
                    for batch_index, claims_list in chunk_claims_map.items():
                        # Get the index of the post within the current chunk
                        post_index_in_chunk = post_indices_in_chunk[batch_index]
                        # Calculate the original index of the post in the full list
                        original_post_index = (chunk_index * batch_size) + post_index_in_chunk
                        all_extracted_claims[original_post_index] = claims_list
            
            # Process all posts and add them to the graph
            for i, post in enumerate(posts):
                post_id = post.get("id") or post.get("id_str")
                if not post_id: continue

                post_node = URIRef(self.EX + f"post/{post_id}")
                g.add((post_node, RDF.type, self.SIOC.Post))

                text = post.get("text") or post.get("title") or post.get("selftext")
                if text:
                    g.add((post_node, self.SCHEMA.headline, Literal(text)))

                # Add the pre-extracted claims for this post
                claims_for_this_post = all_extracted_claims.get(i, [])
                for c in claims_for_this_post:
                    claim_text = c.get("claim", "")
                    if claim_text:
                        self._create_claim_node(g, claim_text, c.get("language", "und"), post_node)
            
            metadata = self._calculate_metadata(g, posts)
            
            return {
                "success": True,
                "kg_turtle": g.serialize(format="turtle"),
                "metadata": metadata
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to build knowledge graph: {e}"}

    def _create_claim_node(self, g: Graph, claim_text: str, language: str, post_node: URIRef):
        """Helper to create a claim node in the graph."""
        cid = hashlib.sha1(claim_text.encode("utf-8")).hexdigest()[:10]
        claim_node = URIRef(self.EX + f"claim/{cid}")
        g.add((claim_node, RDF.type, self.EX.Claim))
        g.add((claim_node, self.EX.canonicalForm, Literal(claim_text)))
        g.add((claim_node, self.EX.hasLanguage, Literal(language)))
        g.add((claim_node, self.PROV.wasDerivedFrom, post_node))

    def _calculate_metadata(self, g: Graph, posts: List[Dict]) -> Dict[str, Any]:
        """Calculates simple metadata about the generated graph."""
        return {
            "total_triples": len(g),
            "posts_processed": len(posts),
            "claims_extracted": len(list(g.subjects(RDF.type, self.EX.Claim))),
            "build_timestamp": datetime.utcnow().isoformat() + "Z"
        }

def create_kg_builder_components():
    """A single factory to create and return the necessary components."""
    claims_agent = ClaimsExtractionAgent()
    kg_builder = SimplifiedKGT()
    return claims_agent, kg_builder



if __name__ == "__main__":
    print("--- Running Local Test for kg_building_agent.py ---")

    # 1. Create mock post data
    mock_posts = [
        {
            "id": "post1",
            "title": "Major breakthrough in AI research announced by scientists.",
            "selftext": "Researchers have developed a new algorithm that can learn from a single example, a significant leap in one-shot learning capabilities."
        },
        {
            "id": "post2",
            "title": "New regulations on cryptocurrency are expected next month.",
            "selftext": "Government officials are finalizing a framework that will require all exchanges to register with the financial authority."
        },
        {
            "id": "post3",
            "title": "A post with no real claims, just opinions.",
            "selftext": "I think this is the best movie of the year, everyone should watch it."
        }, 
        {
            "id": "post3",
            "title": "A post with no real claims, just opinions.",
            "selftext": "I think this is the best movie of the year, everyone should watch it."
        },
        {
            "id": "post3",
            "title": "A post with no real claims, just opinions.",
            "selftext": "I think this is the best movie of the year, everyone should watch it."
        }, 
        {
            "id": "post3",
            "title": "A post with no real claims, just opinions.",
            "selftext": "I think this is the best movie of the year, everyone should watch it."
        }, 
    ]

    # 2. Instantiate components using the factory
    claims_agent, kg_builder = create_kg_builder_components()

    # 3. Execute the build process
    print("\n--- Building Knowledge Graph (with claim extraction) ---")
    result = kg_builder.build_from_posts(
        posts=mock_posts,
        agent=claims_agent,
        extract_claims=True,
        platform="TestPlatform", 
        batch_size=4
    )

    # 4. Print the results
    if result.get("success"):
        print("\n✅ Build successful!")
        print("\n--- Metadata ---")
        print(json.dumps(result.get("metadata"), indent=2))
        print("\n--- Generated TTL Content ---")
        print(result.get("kg_turtle"))
    else:
        print("\n❌ Build failed!")
        print(f"Error: {result.get('error')}")

