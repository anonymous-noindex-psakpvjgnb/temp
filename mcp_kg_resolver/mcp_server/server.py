import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal

# Import the tool functions
from kg_tools import search_wikidata_candidates, search_dbpedia_candidates, CandidateInfo

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MCPServer")

# --- FastAPI App ---
app = FastAPI(
    title="MCP Knowledge Graph Lookup Server",
    description="Exposes Wikidata and DBpedia lookups as an MCP-like service.",
    version="1.0.0"
)

# --- Pydantic Models for Request/Response ---
class KGLookupRequest(BaseModel):
    term: str = Field(..., description="The entity or property term to search for.")
    source_kg: Literal['wikidata', 'dbpedia'] = Field(..., description="The knowledge graph to query.")
    max_results: int = Field(5, description="Maximum number of candidates to return.", ge=1, le=20)
    # Optional: Add parameter for DBpedia SSL verification if needed often
    dbpedia_verify_ssl: bool = Field(True, description="Whether to verify SSL certificate for DBpedia calls.")

class KGLookupResponse(BaseModel):
    candidates: List[CandidateInfo] = Field(..., description="List of candidates found.")

class ServiceInfo(BaseModel):
    id: str
    description: str
    # Add input/output schema details if desired for stricter MCP adherence
    # input_schema: Dict = Field(...)
    # output_schema: Dict = Field(...)

class DiscoveryResponse(BaseModel):
    services: List[ServiceInfo]

# --- MCP-like Endpoints ---

@app.get("/discover", response_model=DiscoveryResponse, summary="Discover available services")
async def discover_services():
    """Provides information about the services hosted by this server."""
    return {
        "services": [
            {
                "id": "knowledge-graph-lookup",
                "description": "Searches Wikidata or DBpedia for candidates matching a term."
                # Add detailed schema if needed:
                # "input_schema": KGLookupRequest.schema(),
                # "output_schema": KGLookupResponse.schema(),
            }
        ]
    }

@app.post("/invoke/knowledge-graph-lookup", response_model=KGLookupResponse, summary="Invoke the KG Lookup Service")
async def invoke_kg_lookup(request: KGLookupRequest):
    """Handles requests to the knowledge-graph-lookup service."""
    logger.info(f"Received invoke request for term='{request.term}', source='{request.source_kg}'")
    try:
        if request.source_kg == 'wikidata':
            candidates = search_wikidata_candidates(request.term, request.max_results)
        elif request.source_kg == 'dbpedia':
            # Pass the SSL verification flag
            candidates = search_dbpedia_candidates(request.term, request.max_results, verify_ssl=request.dbpedia_verify_ssl)
        else:
            # Should be caught by Pydantic/Literal, but good practice
            raise HTTPException(status_code=400, detail=f"Unsupported source_kg: {request.source_kg}")

        return {"candidates": candidates}

    # Handle specific errors raised by the tools
    except (ConnectionError, ConnectionAbortedError) as e:
        logger.error(f"Tool connection error for '{request.term}' in {request.source_kg}: {e}")
        raise HTTPException(status_code=503, detail=f"Service Unavailable: Upstream KG API request failed - {e}")
    except ValueError as e: # JSON errors or other value issues from tools
        logger.error(f"Tool data error for '{request.term}' in {request.source_kg}: {e}")
        raise HTTPException(status_code=502, detail=f"Bad Gateway: Invalid response from upstream KG API - {e}")
    except RuntimeError as e: # Catch-all for unexpected tool errors
         logger.error(f"Tool runtime error for '{request.term}' in {request.source_kg}: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Internal Server Error: Tool failed unexpectedly - {e}")
    except Exception as e:
         logger.error(f"Unexpected server error processing request for '{request.term}': {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@app.get("/health", summary="Health Check")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}

# --- Run Instructions (for local testing) ---
# Use: uvicorn server:app --reload --port 8100
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting MCP Server on http://localhost:8100")
    # Note: --reload is great for development but use multiple workers in production
    uvicorn.run("server:app", host="0.0.0.0", port=8100, reload=True)