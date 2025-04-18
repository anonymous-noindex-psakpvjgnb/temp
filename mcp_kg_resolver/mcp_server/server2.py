import logging
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Optional, Any, Literal
import json # Import json

# Import KG tool functions
from kg_tools import search_wikidata_candidates, search_dbpedia_candidates, CandidateInfo

# Import SPARQLWrapper for execution tool
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MCPServer")

# --- FastAPI App ---
app = FastAPI(
    title="MCP KG Lookup & SPARQL Execution Server",
    description="Exposes KG lookups and SPARQL execution as MCP-like services.",
    version="1.1.0" # Version bump
)

# --- Pydantic Models ---

# KG Lookup Models (from previous step)
class KGLookupRequest(BaseModel):
    term: str = Field(..., description="The entity or property term to search for.")
    source_kg: Literal['wikidata', 'dbpedia'] = Field(..., description="The knowledge graph to query.")
    max_results: int = Field(5, description="Maximum number of candidates to return.", ge=1, le=20)
    dbpedia_verify_ssl: bool = Field(True, description="Whether to verify SSL certificate for DBpedia calls.")

class KGLookupResponse(BaseModel):
    candidates: List[CandidateInfo] = Field(..., description="List of candidates found.")

# SPARQL Execution Models
class SPARQLExecutionRequest(BaseModel):
    endpoint_url: HttpUrl = Field(..., description="The SPARQL endpoint URL to query.")
    query: str = Field(..., description="The SPARQL query string to execute.")
    # Optional: Add timeout parameter if needed

# Using Any for results as SPARQLWrapper returns complex dict
class SPARQLExecutionResponse(BaseModel):
    results: Any = Field(..., description="Results from SPARQLWrapper in JSON format.")


# Service Info Models (from previous step)
class ServiceInfo(BaseModel):
    id: str
    description: str

class DiscoveryResponse(BaseModel):
    services: List[ServiceInfo]


# --- Tool Functions ---

def execute_sparql_tool(endpoint_url: str, query: str, user_agent: str) -> Dict[str, Any]:
    """Tool: Executes a SPARQL query against an endpoint."""
    logger.info(f"Tool: Executing SPARQL query on {endpoint_url}")
    logger.debug(f"Tool: Query:\n{query}")
    sparql = SPARQLWrapper(str(endpoint_url), agent=user_agent) # SPARQLWrapper needs string URL
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(30) # Set a reasonable timeout

    try:
        results = sparql.query().convert()
        logger.info(f"Tool: SPARQL query executed successfully on {endpoint_url}.")
        return results # Return the raw dictionary from SPARQLWrapper
    except SPARQLExceptions.QueryBadFormed as e:
        logger.error(f"Tool: SPARQL QueryBadFormed on {endpoint_url}: {e}")
        raise ValueError(f"SPARQL query is syntactically incorrect: {e}") from e
    except SPARQLExceptions.EndPointNotFound as e:
         logger.error(f"Tool: SPARQL EndPointNotFound: {endpoint_url}: {e}")
         raise ConnectionError(f"SPARQL endpoint not found or invalid: {endpoint_url}") from e
    except SPARQLExceptions.EndPointInternalError as e:
        logger.error(f"Tool: SPARQL EndPointInternalError on {endpoint_url}: {e}")
        raise ConnectionAbortedError(f"SPARQL endpoint reported an internal error: {e}") from e
    except json.JSONDecodeError as e:
        # Sometimes errors come back as non-JSON
        logger.error(f"Tool: SPARQL result JSONDecodeError on {endpoint_url}: {e}. Raw response: {sparql.response.read()[:500]}...")
        raise ValueError(f"SPARQL endpoint returned non-JSON response: {e}") from e
    except Exception as e:
        # Catch other potential issues like timeouts wrapped differently, etc.
        logger.error(f"Tool: Unexpected SPARQL execution error on {endpoint_url}: {e}", exc_info=True)
        # Try to get more context if possible (depends on SPARQLWrapper version/exception type)
        err_context = str(e)
        if hasattr(e, 'response'):
             try:
                 err_context += f" | Response: {e.response.read()[:500]}..."
             except: pass # Ignore errors reading response
        raise RuntimeError(f"Unexpected SPARQL execution error: {err_context}") from e


# --- MCP-like Endpoints ---

@app.get("/discover", response_model=DiscoveryResponse, summary="Discover available services")
async def discover_services():
    """Provides information about the services hosted by this server."""
    return {
        "services": [
            {
                "id": "knowledge-graph-lookup",
                "description": "Searches Wikidata or DBpedia for candidates matching a term."
            },
            {
                "id": "sparql-execution",
                "description": "Executes a given SPARQL query against a specified endpoint."
            }
        ]
    }

# KG Lookup Endpoint (Unchanged from previous step)
@app.post("/invoke/knowledge-graph-lookup", response_model=KGLookupResponse, summary="Invoke the KG Lookup Service")
async def invoke_kg_lookup(request: KGLookupRequest):
    """Handles requests to the knowledge-graph-lookup service."""
    logger.info(f"Received invoke request for term='{request.term}', source='{request.source_kg}'")
    try:
        if request.source_kg == 'wikidata':
            candidates = search_wikidata_candidates(request.term, request.max_results)
        elif request.source_kg == 'dbpedia':
            candidates = search_dbpedia_candidates(request.term, request.max_results, verify_ssl=request.dbpedia_verify_ssl)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported source_kg: {request.source_kg}")
        return {"candidates": candidates}
    except (ConnectionError, ConnectionAbortedError) as e:
        logger.error(f"Tool connection error for '{request.term}' in {request.source_kg}: {e}")
        raise HTTPException(status_code=503, detail=f"Service Unavailable: Upstream KG API request failed - {e}")
    except ValueError as e:
        logger.error(f"Tool data error for '{request.term}' in {request.source_kg}: {e}")
        raise HTTPException(status_code=502, detail=f"Bad Gateway: Invalid response from upstream KG API - {e}")
    except RuntimeError as e:
         logger.error(f"Tool runtime error for '{request.term}' in {request.source_kg}: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Internal Server Error: Tool failed unexpectedly - {e}")
    except Exception as e:
         logger.error(f"Unexpected server error processing KG lookup request for '{request.term}': {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


# NEW SPARQL Execution Endpoint
@app.post("/invoke/sparql-execution", response_model=SPARQLExecutionResponse, summary="Invoke the SPARQL Execution Service")
async def invoke_sparql_exec(request: SPARQLExecutionRequest):
    """Handles requests to the sparql-execution service."""
    logger.info(f"Received invoke request for SPARQL execution on {request.endpoint_url}")
    # Use the same user agent defined in kg_tools for consistency
    from kg_tools import USER_AGENT as KG_USER_AGENT
    try:
        results = execute_sparql_tool(str(request.endpoint_url), request.query, KG_USER_AGENT)
        return {"results": results}
    # Handle specific errors from the tool function
    except ValueError as e: # Bad query format, bad JSON response
        logger.error(f"SPARQL execution tool ValueError: {e}")
        raise HTTPException(status_code=400, detail=f"Bad Request: {e}")
    except ConnectionError as e: # Endpoint not found
        logger.error(f"SPARQL execution tool ConnectionError: {e}")
        raise HTTPException(status_code=404, detail=f"Endpoint Not Found: {e}")
    except ConnectionAbortedError as e: # Endpoint internal error
        logger.error(f"SPARQL execution tool ConnectionAbortedError: {e}")
        raise HTTPException(status_code=502, detail=f"Bad Gateway: Endpoint internal error - {e}")
    except RuntimeError as e: # Other tool errors
        logger.error(f"SPARQL execution tool RuntimeError: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: Tool failed - {e}")
    except Exception as e:
        logger.error(f"Unexpected server error processing SPARQL execution request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@app.get("/health", summary="Health Check")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}

# --- Run (Unchanged) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting MCP Server (KG Lookup + SPARQL Exec) on http://localhost:8100")
    uvicorn.run("server:app", host="0.0.0.0", port=8100, reload=True)
