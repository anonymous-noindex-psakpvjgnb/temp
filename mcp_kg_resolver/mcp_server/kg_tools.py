import requests
import json
import logging
from typing import List, Dict, Optional, Any

# Assuming logging is configured in server.py or use module-level logger
logger = logging.getLogger("kg_tools")

# --- Configuration (can be moved to server config if needed) ---
WIKIDATA_API_ENDPOINT: str = "https://www.wikidata.org/w/api.php"
DBPEDIA_LOOKUP_ENDPOINT: str = "https://lookup.dbpedia.org/api/search" # Use HTTPS
USER_AGENT: str = "AdvancedResolverAgent/1.0/KGTool (via MCP)"
REQUEST_TIMEOUT: int = 15
# CANDIDATE_LIMIT is now passed as argument

# Type alias for candidate structure - Ensure consistency with Host
CandidateInfo = Dict[str, Optional[str]] # e.g., {"id": "Q1", "label": "X", "description": "Y", "uri": "Z"}


# --- Tool Implementations ---

def search_wikidata_candidates(term: str, limit: int) -> List[CandidateInfo]:
    """Tool: Searches Wikidata, returning multiple candidates."""
    logger.info(f"Tool: Querying Wikidata for '{term}' (Limit: {limit})")
    candidates = []
    params = {
        "action": "wbsearchentities", "format": "json", "language": "en",
        "uselang": "en", "search": term, "limit": limit, "origin": "*"
    }
    headers = {'User-Agent': USER_AGENT}

    try:
        # Direct request - server context handles overall availability
        response = requests.get(WIKIDATA_API_ENDPOINT, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status() # Raise HTTPError for bad responses
        data = response.json()
        search_results = data.get("search", [])

        for match in search_results:
            entity_id = match.get("id")
            label = match.get("label")
            description = match.get("description")
            concept_uri = match.get("concepturi")
            if entity_id and label:
                candidates.append({
                    "id": entity_id, "label": label,
                    "description": description or "No description",
                    "uri": concept_uri
                })
        logger.info(f"Tool: Wikidata found {len(candidates)} candidates for '{term}'.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Tool: Wikidata RequestException for '{term}': {e}")
        raise ConnectionError(f"Wikidata API request failed: {e}") from e
    except json.JSONDecodeError as e:
        logger.error(f"Tool: Wikidata JSONDecodeError for '{term}': {e}")
        raise ValueError(f"Wikidata API response was not valid JSON: {e}") from e
    except Exception as e:
        logger.error(f"Tool: Wikidata unexpected error for '{term}': {e}", exc_info=True)
        raise RuntimeError(f"Wikidata tool unexpected error: {e}") from e

    return candidates


def search_dbpedia_candidates(term: str, limit: int, verify_ssl: bool = True) -> List[CandidateInfo]:
    """Tool: Searches DBpedia Lookup, returning multiple candidates."""
    logger.info(f"Tool: Querying DBpedia for '{term}' (Limit: {limit}, VerifySSL: {verify_ssl})")
    candidates = []
    params = {"query": term, "format": "json", "maxResults": limit}
    headers = {'User-Agent': USER_AGENT, 'Accept': 'application/json'}

    try:
        response = requests.get(DBPEDIA_LOOKUP_ENDPOINT, params=params, headers=headers, timeout=REQUEST_TIMEOUT, verify=verify_ssl)
        response.raise_for_status()
        data = response.json()
        docs = data.get("docs", [])

        for match in docs:
            uri_list = match.get("resource", [])
            uri = uri_list[0] if uri_list else None
            label_list = match.get("label", [])
            label = label_list[0] if label_list else None
            comment_list = match.get("comment", [])
            description = comment_list[0] if comment_list else "No description"
            if uri and label:
                candidates.append({
                    "id": None, "label": label,
                    "description": description, "uri": uri
                })
        logger.info(f"Tool: DBpedia found {len(candidates)} candidates for '{term}'.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Tool: DBpedia RequestException for '{term}': {e}")
        # Check specifically for SSLError to give a hint
        if isinstance(e, requests.exceptions.SSLError):
             logger.error("Tool: DBpedia encountered SSL Error. Consider setting verify_ssl=False if appropriate for your environment.")
             raise ConnectionAbortedError(f"DBpedia SSL verification failed: {e}") from e
        raise ConnectionError(f"DBpedia API request failed: {e}") from e
    except json.JSONDecodeError as e:
        logger.error(f"Tool: DBpedia JSONDecodeError for '{term}': {e}")
        raise ValueError(f"DBpedia API response was not valid JSON: {e}") from e
    except Exception as e:
        logger.error(f"Tool: DBpedia unexpected error for '{term}': {e}", exc_info=True)
        raise RuntimeError(f"DBpedia tool unexpected error: {e}") from e

    return candidates