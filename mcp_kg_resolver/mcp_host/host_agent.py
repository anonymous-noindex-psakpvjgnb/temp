import requests
import json
import logging
import sys
import time
from typing import List, Dict, Optional, Any

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MCPHostAgent")

# --- Configuration ---
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "deepseek-llm:latest"

# URL of the locally running MCP Server
MCP_SERVER_URL: str = "http://localhost:8100" # Ensure this matches the server's port
KG_LOOKUP_INVOKE_URL = f"{MCP_SERVER_URL}/invoke/knowledge-graph-lookup"

# Agent Configuration
CANDIDATE_LIMIT: int = 5
REQUEST_TIMEOUT: int = 20 # Timeout for calling the MCP server itself
DBPEDIA_VERIFY_SSL: bool = True # Set to False if needed, based on previous SSLError troubleshooting

# Type alias (ensure matches server/tools definition)
CandidateInfo = Dict[str, Optional[str]]

# --- Initialize LLM (Same as before) ---
try:
    from langchain_community.llms import Ollama
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    logger.info(f"Host: Successfully connected to Ollama model: {OLLAMA_MODEL}")
except ImportError:
    logger.critical("Host: LangChain/Ollama libraries not found.")
    sys.exit(1)
except Exception as e:
    logger.critical(f"Host: Connecting to Ollama failed. Details: {e}")
    sys.exit(1)


# --- MCP Client Function ---
def call_mcp_lookup_service(term: str, source_kg: str, max_results: int, dbpedia_verify_ssl: bool = DBPEDIA_VERIFY_SSL) -> List[CandidateInfo]:
    """Calls the MCP server's knowledge-graph-lookup service."""
    logger.debug(f"Calling MCP Server for term='{term}', source='{source_kg}'")
    payload = {
        "term": term,
        "source_kg": source_kg,
        "max_results": max_results,
        "dbpedia_verify_ssl": dbpedia_verify_ssl # Pass the SSL flag
    }
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

    try:
        response = requests.post(KG_LOOKUP_INVOKE_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status() # Check for HTTP errors from MCP server (4xx, 5xx)
        data = response.json()
        candidates = data.get("candidates", [])
        logger.debug(f"MCP Server returned {len(candidates)} candidates for '{term}' from {source_kg}.")
        # Add basic validation if needed: check if list items match CandidateInfo structure
        return candidates
    except requests.exceptions.Timeout:
         logger.error(f"MCP Client: Timeout calling MCP server at {KG_LOOKUP_INVOKE_URL}")
         return []
    except requests.exceptions.ConnectionError:
         logger.error(f"MCP Client: Connection Error calling MCP server at {KG_LOOKUP_INVOKE_URL}. Is it running?")
         return []
    except requests.exceptions.HTTPError as e:
         logger.error(f"MCP Client: HTTP Error calling MCP server: {e.response.status_code} {e.response.reason}. Response: {e.response.text[:200]}...")
         return []
    except requests.exceptions.RequestException as e:
        logger.error(f"MCP Client: Error calling MCP server: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"MCP Client: Failed to decode JSON response from MCP server. Error: {e}. Response: {response.text[:200]}...")
        return []
    except Exception as e:
        logger.error(f"MCP Client: Unexpected error calling MCP service: {e}", exc_info=True)
        return []

# --- LLM Functions (Extraction & Disambiguation - Same as advanced agent before) ---

def extract_focused_terms_llm(natural_query: str) -> List[str]:
    """FEATURE 3: Uses LLM for focused extraction."""
    # ... (Keep the implementation from the previous advanced agent) ...
    if not llm: logger.error("LLM not available..."); return []
    logger.info(f"Host Step 1: LLM Extracting *focused* terms from: '{natural_query}'")
    prompt_template = PromptTemplate.from_template(
        "Analyze the following question. Identify the key semantic components that likely correspond to specific entities (like people, places, organizations, specific concepts with proper names) or properties/relationships (like 'population of', 'capital of', 'director of', 'member of', 'instance of') in a knowledge graph (like Wikidata or DBpedia).\n"
        "Focus on terms crucial for forming a query. Avoid generic verbs, articles, pronouns, and common nouns unless they are part of a well-known concept (e.g., 'boiling point', 'prime minister').\n"
        "List the extracted phrases precisely as they appear or reasonably inferred, separated ONLY by commas. Do not add explanations.\n\n"
        "Question: {query}\n\n"
        "Focused Entities/Properties:"
    )
    chain = prompt_template | llm | StrOutputParser()
    try:
        result = chain.invoke({"query": natural_query})
        if not result or not isinstance(result, str): return []
        terms = [term.strip() for term in result.split(',') if term.strip()]
        terms = [t for t in terms if len(t) > 1]
        logger.info(f"   LLM Focused Extracted: {terms}")
        return terms
    except Exception as e:
        logger.error(f"   Error during LLM focused term extraction: {e}", exc_info=True)
        return []


def disambiguate_term_llm(nlq: str, term: str, candidates: List[CandidateInfo], source_kg: str) -> Optional[CandidateInfo]:
    """FEATURE 1 & 2 Core: Uses LLM to choose the best candidate."""
    # ... (Keep the implementation from the previous advanced agent) ...
    if not candidates: return None
    if len(candidates) == 1: return candidates[0]
    if not llm: logger.error(f"LLM not available for disambiguation of '{term}'."); return None

    logger.info(f"   LLM Disambiguating '{term}' using {len(candidates)} candidates from {source_kg}...")
    candidate_text = ""
    for i, cand in enumerate(candidates):
        ident = cand.get('id') or cand.get('uri')
        label = cand.get('label')
        desc = cand.get('description')
        candidate_text += f"{i+1}. Identifier: {ident}\n   Label: {label}\n   Description: {desc}\n\n"

    prompt_template = PromptTemplate.from_template(
        "You need to select the most relevant entity or property from the list below that best matches the term '{term}' within the context of the following natural language query:\n"
        "Natural Language Query: \"{nlq}\"\n\n"
        "Term to Disambiguate: \"{term}\"\n\n"
        "Candidates from {source_kg}:\n{candidate_list}\n"
        "Consider the meaning and context of the query. For example, if the query asks about 'President of India', choose the candidate representing the head of state, not a corporate title. If the query asks about a relation like 'capital of', choose the property candidate that best represents that relationship.\n\n"
        "Which candidate number is the most appropriate match? Respond ONLY with the number (e.g., '1', '3'). If no candidate seems appropriate, respond with '0'."
    )
    chain = prompt_template | llm | StrOutputParser()
    try:
        response = chain.invoke({
            "nlq": nlq, "term": term,
            "source_kg": source_kg, "candidate_list": candidate_text
        })
        choice_str = ''.join(filter(str.isdigit, response.strip()))
        if choice_str.isdigit():
            choice_index = int(choice_str) - 1
            if 0 <= choice_index < len(candidates):
                selected_candidate = candidates[choice_index]
                logger.info(f"      LLM selected candidate {choice_index + 1} for '{term}'.")
                return selected_candidate
            elif choice_index == -1:
                 logger.info(f"      LLM indicated no suitable candidate for '{term}'.")
                 return None
            else: logger.warning(f"LLM chose invalid number {choice_str} for '{term}'."); return None
        else: logger.warning(f"LLM response for '{term}' was not a clear number: '{response}'."); return None
    except Exception as e:
        logger.error(f"Error during LLM disambiguation for '{term}': {e}", exc_info=True); return None


# --- Main Agent Function (Modified Workflow) ---

def resolve_terms_agent_mcp(natural_query: str) -> Dict[str, Dict[str, Optional[CandidateInfo]]]:
    """Advanced agent using MCP server for KG lookups."""
    logger.info(f"--- Starting MCP Host Agent for query: '{natural_query}' ---")

    # 1. Focused term extraction (Local LLM call)
    extracted_terms = extract_focused_terms_llm(natural_query)
    if not extracted_terms:
        logger.warning("Host Agent: No terms extracted. Cannot proceed.")
        return {}

    # 2. & 3. Get candidates via MCP and disambiguate locally
    final_resolution: Dict[str, Dict[str, Optional[CandidateInfo]]] = {}
    logger.info("Step 2 & 3: Getting candidates via MCP and disambiguating terms...")

    for term in extracted_terms:
        logger.info(f"--- Processing term: '{term}' ---")
        term_results: Dict[str, Optional[CandidateInfo]] = {"wikidata": None, "dbpedia": None}

        # Wikidata Resolution Path (via MCP Server)
        logger.info(f"   Calling MCP for Wikidata candidates of '{term}'...")
        wikidata_candidates = call_mcp_lookup_service(term, "wikidata", CANDIDATE_LIMIT)
        if wikidata_candidates:
            # Disambiguation uses Local LLM
            selected_wikidata = disambiguate_term_llm(natural_query, term, wikidata_candidates, "Wikidata")
            term_results["wikidata"] = selected_wikidata
        else:
             logger.info(f"   No Wikidata candidates received from MCP Server for '{term}'.")

        # DBpedia Resolution Path (via MCP Server)
        logger.info(f"   Calling MCP for DBpedia candidates of '{term}'...")
        dbpedia_candidates = call_mcp_lookup_service(term, "dbpedia", CANDIDATE_LIMIT) # SSL handled by server call
        if dbpedia_candidates:
            # Disambiguation uses Local LLM
            selected_dbpedia = disambiguate_term_llm(natural_query, term, dbpedia_candidates, "DBpedia")
            term_results["dbpedia"] = selected_dbpedia
        else:
             logger.info(f"   No DBpedia candidates received from MCP Server for '{term}'.")

        final_resolution[term] = term_results

    logger.info("--- MCP Host Agent Finished ---")
    return final_resolution

# --- Example Usage ---
if __name__ == "__main__":
    # Check if MCP Server is reachable first (optional but good practice)
    try:
        health_url = f"{MCP_SERVER_URL}/health"
        health_check = requests.get(health_url, timeout=5)
        health_check.raise_for_status()
        logger.info(f"MCP Server health check OK: {health_check.json()}")
    except requests.exceptions.RequestException as e:
        logger.error(f"MCP Server at {MCP_SERVER_URL} is unreachable! Please start the server first.")
        logger.error(f"Error details: {e}")
        sys.exit(1)

    # Example Queries
    q1 = "Who is the president of India?"
    q2 = "What is the population of Paris, France?"
    q3 = "Show me action movies directed by Christopher Nolan."
    q4 = "What is the boiling point of water?"

    queries = [q1, q2, q3, q4]
    all_results = {}

    for i, query in enumerate(queries):
        logger.info("\n" + "="*60)
        logger.info(f"Processing Query {i+1}: {query}")
        results = resolve_terms_agent_mcp(query)
        all_results[f"Query {i+1}"] = results
        logger.info(f"\nFinal Resolved Output (Query {i+1}):\n%s", json.dumps(results, indent=2))
        if i < len(queries) - 1:
             time.sleep(1) # Short delay between agent runs

    # Optionally print all results at the end
    # logger.info("\n" + "="*60 + "\nAll Results Summary:\n" + "="*60)
    # logger.info(json.dumps(all_results, indent=2))