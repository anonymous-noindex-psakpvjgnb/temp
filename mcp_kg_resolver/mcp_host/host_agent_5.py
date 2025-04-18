# mcp_host/host_agent.py (Complete V5 Implementation)

import requests
import json
import logging
import sys
import time
import re
import pandas as pd
import gradio as gr
import os # Import os for environment variables
from typing import List, Dict, Optional, Any, Tuple, Literal, TypedDict

# Optional Graphing Imports
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import io
    import base64
    GRAPHING_ENABLED = True
except ImportError:
    GRAPHING_ENABLED = False
    logging.warning("NetworkX or Matplotlib not found. Graph visualization disabled. Install with `pip install networkx matplotlib`")

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)]) # Ensure logs go to stdout
logger = logging.getLogger("MCPHostAgentV5") # V5

# --- Configuration ---
# Load from environment variables or use defaults
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "deepseek-llm:latest") # Ensure model good at JSON output and reasoning

MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "http://localhost:8100")
KG_LOOKUP_INVOKE_URL = f"{MCP_SERVER_URL}/invoke/knowledge-graph-lookup"
SPARQL_EXEC_INVOKE_URL = f"{MCP_SERVER_URL}/invoke/sparql-execution"

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
DBPEDIA_ENDPOINT = "https://dbpedia.org/sparql"
WIKIDATA_API_ENDPOINT = "https://www.wikidata.org/w/api.php" # For direct abstract fetch

# Agent Configuration
CANDIDATE_LIMIT: int = 5
REQUEST_TIMEOUT: int = 30 # Timeout for external calls (MCP server, Wikidata API)
DBPEDIA_VERIFY_SSL: bool = os.getenv("DBPEDIA_VERIFY_SSL", "true").lower() == "true" # Default True, allow override via env var
USER_AGENT: str = "AdvancedResolverAgentV5/Host (YourApp; contact@example.com)"

# --- Type Definitions ---
CandidateInfo = Dict[str, Optional[str]] # e.g., {"id": "Q1", "label": "X", "description": "Y", "uri": "Z"}

class NlqAnalysisResult(TypedDict):
    entities: List[str] # List of entity strings
    target_relation_phrase: Optional[str] # The core relation/attribute phrase

class ResolvedTermData(TypedDict):
    wikidata: Optional[CandidateInfo]
    dbpedia: Optional[CandidateInfo]

ResolvedEntitiesMap = Dict[str, ResolvedTermData] # Maps entity term string to resolved data
ResolvedPropertyInfo = ResolvedTermData # Reuse structure for the single resolved property


# --- Initialize LLM ---
try:
    from langchain_community.llms import Ollama
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    llm = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    logger.info(f"Host: Successfully connected to Ollama model: {OLLAMA_MODEL} at {OLLAMA_BASE_URL}")
except ImportError:
    logger.critical("Host: LangChain/Ollama libraries not found. Please install them (`pip install langchain_community ollama`)")
    sys.exit(1)
except Exception as e:
    logger.critical(f"Host: Connecting to Ollama failed. Is Ollama running and model '{OLLAMA_MODEL}' pulled? Details: {e}")
    sys.exit(1)


# --- MCP Client Functions ---
def call_mcp_service(url: str, payload: Dict, context: str) -> Optional[Dict[str, Any]]:
    """Generic function to call an MCP service endpoint with robust error handling."""
    logger.debug(f"Calling MCP Service ({context}) at {url}")
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    response_data = None
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        # Check explicit status codes for better error propagation
        if response.status_code == 200:
            response_data = response.json()
            logger.debug(f"MCP Service ({context}) successful.")
        elif 400 <= response.status_code < 500:
             error_detail = f"Client Error {response.status_code}"
             try: server_error = response.json(); error_detail += f": {server_error.get('detail', response.reason)}"
             except: error_detail += f": {response.reason}" # Fallback if response isn't JSON
             logger.error(f"MCP Client: {context} failed: {error_detail}")
             response_data = {"error": error_detail} # Propagate structured error
        elif response.status_code >= 500:
             error_detail = f"Server Error {response.status_code}"
             try: server_error = response.json(); error_detail += f": {server_error.get('detail', response.reason)}"
             except: error_detail += f": {response.reason}"
             logger.error(f"MCP Client: {context} failed: {error_detail}")
             response_data = {"error": error_detail}
        else:
            # Raise for other unexpected statuses
            response.raise_for_status()
    except requests.exceptions.Timeout:
         logger.error(f"MCP Client: Timeout calling {context} service at {url}")
         response_data = {"error": f"Timeout calling MCP service ({context})."}
    except requests.exceptions.ConnectionError as e:
         logger.error(f"MCP Client: Connection Error calling {context} service at {url}: {e}")
         response_data = {"error": f"MCP connection error ({context}): {e}"}
    except requests.exceptions.RequestException as e:
        logger.error(f"MCP Client: RequestException calling {context} service at {url}: {e}")
        response_data = {"error": f"MCP request error ({context}): {e}"}
    except json.JSONDecodeError as e:
        # Log the problematic text if possible (be careful with large responses)
        resp_text = ""
        if 'response' in locals() and hasattr(response, 'text'):
            resp_text = response.text[:500] # Log beginning of text
        logger.error(f"MCP Client: Failed to decode JSON response from {context} service. Error: {e}. Response snippet: {resp_text}...")
        response_data = {"error": f"Invalid JSON response from MCP service ({context})."}
    except Exception as e:
        logger.error(f"MCP Client: Unexpected error calling {context} service: {e}", exc_info=True)
        response_data = {"error": f"Unexpected client error calling MCP ({context})."}

    return response_data


def call_mcp_lookup_service(term: str, source_kg: str, max_results: int) -> List[CandidateInfo]:
    """Calls the MCP server's knowledge-graph-lookup service."""
    payload = {
        "term": term,
        "source_kg": source_kg,
        "max_results": max_results,
        "dbpedia_verify_ssl": DBPEDIA_VERIFY_SSL
    }
    response_data = call_mcp_service(KG_LOOKUP_INVOKE_URL, payload, f"KG Lookup ({source_kg} for '{term}')")
    # Check if response is valid and doesn't contain an error key
    if response_data and isinstance(response_data, dict) and not response_data.get("error"):
        candidates = response_data.get("candidates", [])
        if isinstance(candidates, list): # Ensure it's a list
             return candidates
        else:
             logger.warning(f"MCP Lookup response for '{term}' contained 'candidates' but it wasn't a list.")
             return []
    # Log error if present
    if response_data and isinstance(response_data, dict) and response_data.get("error"):
        logger.error(f"MCP Lookup service returned error for '{term}': {response_data['error']}")
    return [] # Return empty list on error or invalid response


def call_mcp_sparql_execution(endpoint_url: str, query: str, context: str = "SPARQL Execution") -> Optional[Dict[str, Any]]:
    """Calls the MCP server's SPARQL execution service."""
    payload = {"endpoint_url": endpoint_url, "query": query}
    # Returns the full response dict, which might contain 'results' or 'error' key
    return call_mcp_service(SPARQL_EXEC_INVOKE_URL, payload, context)


# --- LLM Functions ---

def analyze_nlq_structure_llm(nlq: str) -> Optional[NlqAnalysisResult]:
    """V5 Core: Analyzes NLQ for entities and target relation phrase."""
    if not llm: logger.error("LLM not available for NLQ Analysis."); return None
    logger.info(f"Step 1: LLM Analyzing NLQ structure: '{nlq}'")
    prompt = PromptTemplate(
        template=(
            "Analyze the user's question. Identify:\n"
            "1. The main specific entities involved (people, places, organizations, named concepts).\n"
            "2. The single, core relationship or attribute being asked about (e.g., 'president', 'spouse', 'capital', 'population', 'director', 'boiling point', 'chemical formula'). If the question asks for a general description (e.g., 'Tell me about Berlin'), output null for the relation.\n\n"
            "Output ONLY a valid JSON object with exactly two keys:\n"
            "- \"entities\": A JSON list of the entity strings found.\n"
            "- \"target_relation_phrase\": A JSON string representing the core relationship/attribute, or null if none is clearly identified.\n\n"
            "Example Question: 'Who is the wife of Barack Obama?'\n"
            "Example Output: {{\"entities\": [\"Barack Obama\"], \"target_relation_phrase\": \"wife\"}}\n\n"
            "Example Question: 'What is the population of Berlin?'\n"
            "Example Output: {{\"entities\": [\"Berlin\"], \"target_relation_phrase\": \"population\"}}\n\n"
            "Example Question: 'Describe the Eiffel Tower.'\n"
            "Example Output: {{\"entities\": [\"Eiffel Tower\"], \"target_relation_phrase\": null}}\n\n"
            "User Question: \"{query}\"\n\n"
            "JSON Output:"
        ), input_variables=["query"],
    )
    chain = prompt | llm | StrOutputParser()
    try:
        response_str = chain.invoke({"query": nlq})
        logger.debug(f"LLM Raw NLQ Analysis Output: {response_str}")
        # Robust JSON extraction and parsing
        match = re.search(r'\{.*\}', response_str, re.DOTALL)
        if not match: logger.error(f"Could not find JSON object in LLM NLQ analysis response: {response_str}"); return None
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            # Basic Validation
            if isinstance(data, dict) and 'entities' in data and isinstance(data['entities'], list) and 'target_relation_phrase' in data:
                entities_list = [str(e).strip() for e in data['entities'] if isinstance(e, (str, int, float)) and str(e).strip()] # Ensure non-empty strings
                relation_phrase = data['target_relation_phrase']
                result: NlqAnalysisResult = {
                    "entities": entities_list,
                    "target_relation_phrase": str(relation_phrase).strip() if isinstance(relation_phrase, str) and str(relation_phrase).strip() else None # Ensure null if empty/not string
                }
                logger.info(f"   LLM NLQ Analysis Result: {result}")
                return result
            else: logger.error(f"LLM NLQ analysis output JSON lacks required structure: {data}"); return None
        except json.JSONDecodeError as json_err: logger.error(f"Failed parsing LLM NLQ analysis JSON: {json_err}. Response: {response_str}"); return None
    except Exception as e: logger.error(f"   Error during LLM NLQ analysis: {e}", exc_info=True); return None


def find_wikidata_properties(term_phrase: str, limit: int) -> List[CandidateInfo]:
    """Searches Wikidata specifically for properties matching the phrase."""
    logger.info(f"   Querying Wikidata for PROPERTY candidates of: '{term_phrase}'")
    candidates = []
    params = { "action": "wbsearchentities", "format": "json", "language": "en", "uselang": "en", "search": term_phrase, "limit": limit, "type": "property", "origin": "*" }
    headers = {'User-Agent': USER_AGENT}
    try:
        # Direct API Call (Simpler than MCP for this specific internal tool)
        response = requests.get(WIKIDATA_API_ENDPOINT, params=params, headers=headers, timeout=REQUEST_TIMEOUT); response.raise_for_status(); data = response.json(); search_results = data.get("search", [])
        for match in search_results:
            if match.get("id") and match["id"].startswith("P"): # Ensure it's a property
                candidates.append({"id": match.get("id"), "label": match.get("label"), "description": match.get("description") or "No description", "uri": match.get("concepturi")})
        logger.info(f"      Wikidata Property Search: Found {len(candidates)} candidates for '{term_phrase}'.")
    except Exception as e: logger.error(f"      Wikidata Property Search error for '{term_phrase}': {e}")
    return candidates


def find_dbpedia_properties(term_phrase: str, limit: int) -> List[CandidateInfo]:
    """Searches DBpedia for properties matching the phrase using SPARQL via MCP."""
    logger.info(f"   Querying DBpedia for PROPERTY candidates of: '{term_phrase}' via SPARQL")
    candidates = []
    escaped_term = re.sub(r'[-\/\\^$*+?.()|[\]{}]', r'\\&', term_phrase) # Basic escaping for regex
    # Look for both rdf:Property and specific ontology properties
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    SELECT DISTINCT ?property ?label ?comment WHERE {{
      {{ ?property a rdf:Property }} UNION {{ ?property a owl:ObjectProperty }} UNION {{ ?property a owl:DatatypeProperty }}
      ?property rdfs:label ?label .
      OPTIONAL {{ ?property rdfs:comment ?comment . FILTER(LANG(?comment) = 'en') }}
      FILTER(REGEX(STR(?label), "{escaped_term}", "i"))
      FILTER(LANG(?label) = 'en')
    }} LIMIT {limit}
    """
    results_dict = call_mcp_sparql_execution(DBPEDIA_ENDPOINT, query, context="DBpedia Property Search")

    if results_dict and not results_dict.get("error"):
        bindings = results_dict.get("results", {}).get("bindings", [])
        for binding in bindings:
            uri = binding.get("property", {}).get("value"); label = binding.get("label", {}).get("value"); desc = binding.get("comment", {}).get("value", "No description")
            if uri and label: candidates.append({"id": None, "label": label, "description": desc, "uri": uri})
        logger.info(f"      DBpedia Property Search: Found {len(candidates)} candidates for '{term_phrase}'.")
    else:
        err_msg = results_dict.get('error', 'Unknown') if isinstance(results_dict, dict) else 'No response or invalid format'
        logger.warning(f"      DBpedia Property Search failed for '{term_phrase}'. Error: {err_msg}")
    return candidates


def disambiguate_term_llm(nlq: str, term_being_resolved: str, is_entity: bool, candidates: List[CandidateInfo], source_kg: str, target_relation_phrase: Optional[str] = None, resolved_entities: Optional[ResolvedEntitiesMap] = None) -> Optional[CandidateInfo]:
    """Uses LLM for disambiguation, using relation for entities, entities for relations."""
    if not candidates: return None
    if len(candidates) == 1: logger.info(f"   Only one candidate for '{term_being_resolved}', selecting directly."); return candidates[0]
    if not llm: logger.error(f"LLM not available for disambiguation of '{term_being_resolved}'."); return None

    logger.info(f"   LLM Disambiguating {'ENTITY' if is_entity else 'PROPERTY'} '{term_being_resolved}' for query '{nlq}'...")

    # Prepare candidate list text
    candidate_text = ""
    for i, cand in enumerate(candidates):
        ident = cand.get('id') or cand.get('uri')
        label = cand.get('label'); desc = cand.get('description')
        candidate_text += f"{i+1}. Identifier: {ident}\n   Label: {label}\n   Description: {desc}\n\n"

    # Prepare specific context based on what's being resolved
    extra_context = ""
    if is_entity and target_relation_phrase:
        extra_context = f"Context: The user is asking about the relation/attribute '{target_relation_phrase}' involving this entity.\nChoose the entity sense that fits this relationship.\n"
    elif not is_entity and resolved_entities: # Resolving property phrase
        extra_context += "Context Entities (choose property that best connects/describes these):\n"
        entity_count = 0
        for entity_term, entity_data in resolved_entities.items():
             # Show resolved info from the *same* KG if possible for context
             entity_kg_info = entity_data.get(source_kg.lower())
             if entity_kg_info:
                 ident = entity_kg_info.get('id') or entity_kg_info.get('uri')
                 label = entity_kg_info.get('label')
                 extra_context += f"- '{entity_term}': {label} ({ident})\n"
                 entity_count += 1
        if entity_count == 0: extra_context = "" # Clear if no relevant entities found

    # Disambiguation Prompt
    prompt_template = PromptTemplate(
        template=(
            "Select the most relevant item from the list below for the term '{term}' based on the user's query.\n"
            "Query: \"{nlq}\"\n"
            "Term to Disambiguate: \"{term}\"\n\n"
            "{extra_context}" # Context based on whether it's entity or property
            "Candidates from {source_kg}:\n{candidate_list}\n"
            "Instructions: Consider the query and context. Choose the best fit.\n\n"
            "Which candidate number is most appropriate? Respond ONLY with the number (e.g., '1', '3'). If none fit, respond with '0'."
        ), input_variables=["nlq", "term", "extra_context", "source_kg", "candidate_list"]
    )
    chain = prompt_template | llm | StrOutputParser()
    try:
        response = chain.invoke({ "nlq": nlq, "term": term_being_resolved, "extra_context": extra_context, "source_kg": source_kg, "candidate_list": candidate_text })
        # Robust number parsing
        response_text = response.strip(); choice_str = None; match = re.search(r'\b(\d+)\b', response_text)
        if match: choice_str = match.group(1); logger.debug(f"Disambiguation: Found number '{choice_str}' via regex.")
        else: digits = ''.join(filter(str.isdigit, response_text));
        if digits: choice_str = digits[0]; logger.debug(f"Disambiguation: Found digits '{digits}', taking first '{choice_str}' (fallback).")

        if choice_str and choice_str.isdigit():
            choice_index = int(choice_str) - 1
            if 0 <= choice_index < len(candidates): logger.info(f"LLM selected candidate {choice_index + 1}."); return candidates[choice_index]
            elif choice_index == -1: logger.info("LLM indicated no suitable candidate (chose 0)."); return None
            else: logger.warning(f"LLM chose out-of-bounds number {choice_str}. Max index: {len(candidates)-1}."); return None
        else: logger.warning(f"Failed to parse number from LLM disambiguation: '{response_text}'."); return None
    except Exception as e: logger.error(f"Error during LLM disambiguation for '{term_being_resolved}': {e}", exc_info=True); return None


def generate_sparql_queries_llm_v5(nlq: str, resolved_entities: ResolvedEntitiesMap, resolved_property: Optional[ResolvedPropertyInfo]) -> Tuple[Optional[str], Optional[str]]:
    """Generates SPARQL using resolved entities and the single target property."""
    if not llm: logger.error("LLM not available for SPARQL generation."); return None, None
    logger.info("Step 4: LLM Generating SPARQL queries (V5 - Direct Property)...")

    # Format context
    entity_context = "Entities:\n"
    if not resolved_entities: entity_context += "  (None resolved)\n"
    else:
        for term, data in resolved_entities.items():
            entity_context += f"- '{term}':\n"
            wd_info = data.get('wikidata'); db_info = data.get('dbpedia')
            if wd_info: entity_context += f"  - WD: {wd_info.get('label')} (wd:{wd_info.get('id')})\n"
            if db_info: entity_context += f"  - DB: {db_info.get('label')} (<{db_info.get('uri')}>)\n"

    prop_context = "Target Relation/Attribute:\n"
    if not resolved_property: prop_context += "  (None identified or resolved)\n"
    else:
        wd_info = resolved_property.get('wikidata'); db_info = resolved_property.get('dbpedia')
        prop_context += "- Used As:\n" # Indicate which KG version to use based on resolution
        if wd_info: prop_context += f"  - WD: {wd_info.get('label')} (wdt:{wd_info.get('id')})\n" # Assume property, use wdt:
        if db_info: prop_context += f"  - DB: {db_info.get('label')} (<{db_info.get('uri')}>)\n"

    # Simplified SPARQL Generation Prompt
    prompt_template = PromptTemplate.from_template(
        "Generate TWO SPARQL queries (Wikidata, DBpedia) to answer the query using the provided entities and target relation.\n\n"
        "Natural Language Query: {nlq}\n\n"
        "Context:\n{entity_context}{prop_context}\n"
        "Instructions:\n"
        "1. Construct queries connecting the identified entities using the identified target relation/attribute. The variable to select should correspond to what the user is asking for (e.g., if asking 'Who is X of Y?', select the 'who').\n"
        "2. If the target relation is missing, try to formulate a query to get a description or basic facts about the main entity.\n"
        "3. For Wikidata: Use `wd:`/`wdt:` prefixes accurately. Include labels for results using `?varLabel` and the `SERVICE wikibase:label` block.\n"
        "4. For DBpedia: Use full URIs. Include `rdfs:label` filtered by `LANG='en'`.\n"
        "5. Output ONLY the two queries separated by `--- WIKIDATA ---` and `--- DBPEDIA ---` in ```sparql blocks.\n\n"
        "--- WIKIDATA ---\n```sparql\n# Wikidata Query Here\n```\n\n"
        "--- DBPEDIA ---\n```sparql\n# DBpedia Query Here\n```\n"
    )
    chain = prompt_template | llm | StrOutputParser()
    try:
        response = chain.invoke({ "nlq": nlq, "entity_context": entity_context, "prop_context": prop_context })
        # Regex parsing
        wd_query, db_query = None, None
        wd_match = re.search(r"--- WIKIDATA ---\s*```sparql\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        db_match = re.search(r"--- DBPEDIA ---\s*```sparql\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if wd_match: wd_query = wd_match.group(1).strip(); logger.info("   LLM Generated Wikidata Query (V5).")
        if db_match: db_query = db_match.group(1).strip(); logger.info("   LLM Generated DBpedia Query (V5).")
        if not wd_query and not db_query: logger.warning(f"LLM failed SPARQL generation format.")
        elif not wd_query: logger.warning("LLM failed WD SPARQL generation.")
        elif not db_query: logger.warning("LLM failed DB SPARQL generation.")
        return wd_query, db_query
    except Exception as e: logger.error(f"Error during LLM SPARQL generation V5: {e}", exc_info=True); return None, None


# --- Fallback Functions ---
def fetch_wikidata_abstract(qid: str) -> Optional[str]:
    """Fetches abstract/description directly from Wikidata API."""
    logger.info(f"Fallback: Fetching Wikidata description for {qid}")
    params = {"action": "wbgetentities", "format": "json", "ids": qid, "props": "descriptions", "languages": "en"}
    headers = {'User-Agent': USER_AGENT}
    try:
        response = requests.get(WIKIDATA_API_ENDPOINT, params=params, headers=headers, timeout=REQUEST_TIMEOUT); response.raise_for_status(); data = response.json()
        description = data.get("entities", {}).get(qid, {}).get("descriptions", {}).get("en", {}).get("value")
        if description: logger.info(f"   Found Wikidata description for {qid}."); return description
        else: logger.info(f"   No English Wikidata description found for {qid}."); return None
    except Exception as e: logger.error(f"   Error fetching Wikidata description for {qid}: {e}"); return None

def fetch_dbpedia_abstract(uri: str) -> Optional[str]:
    """Fetches abstract from DBpedia using SPARQL via MCP server."""
    logger.info(f"Fallback: Fetching DBpedia abstract for <{uri}>")
    # Ensure URI has angle brackets if not already present for SPARQL
    sparql_uri = uri if uri.startswith('<') and uri.endswith('>') else f"<{uri}>"
    query = f"""
    PREFIX dbo: [http://dbpedia.org/ontology/](http://dbpedia.org/ontology/)
    SELECT ?abstract WHERE {{
      {sparql_uri} dbo:abstract ?abstract .
      FILTER(LANG(?abstract) = 'en')
    }} LIMIT 1
    """
    results_dict = call_mcp_sparql_execution(DBPEDIA_ENDPOINT, query, context="DBpedia Abstract Fetch")
    if results_dict and not results_dict.get("error"):
        bindings = results_dict.get("results", {}).get("bindings", [])
        if bindings:
            abstract = bindings[0].get("abstract", {}).get("value")
            if abstract: logger.info(f"   Found DBpedia abstract for {sparql_uri}."); return abstract
    logger.info(f"   No English DBpedia abstract found or error fetching for {sparql_uri}.")
    return None

def answer_from_abstracts_llm(nlq: str, wd_abstract: Optional[str], db_abstract: Optional[str]) -> str:
    """Fallback: Uses LLM to answer based on fetched abstracts."""
    if not wd_abstract and not db_abstract: return "Fallback failed: Could not retrieve abstracts."
    if not llm: return "Fallback failed: LLM not available."
    logger.info("Step 6 (Fallback): LLM Answering from abstracts...")
    context = "Answer the original question based *only* on the following abstract(s). If the abstracts don't contain the answer, state that clearly.\n\n"
    if wd_abstract: context += f"Wikidata Abstract/Description:\n{wd_abstract}\n\n"
    if db_abstract: context += f"DBpedia Abstract:\n{db_abstract}\n\n"
    prompt_template = PromptTemplate(template=("Original Question: \"{nlq}\"\n\n{context}Answer based *only* on the provided abstract(s):"), input_variables=["nlq", "context"])
    chain = prompt_template | llm | StrOutputParser()
    try:
        answer = chain.invoke({"nlq": nlq, "context": context})
        logger.info("   LLM generated answer from abstracts.")
        return answer.strip()
    except Exception as e: logger.error(f"Error during LLM fallback answering: {e}", exc_info=True); return "Error: Failed to generate answer from abstracts."


# --- Result Formatting ---
def format_results_for_display(results_dict: Optional[Dict[str, Any]]) -> Tuple[pd.DataFrame, str, Optional[Any]]:
    """Formats SPARQL results into DataFrame, Triples String, and optionally a Graph object."""
    # Initialize default return values
    df_empty = pd.DataFrame()
    triples_str = "N/A"
    graph = None

    if not results_dict: return df_empty, "No results received from execution service.", None
    if results_dict.get("error"): return df_empty, f"Execution Error: {results_dict['error']}", None

    bindings = results_dict.get("results", {}).get("bindings", [])
    if not bindings: return df_empty, "Query executed successfully, but returned no results.", None

    # 1. DataFrame Processing (Robust)
    processed_data = []
    variables = results_dict.get("head", {}).get("vars", [])
    label_vars = {v + "Label" for v in variables}

    for binding in bindings:
        row = {}
        processed_simple_vars = set()
        for var in variables:
            if var in processed_simple_vars: continue
            label_var = f"{var}Label"; value_info = binding.get(var); label_info = binding.get(label_var)
            display_value = None; raw_value = value_info.get("value") if value_info else None
            # Prioritize label
            if label_info and label_info.get("value"): display_value = label_info.get("value"); processed_simple_vars.add(var); processed_simple_vars.add(label_var)
            elif value_info:
                display_value = raw_value
                if value_info.get("type") == "uri": # Cleanup URIs only if no label
                    if "wikidata.org/entity/" in display_value: display_value = f"wd:{display_value.rsplit('/', 1)[-1]}"
                    elif "dbpedia.org/resource/" in display_value: display_value = f"dbr:{display_value.rsplit('/', 1)[-1]}"
                    elif "dbpedia.org/ontology/" in display_value: display_value = f"dbo:{display_value.rsplit('/', 1)[-1]}"
                    elif "dbpedia.org/property/" in display_value: display_value = f"dbp:{display_value.rsplit('/', 1)[-1]}"
                processed_simple_vars.add(var)
            row[var] = display_value
        processed_data.append(row)

    df = pd.DataFrame(processed_data)
    # Reorder columns gracefully
    display_vars = [v for v in variables if v not in label_vars or v+'Label' not in df.columns]
    available_display_vars = [v for v in display_vars if v in df.columns]
    if available_display_vars:
        try: df = df[available_display_vars]
        except KeyError: logger.warning("Could not reorder DataFrame columns perfectly."); pass # Keep original order on error

    # 2. Triples String Extraction (Heuristic)
    if GRAPHING_ENABLED: graph = nx.DiGraph()
    added_triples = set(); triples_list = []
    # Try harder to find subject, predicate, object columns
    subj_var, pred_var, obj_var = None, None, None
    common_subj = ['s', 'item', 'subject', 'entity']
    common_pred = ['p', 'prop', 'property', 'predicate', 'relation']
    common_obj = ['o', 'obj', 'value', 'object']
    vars_lower = {v.lower() for v in df.columns}

    for s in common_subj:
        if s in vars_lower: subj_var = df.columns[list(vars_lower).index(s)]; break
    for p in common_pred:
        if p in vars_lower: pred_var = df.columns[list(vars_lower).index(p)]; break
    for o in common_obj:
         # Avoid picking the same column twice
        if o in vars_lower and df.columns[list(vars_lower).index(o)] != subj_var and df.columns[list(vars_lower).index(o)] != pred_var:
            obj_var = df.columns[list(vars_lower).index(o)]; break
    # Fallback if specific names aren't found
    if not (subj_var and pred_var and obj_var) and len(df.columns) >= 3:
         subj_var, pred_var, obj_var = df.columns[0], df.columns[1], df.columns[2] # Positional guess

    if subj_var and pred_var and obj_var:
        logger.info(f"Attempting triple extraction using ({subj_var}, {pred_var}, {obj_var})")
        for _, row_data in df.iterrows(): # Iterate over DataFrame rows
            s = row_data.get(subj_var); p = row_data.get(pred_var); o = row_data.get(obj_var)
            if s and p and o and pd.notna(s) and pd.notna(p) and pd.notna(o): # Check for non-null/NaN
                s_disp = str(s); p_disp = str(p).split('/')[-1].split('#')[-1]; o_disp = str(o)
                triple = (s_disp, p_disp, o_disp)
                if triple not in added_triples:
                    triples_list.append(f"({s_disp}) --[{p_disp}]--> ({o_disp})"); added_triples.add(triple)
                    if GRAPHING_ENABLED and graph is not None: graph.add_edge(s_disp, o_disp, label=p_disp) # Add edge to graph
        if triples_list: triples_str = f"Found {len(triples_list)} unique triples:\n" + "\n".join(triples_list)
        else: triples_str = "Could not extract triples from results."
    else: triples_str = "Could not identify subject/predicate/object variables for triple extraction."

    return df, triples_str, graph


def render_graph_to_base64(graph) -> Optional[str]:
    """Renders a NetworkX graph to a base64 encoded PNG image string."""
    if not GRAPHING_ENABLED or graph is None or graph.number_of_nodes() == 0: return None
    try:
        plt.figure(figsize=(12, 9)) # Slightly larger figure
        # Use a layout that spaces nodes better, increase repulsion
        pos = nx.spring_layout(graph, k=0.8/max(1, (graph.number_of_nodes()**0.5)), iterations=75, seed=42)

        node_sizes = [max(500, 1000 + graph.degree(n) * 200) for n in graph.nodes()] # Size nodes by degree
        nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color='lightblue', alpha=0.9)
        nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.6, arrowsize=15, connectionstyle='arc3,rad=0.1', node_size=node_sizes) # Adjust node_size for edge endpoint calculation
        nx.draw_networkx_labels(graph, pos, font_size=9)
        edge_labels = nx.get_edge_attributes(graph, 'label')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, label_pos=0.3, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

        plt.title("Result Subgraph")
        plt.axis('off')
        buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); plt.close(); buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8'); return f"data:image/png;base64,{img_base64}"
    except Exception as e: logger.error(f"Failed to render graph: {e}", exc_info=True); return None


# --- Agent 1 Internal Orchestration (REVISED V5) ---
def resolve_terms_agent_v5(natural_query: str, analysis: Optional[NlqAnalysisResult]) -> Tuple[ResolvedEntitiesMap, Optional[ResolvedPropertyInfo]]:
    """Internal V5: Resolves entities and the single target property."""
    resolved_entities: ResolvedEntitiesMap = {}
    resolved_property: Optional[ResolvedPropertyInfo] = None

    if not analysis:
        logger.warning("Agent 1 (V5): NLQ Analysis failed or returned None.")
        return resolved_entities, resolved_property

    entity_terms = analysis['entities']
    target_relation_phrase = analysis['target_relation_phrase']

    logger.info("Agent 1 (V5): Resolving Entities...")
    for term in entity_terms:
        logger.info(f"--- Processing ENTITY: '{term}' ---")
        term_results: ResolvedTermData = {"wikidata": None, "dbpedia": None}
        wd_candidates = call_mcp_lookup_service(term, "wikidata", CANDIDATE_LIMIT)
        db_candidates = call_mcp_lookup_service(term, "dbpedia", CANDIDATE_LIMIT)
        # Disambiguate Entity using the target relation phrase as context
        if wd_candidates: term_results["wikidata"] = disambiguate_term_llm(natural_query, term, True, wd_candidates, "Wikidata", target_relation_phrase=target_relation_phrase)
        if db_candidates: term_results["dbpedia"] = disambiguate_term_llm(natural_query, term, True, db_candidates, "DBpedia", target_relation_phrase=target_relation_phrase)
        if term_results["wikidata"] or term_results["dbpedia"]: resolved_entities[term] = term_results
        else: logger.info(f"   No resolution found for ENTITY '{term}'.")

    if target_relation_phrase:
        logger.info(f"Agent 1 (V5): Resolving Target Relation: '{target_relation_phrase}'...")
        prop_results: ResolvedPropertyInfo = {"wikidata": None, "dbpedia": None}
        # Find property candidates using specialized functions
        wd_prop_candidates = find_wikidata_properties(target_relation_phrase, CANDIDATE_LIMIT)
        db_prop_candidates = find_dbpedia_properties(target_relation_phrase, CANDIDATE_LIMIT)
        # Disambiguate Property using resolved entities map as context
        if wd_prop_candidates: prop_results["wikidata"] = disambiguate_term_llm(natural_query, target_relation_phrase, False, wd_prop_candidates, "Wikidata", resolved_entities=resolved_entities)
        if db_prop_candidates: prop_results["dbpedia"] = disambiguate_term_llm(natural_query, target_relation_phrase, False, db_prop_candidates, "DBpedia", resolved_entities=resolved_entities)
        if prop_results["wikidata"] or prop_results["dbpedia"]: resolved_property = prop_results
        else: logger.info(f"   No resolution found for PROPERTY '{target_relation_phrase}'.")
    else:
        logger.info("Agent 1 (V5): No target relation phrase identified in query.")

    logger.info("Agent 1 (V5): Resolution finished.")
    return resolved_entities, resolved_property


# --- Main Pipeline Function for Gradio (REVISED V5) ---
def run_complete_pipeline_v5(natural_language_query: str):
    """Orchestrates the full V5 flow."""
    start_time = time.time()
    logger.info(f"\n--- Pipeline V5 Start: '{natural_language_query}' ---")

    # 1. Analyze NLQ Structure (Local LLM)
    nlq_analysis = analyze_nlq_structure_llm(natural_language_query)
    target_relation_phrase_display = nlq_analysis.get("target_relation_phrase", "N/A") if nlq_analysis else "Analysis Failed"

    # 2. Resolve Entities & Target Property (Agent 1 via MCP + Local LLM Disambiguation)
    resolved_entities, resolved_property = resolve_terms_agent_v5(natural_language_query, nlq_analysis)
    combined_resolved = { "entities": resolved_entities, "target_property": resolved_property or "Not Resolved" }; resolved_terms_json = json.dumps(combined_resolved, indent=2)

    # 3. Generate SPARQL Queries (Agent 2 - Local LLM)
    wd_sparql, db_sparql = generate_sparql_queries_llm_v5(natural_language_query, resolved_entities, resolved_property)

    # 4. Execute Queries & Format Results
    wd_ok, db_ok = False, False
    wd_results_df, wd_triples, wd_graph_img = pd.DataFrame(), "Execution not attempted.", None
    db_results_df, db_triples, db_graph_img = pd.DataFrame(), "Execution not attempted.", None

    if wd_sparql and wd_sparql != "# N/A":
        logger.info("Pipeline V5: Executing Wikidata query via MCP...")
        wd_results_raw = call_mcp_sparql_execution(WIKIDATA_ENDPOINT, wd_sparql, "Wikidata Query Exec")
        wd_results_df, wd_triples, wd_graph = format_results_for_display(wd_results_raw)
        if wd_graph: wd_graph_img = render_graph_to_base64(wd_graph)
        if wd_results_raw and not wd_results_raw.get("error") and wd_results_raw.get("results", {}).get("bindings"): wd_ok = True
    elif not wd_sparql: wd_sparql = "# No Wikidata query generated."; wd_triples = "Query not generated."

    if db_sparql and db_sparql != "# N/A":
        logger.info("Pipeline V5: Executing DBpedia query via MCP...")
        db_results_raw = call_mcp_sparql_execution(DBPEDIA_ENDPOINT, db_sparql, "DBpedia Query Exec")
        db_results_df, db_triples, db_graph = format_results_for_display(db_results_raw)
        if db_graph: db_graph_img = render_graph_to_base64(db_graph)
        if db_results_raw and not db_results_raw.get("error") and db_results_raw.get("results", {}).get("bindings"): db_ok = True
    elif not db_sparql: db_sparql = "# No DBpedia query generated."; db_triples = "Query not generated."


    # 5. Determine Final Answer: Summarize successful results or use Fallback
    consolidated_answer = ""
    if wd_ok or db_ok:
        logger.info("Pipeline V5: Summarizing successful SPARQL results...")
        wd_input_summary = wd_triples if wd_ok else "Wikidata query failed or returned no results."
        db_input_summary = db_triples if db_ok else "DBpedia query failed or returned no results."
        consolidated_answer = summarize_results_llm(natural_language_query, wd_input_summary, db_input_summary)
    else:
        logger.warning("Pipeline V5: Both queries failed or returned no results. Attempting fallback...")
        consolidated_answer = "SPARQL queries failed or yielded no results. Fallback Answer based on abstracts:\n\n"
        primary_entity_qid, primary_entity_uri = None, None
        # Heuristic: Find first resolved ENTITY term for fallback
        if nlq_analysis and resolved_entities:
             for entity_term in nlq_analysis['entities']: # Check in original classified order
                 if entity_term in resolved_entities:
                     data = resolved_entities[entity_term]
                     if data.get('wikidata') and data['wikidata'].get('id'): primary_entity_qid = data['wikidata']['id']
                     if data.get('dbpedia') and data['dbpedia'].get('uri'): primary_entity_uri = data['dbpedia']['uri']
                     if primary_entity_qid or primary_entity_uri:
                          logger.info(f"Fallback: Identified primary entity '{entity_term}' (WD: {primary_entity_qid}, DB: {primary_entity_uri})")
                          break # Use the first one found

        wd_abstract, db_abstract = None, None
        if primary_entity_qid: wd_abstract = fetch_wikidata_abstract(primary_entity_qid)
        if primary_entity_uri: db_abstract = fetch_dbpedia_abstract(primary_entity_uri)

        if wd_abstract or db_abstract:
            fallback_answer = answer_from_abstracts_llm(natural_language_query, wd_abstract, db_abstract)
            consolidated_answer += fallback_answer
        else:
            consolidated_answer += "Fallback failed: Could not identify a primary entity or retrieve its abstract."

    end_time = time.time(); logger.info(f"--- Pipeline V5 End. Total Time: {end_time - start_time:.2f} seconds ---")

    # Ensure DataFrames are always returned for Gradio, even if empty
    wd_df_out = wd_results_df if isinstance(wd_results_df, pd.DataFrame) else pd.DataFrame()
    db_df_out = db_results_df if isinstance(db_results_df, pd.DataFrame) else pd.DataFrame()

    # Return results for Gradio components
    return (
        target_relation_phrase_display, resolved_terms_json, wd_sparql, db_sparql,
        wd_df_out, db_df_out, wd_triples, db_triples,
        f'<img src="{wd_graph_img}" alt="Wikidata Graph Viz" style="max-width: 100%; height: auto;">' if wd_graph_img else "<p>Graph not available or no data.</p>",
        f'<img src="{db_graph_img}" alt="DBpedia Graph Viz" style="max-width: 100%; height: auto;">' if db_graph_img else "<p>Graph not available or no data.</p>",
        consolidated_answer
    )


# --- Gradio Interface Definition ---
def create_gradio_interface_v5():
    """Creates the Gradio UI for the V5 Agent."""
    logger.info("Creating Gradio interface V5...")
    with gr.Blocks(theme=gr.themes.Soft(), title="NLQ to Knowledge Graph Agent v5") as demo:
        gr.Markdown("# Natural Language Query to Knowledge Graph Agent (v5 - Direct Relation Focus)")
        gr.Markdown("Enter query -> Analyze Structure (Entities/Relation) -> Resolve -> Generate/Execute SPARQL -> Summarize/Fallback.")

        with gr.Row():
            nlq_input = gr.Textbox(label="Enter your Natural Language Query:", placeholder="e.g., Who is the prime minister of Canada?", lines=2, scale=4)
            submit_button = gr.Button("Run Query", variant="primary", scale=1)

        gr.Markdown("## Final Consolidated Answer")
        consolidated_output = gr.Markdown(label="Consolidated Answer") # Use Markdown for better text flow

        with gr.Accordion("See Detailed Steps & Results", open=False):
            with gr.Tabs():
                with gr.TabItem("0. NLQ Analysis"):
                    relation_output = gr.Textbox(label="Identified Target Relation/Attribute Phrase", interactive=False)

                with gr.TabItem("1. Resolved Terms"):
                     resolved_output = gr.JSON(label="Resolved Entities & Property") # Renamed for clarity

                with gr.TabItem("2. SPARQL Queries"):
                     with gr.Row():
                         wd_sparql_output = gr.Code(label="Wikidata SPARQL", language="sparql", lines=15)
                         db_sparql_output = gr.Code(label="DBpedia SPARQL", language="sparql", lines=15)

                with gr.TabItem("3. Wikidata Results"):
                     with gr.Column():
                         wd_results_df_output = gr.DataFrame(label="Wikidata Results Table", wrap=True, height=300, interactive=False)
                         wd_triples_output = gr.Textbox(label="Wikidata Results (Triples)", lines=10, interactive=False)
                         wd_graph_output = gr.HTML(label="Wikidata Graph Visualization") # Use HTML for img tag

                with gr.TabItem("4. DBpedia Results"):
                     with gr.Column():
                         db_results_df_output = gr.DataFrame(label="DBpedia Results Table", wrap=True, height=300, interactive=False)
                         db_triples_output = gr.Textbox(label="DBpedia Results (Triples)", lines=10, interactive=False)
                         db_graph_output = gr.HTML(label="DBpedia Graph Visualization") # Use HTML for img tag

        # Define outputs list matching the return order of the pipeline function
        outputs = [
            relation_output, # 0 - Target Relation Phrase
            resolved_output, # 1 - Resolved Terms JSON
            wd_sparql_output, # 2 - WD SPARQL Code
            db_sparql_output, # 3 - DB SPARQL Code
            wd_results_df_output, # 4 - WD Results DF
            db_results_df_output, # 5 - DB Results DF
            wd_triples_output, # 6 - WD Triples Text
            db_triples_output, # 7 - DB Triples Text
            wd_graph_output, # 8 - WD Graph HTML
            db_graph_output, # 9 - DB Graph HTML
            consolidated_output # 10 - Final Answer Markdown
        ]

        submit_button.click(
            fn=run_complete_pipeline_v5, # Use the V5 pipeline function
            inputs=nlq_input,
            outputs=outputs,
            api_name="query_kg_agent_v5"
        )
        # Define examples for the Gradio interface
        gr.Examples(
            examples=[
                "Who is the president of India?",
                "What is the population of Paris, France?",
                "Which actors starred in the movie Inception?",
                "Show me the chemical formula for caffeine.",
                "What are the main ingredients in Aspirin?",
                "Tell me about Alan Turing." # Example for description/fallback
            ],
            inputs=nlq_input,
            # outputs=outputs, # Outputs list can be inferred by Gradio if fn returns list/tuple
            fn=run_complete_pipeline_v5,
            cache_examples=False # Disable caching if results are dynamic or LLM-based
        )
    return demo

# --- Main Execution ---
if __name__ == "__main__":
    # Health check for the MCP server before launching UI
    try:
        health_url = f"{MCP_SERVER_URL}/health"
        logger.info(f"Checking MCP server health at {health_url}...")
        health_check = requests.get(health_url, timeout=5)
        health_check.raise_for_status()
        logger.info(f"MCP Server health check OK: {health_check.json()}")
    except requests.exceptions.RequestException as e:
        logger.error(f"CRITICAL: MCP Server at {MCP_SERVER_URL} is unreachable!")
        logger.error("Please ensure the MCP Server is running in a separate terminal using:")
        logger.error("cd ../mcp_server && uvicorn server:app --host 0.0.0.0 --port 8100")
        logger.error(f"(Error details: {e})")
        sys.exit(1) # Exit if server isn't running

    # Launch the Gradio UI
    interface = create_gradio_interface_v5()
    interface.launch(server_name="0.0.0.0") # Bind to 0.0.0.0 to allow access from network if needed
    logger.info("Gradio interface V5 launched. Access it via the provided URL.")
    # Keep the script running while Gradio is active
    try:
        while True: time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Gradio server shutting down.")
