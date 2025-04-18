# mcp_host/host_agent.py (V5 - Direct Relation Focus)

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

# --- Configuration (Using sensible defaults, adjust if needed) ---
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
REQUEST_TIMEOUT: int = 30 # Slightly longer default timeout for external calls
DBPEDIA_VERIFY_SSL: bool = True # Default to True for security, adjust based on env
USER_AGENT: str = "AdvancedResolverAgentV5/Host (YourApp; contact@example.com)"

# --- Type Definitions ---
CandidateInfo = Dict[str, Optional[str]]
class NlqAnalysisResult(TypedDict): entities: List[str]; target_relation_phrase: Optional[str]
class ResolvedTermData(TypedDict): wikidata: Optional[CandidateInfo]; dbpedia: Optional[CandidateInfo]
ResolvedEntitiesMap = Dict[str, ResolvedTermData]
ResolvedPropertyInfo = ResolvedTermData # Reuse structure

# --- Initialize LLM ---
try:
    from langchain_community.llms import Ollama
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser # JsonOutputParser can be less reliable
    llm = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    logger.info(f"Host: Successfully connected to Ollama model: {OLLAMA_MODEL}")
except ImportError: logger.critical("Host: LangChain/Ollama libraries not found."); sys.exit(1)
except Exception as e: logger.critical(f"Host: Connecting to Ollama failed. Details: {e}"); sys.exit(1)


# --- MCP Client Functions (Robust version from V4/V5 thoughts) ---
def call_mcp_service(url: str, payload: Dict, context: str) -> Optional[Dict[str, Any]]:
    """Generic function to call an MCP service endpoint with error handling."""
    logger.debug(f"Calling MCP Service ({context}) at {url}")
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    response_data = None
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            response_data = response.json()
            logger.debug(f"MCP Service ({context}) successful.")
        # Handle specific HTTP error ranges with more informative logging/error structure
        elif 400 <= response.status_code < 500:
             error_detail = f"Client Error {response.status_code}"
             try: server_error = response.json(); error_detail += f": {server_error.get('detail', response.reason)}"
             except: error_detail += f": {response.reason}"
             logger.error(f"MCP Client: {context} failed: {error_detail}")
             response_data = {"error": error_detail} # Propagate structured error
        elif response.status_code >= 500:
             error_detail = f"Server Error {response.status_code}"
             try: server_error = response.json(); error_detail += f": {server_error.get('detail', response.reason)}"
             except: error_detail += f": {response.reason}"
             logger.error(f"MCP Client: {context} failed: {error_detail}")
             response_data = {"error": error_detail}
        else:
            response.raise_for_status() # Raise for other unexpected statuses
    except requests.exceptions.Timeout: logger.error(f"MCP Client: Timeout calling {context} service at {url}"); response_data = {"error": f"Timeout calling MCP service ({context})."}
    except requests.exceptions.ConnectionError as e: logger.error(f"MCP Client: Connection Error calling {context} service at {url}: {e}"); response_data = {"error": f"MCP connection error ({context}): {e}"}
    except requests.exceptions.RequestException as e: logger.error(f"MCP Client: RequestException calling {context} service at {url}: {e}"); response_data = {"error": f"MCP request error ({context}): {e}"}
    except json.JSONDecodeError as e: logger.error(f"MCP Client: Failed to decode JSON response from {context} service. Error: {e}. Response: {response.text[:200]}..."); response_data = {"error": f"Invalid JSON response from MCP service ({context})."}
    except Exception as e: logger.error(f"MCP Client: Unexpected error calling {context} service: {e}", exc_info=True); response_data = {"error": f"Unexpected client error calling MCP ({context})."}
    return response_data

def call_mcp_lookup_service(term: str, source_kg: str, max_results: int) -> List[CandidateInfo]:
    payload = {"term": term, "source_kg": source_kg, "max_results": max_results, "dbpedia_verify_ssl": DBPEDIA_VERIFY_SSL}
    response_data = call_mcp_service(KG_LOOKUP_INVOKE_URL, payload, f"KG Lookup ({source_kg} for '{term}')")
    if response_data and not response_data.get("error"):
        return response_data.get("candidates", [])
    return []

def call_mcp_sparql_execution(endpoint_url: str, query: str, context: str = "SPARQL Execution") -> Optional[Dict[str, Any]]:
    payload = {"endpoint_url": endpoint_url, "query": query}
    # Returns dict which might contain 'results' or 'error' key
    return call_mcp_service(SPARQL_EXEC_INVOKE_URL, payload, context)


# --- LLM Functions ---

def analyze_nlq_structure_llm(nlq: str) -> Optional[NlqAnalysisResult]:
    """V5 Core: Analyzes NLQ for entities and target relation phrase."""
    if not llm: logger.error("LLM not available..."); return None
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
        # Robust JSON parsing
        match = re.search(r'\{.*\}', response_str, re.DOTALL)
        if not match: logger.error(f"Could not find JSON in LLM NLQ analysis: {response_str}"); return None
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            if isinstance(data, dict) and 'entities' in data and isinstance(data['entities'], list) and 'target_relation_phrase' in data:
                entities_list = [str(e) for e in data['entities'] if isinstance(e, (str, int, float))]
                relation_phrase = data['target_relation_phrase']
                result: NlqAnalysisResult = { "entities": entities_list, "target_relation_phrase": str(relation_phrase) if isinstance(relation_phrase, str) else None }
                logger.info(f"   LLM NLQ Analysis Result: {result}")
                return result
            else: logger.error(f"LLM NLQ analysis JSON lacks structure: {data}"); return None
        except json.JSONDecodeError as json_err: logger.error(f"Failed parsing LLM NLQ analysis JSON: {json_err}. Response: {response_str}"); return None
    except Exception as e: logger.error(f"   Error during LLM NLQ analysis: {e}", exc_info=True); return None

# --- Specific KG Search Functions (Same as V4/V5) ---
def find_wikidata_properties(term_phrase: str, limit: int) -> List[CandidateInfo]:
    logger.info(f"   Querying Wikidata for PROPERTY candidates of: '{term_phrase}'")
    candidates = []
    params = { "action": "wbsearchentities", "format": "json", "language": "en", "uselang": "en", "search": term_phrase, "limit": limit, "type": "property", "origin": "*" }
    headers = {'User-Agent': USER_AGENT}
    try:
        response = requests.get(WIKIDATA_API_ENDPOINT, params=params, headers=headers, timeout=REQUEST_TIMEOUT); response.raise_for_status(); data = response.json(); search_results = data.get("search", [])
        for match in search_results:
            if match.get("id") and match["id"].startswith("P"): candidates.append({"id": match.get("id"), "label": match.get("label"), "description": match.get("description") or "No desc", "uri": match.get("concepturi")})
        logger.info(f"      Wikidata Property Search: Found {len(candidates)} for '{term_phrase}'.")
    except Exception as e: logger.error(f"      Wikidata Property Search error for '{term_phrase}': {e}")
    return candidates

def find_dbpedia_properties(term_phrase: str, limit: int) -> List[CandidateInfo]:
    logger.info(f"   Querying DBpedia for PROPERTY candidates of: '{term_phrase}' via SPARQL")
    candidates = []
    escaped_term = re.sub(r'[-\/\\^$*+?.()|[\]{}]', r'\\&', term_phrase)
    query = f"""SELECT DISTINCT ?property ?label ?comment WHERE {{ ?property a rdf:Property . ?property rdfs:label ?label . OPTIONAL {{ ?property rdfs:comment ?comment . FILTER(LANG(?comment) = 'en') }} FILTER(REGEX(STR(?label), "{escaped_term}", "i")) FILTER(LANG(?label) = 'en') }} LIMIT {limit}"""
    results_dict = call_mcp_sparql_execution(DBPEDIA_ENDPOINT, query, context="DBpedia Property Search")
    if results_dict and not results_dict.get("error"):
        bindings = results_dict.get("results", {}).get("bindings", [])
        for binding in bindings:
            uri = binding.get("property", {}).get("value"); label = binding.get("label", {}).get("value"); desc = binding.get("comment", {}).get("value", "No description")
            if uri and label: candidates.append({"id": None, "label": label, "description": desc, "uri": uri})
        logger.info(f"      DBpedia Property Search: Found {len(candidates)} for '{term_phrase}'.")
    else: logger.warning(f"      DBpedia Property Search failed for '{term_phrase}'. Error: {results_dict.get('error', 'Unknown') if results_dict else 'No response'}")
    return candidates

# --- LLM Disambiguation (Using V5 robust parsing and context logic) ---
def disambiguate_term_llm( nlq: str, term_being_resolved: str, is_entity: bool, candidates: List[CandidateInfo], source_kg: str, target_relation_phrase: Optional[str] = None, resolved_entities: Optional[ResolvedTermsMap] = None) -> Optional[CandidateInfo]:
    if not candidates: return None
    if len(candidates) == 1: logger.info(f"   Only one candidate for '{term_being_resolved}', selecting directly."); return candidates[0]
    if not llm: logger.error(f"LLM not available for disambiguation of '{term_being_resolved}'."); return None
    logger.info(f"   LLM Disambiguating {'ENTITY' if is_entity else 'PROPERTY'} '{term_being_resolved}' for query '{nlq}'...")
    candidate_text = "" #...(same candidate formatting loop)...
    for i, cand in enumerate(candidates): ident = cand.get('id') or cand.get('uri'); label = cand.get('label'); desc = cand.get('description'); candidate_text += f"{i+1}. Id: {ident}\n   Lbl: {label}\n   Desc: {desc}\n\n"
    extra_context = "" #...(same context building logic)...
    if is_entity and target_relation_phrase: extra_context = f"Context: User wants relation '{target_relation_phrase}'.\n"
    elif not is_entity and resolved_entities: extra_context += "Context Entities:\n" #...(same entity listing logic)...

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
            elif choice_index == -1: logger.info("LLM indicated no suitable candidate."); return None
            else: logger.warning(f"LLM chose out-of-bounds number {choice_str}."); return None
        else: logger.warning(f"Failed to parse number from LLM disambiguation: '{response_text}'."); return None
    except Exception as e: logger.error(f"Error during LLM disambiguation for '{term_being_resolved}': {e}", exc_info=True); return None


# --- SPARQL Generation (Using V5 prompt focused on direct relation) ---
def generate_sparql_queries_llm_v5(nlq: str, resolved_entities: ResolvedTermsMap, resolved_property: Optional[ResolvedPropertyInfo]) -> Tuple[Optional[str], Optional[str]]:
    if not llm: logger.error("LLM not available..."); return None, None
    logger.info("Step 4: LLM Generating SPARQL queries (V5 - Direct Property)...")
    entity_context = "Entities:\n"; prop_context = "Target Relation/Attribute:\n"; #...(V5 context formatting)...
    if not resolved_entities: entity_context += "  (None found)\n"
    else: #...(loop to format entities)...
    if not resolved_property: prop_context += "  (None identified or resolved)\n"
    else: #...(format property)...
    # V5 Prompt Template
    prompt_template = PromptTemplate( template=("..."), input_variables=["nlq", "entity_context", "prop_context"] ) # V5 Prompt
    chain = prompt_template | llm | StrOutputParser()
    try:
        response = chain.invoke({"nlq": nlq, "entity_context": entity_context, "prop_context": prop_context})
        # ... (V5 Regex parsing for WD/DB queries) ...
        return wd_query, db_query
    except Exception as e: logger.error(f"Error during LLM SPARQL generation V5: {e}", exc_info=True); return None, None


# --- Fallback Funcs (fetch_wikidata_abstract, fetch_dbpedia_abstract, answer_from_abstracts_llm - Same as V4/V5) ---
def fetch_wikidata_abstract(qid: str) -> Optional[str]: #...(same V4/V5 implementation)...
def fetch_dbpedia_abstract(uri: str) -> Optional[str]: #...(same V4/V5 implementation)...
def answer_from_abstracts_llm(nlq: str, wd_abstract: Optional[str], db_abstract: Optional[str]) -> str: #...(same V4/V5 implementation)...


# --- Result Formatting (format_results_for_display, render_graph_to_base64 - Same as V4/V5) ---
def format_results_for_display(results_dict: Optional[Dict[str, Any]]) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[Any]]: #...(same V4/V5 implementation)...
def render_graph_to_base64(graph) -> Optional[str]: #...(same V4/V5 implementation)...


# --- Agent 1 Internal Orchestration (Using V5 structure) ---
def resolve_terms_agent_v5(natural_query: str, analysis: NlqAnalysisResult) -> Tuple[ResolvedEntitiesMap, Optional[ResolvedPropertyInfo]]:
    # ... (Keep V5 implementation: resolve entities first, then property using context) ...
    resolved_entities: ResolvedEntitiesMap = {}; resolved_property: Optional[ResolvedPropertyInfo] = None
    if not analysis: logger.warning("Agent 1 (V5): NLQ Analysis failed..."); return resolved_entities, resolved_property
    entity_terms = analysis['entities']; target_relation_phrase = analysis['target_relation_phrase']
    logger.info("Agent 1 (V5): Resolving Entities..."); #...(loop entities, call lookup, call disambiguate)...
    if target_relation_phrase: logger.info(f"Agent 1 (V5): Resolving Target Relation: '{target_relation_phrase}'..."); #...(find props, call disambiguate)...
    logger.info("Agent 1 (V5): Resolution finished."); return resolved_entities, resolved_property


# --- Summarization Func (Same as V4/V5) ---
def summarize_results_llm(nlq: str, wd_results_str: str, db_results_str: str) -> str:
    # ... (Keep V4/V5 implementation) ...


# --- Main Pipeline Function for Gradio (Using V5 structure) ---
def run_complete_pipeline_v5(natural_language_query: str):
    """Orchestrates the full V5 flow."""
    start_time = time.time()
    logger.info(f"\n--- Pipeline V5 Start: '{natural_language_query}' ---")

    # 1. Analyze NLQ Structure
    nlq_analysis = analyze_nlq_structure_llm(natural_language_query)
    target_relation_phrase_display = nlq_analysis.get("target_relation_phrase", "N/A") if nlq_analysis else "Analysis Failed"

    # 2. Resolve Entities & Target Property
    resolved_entities, resolved_property = resolve_terms_agent_v5(natural_language_query, nlq_analysis)
    combined_resolved = { "entities": resolved_entities, "target_property": resolved_property or "Not Resolved" }; resolved_terms_json = json.dumps(combined_resolved, indent=2)

    # 3. Generate SPARQL Queries
    wd_sparql, db_sparql = generate_sparql_queries_llm_v5(natural_language_query, resolved_entities, resolved_property)

    # 4. Execute Queries & Format Results
    wd_ok, db_ok = False, False #...(execution logic same as V5)...
    wd_results_df, wd_triples, wd_graph_img = pd.DataFrame(), "N/A", None
    db_results_df, db_triples, db_graph_img = pd.DataFrame(), "N/A", None
    if wd_sparql: wd_results_raw = call_mcp_sparql_execution(...); wd_results_df, wd_triples, wd_graph = format_results_for_display(wd_results_raw); #...(graph, set wd_ok)...
    else: wd_sparql = "# N/A"; wd_triples = "Query not generated."
    if db_sparql: db_results_raw = call_mcp_sparql_execution(...); db_results_df, db_triples, db_graph = format_results_for_display(db_results_raw); #...(graph, set db_ok)...
    else: db_sparql = "# N/A"; db_triples = "Query not generated."

    # 5. Determine Final Answer (Summarize or Fallback)
    consolidated_answer = "" #...(Fallback/Summarize logic same as V5)...
    if wd_ok or db_ok: #...(Summarize successful results)...
    else: # Fallback #...(Fallback logic same as V5)...

    end_time = time.time(); logger.info(f"--- Pipeline V5 End. Time: {end_time - start_time:.2f}s ---")

    # Return results for Gradio
    return ( target_relation_phrase_display, resolved_terms_json, wd_sparql, db_sparql,
             wd_results_df if wd_results_df is not None else pd.DataFrame(),
             db_results_df if db_results_df is not None else pd.DataFrame(),
             wd_triples, db_triples,
             f'<img src="{wd_graph_img}" alt="WD Graph">' if wd_graph_img else "Graph N/A.",
             f'<img src="{db_graph_img}" alt="DB Graph">' if db_graph_img else "Graph N/A.",
             consolidated_answer )


# --- Gradio Interface Definition (Using V5 structure) ---
def create_gradio_interface_v5():
    logger.info("Creating Gradio interface V5...")
    with gr.Blocks(theme=gr.themes.Soft(), title="NLQ to KG Agent v5") as demo:
        # ... (Keep V5 Layout: Title, Input, Final Answer, Accordion with Tabs) ...
        gr.Markdown("# NLQ to Knowledge Graph Agent (v5 - Direct Relation Focus)")
        with gr.Row(): nlq_input = gr.Textbox(...); submit_button = gr.Button(...)
        gr.Markdown("## Final Consolidated Answer"); consolidated_output = gr.Markdown(...)
        with gr.Accordion(...):
            with gr.Tabs():
                 with gr.TabItem("0. NLQ Analysis"): relation_output = gr.Textbox(label="Identified Target Relation/Attribute Phrase", ...)
                 with gr.TabItem("1. Resolved Terms"): resolved_output = gr.JSON(...)
                 with gr.TabItem("2. SPARQL Queries"): wd_sparql_output = gr.Code(...); db_sparql_output = gr.Code(...)
                 with gr.TabItem("3. Wikidata Results"): wd_results_df_output = gr.DataFrame(...); wd_triples_output = gr.Textbox(...); wd_graph_output = gr.HTML(...)
                 with gr.TabItem("4. DBpedia Results"): db_results_df_output = gr.DataFrame(...); db_triples_output = gr.Textbox(...); db_graph_output = gr.HTML(...)

        # Outputs list matching the return order of the pipeline function
        outputs = [ relation_output, resolved_output, wd_sparql_output, db_sparql_output, wd_results_df_output, db_results_df_output, wd_triples_output, db_triples_output, wd_graph_output, db_graph_output, consolidated_output ]

        submit_button.click( fn=run_complete_pipeline_v5, inputs=nlq_input, outputs=outputs, api_name="query_kg_agent_v5" )
        gr.Examples( examples=[...], inputs=nlq_input, fn=run_complete_pipeline_v5, )
    return demo

# --- Main Execution ---
if __name__ == "__main__":
    # ... (Keep health check) ...
    try: health_url = f"{MCP_SERVER_URL}/health"; health_check = requests.get(health_url, timeout=5); health_check.raise_for_status(); logger.info(f"MCP Server health check OK.")
    except requests.exceptions.RequestException as e: logger.error(f"CRITICAL: MCP Server at {MCP_SERVER_URL} is unreachable! Start it first."); sys.exit(1)

    interface = create_gradio_interface_v5() # Call correct function
    interface.launch(server_name="0.0.0.0") # Bind to allow network access if needed
    logger.info("Gradio interface V5 launched.")
