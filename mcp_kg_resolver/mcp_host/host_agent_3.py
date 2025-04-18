import requests
import json
import logging
import sys
import time
import re # <--- Import regex
import pandas as pd
import gradio as gr
from typing import List, Dict, Optional, Any, Tuple

# Optional Graphing Imports (same as before)
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import io
    import base64
    GRAPHING_ENABLED = True
except ImportError:
    GRAPHING_ENABLED = False
    logging.warning("NetworkX or Matplotlib not found. Graph visualization disabled.")

# --- Logging, Config, LLM Init, MCP Client Functions (Same as before) ---
# (Keep all the setup, config, LLM init, call_mcp_lookup_service, call_mcp_sparql_execution)
# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MCPHostAgent")

# --- Configuration ---
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "deepseek-llm:latest"
MCP_SERVER_URL: str = "http://localhost:8100"
KG_LOOKUP_INVOKE_URL = f"{MCP_SERVER_URL}/invoke/knowledge-graph-lookup"
SPARQL_EXEC_INVOKE_URL = f"{MCP_SERVER_URL}/invoke/sparql-execution"
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
DBPEDIA_ENDPOINT = "https://dbpedia.org/sparql"
CANDIDATE_LIMIT: int = 5
REQUEST_TIMEOUT: int = 30
DBPEDIA_VERIFY_SSL: bool = True # Adjust if needed

# Type alias
CandidateInfo = Dict[str, Optional[str]]

# --- Initialize LLM ---
try:
    from langchain_community.llms import Ollama
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    llm = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    logger.info(f"Host: Successfully connected to Ollama model: {OLLAMA_MODEL}")
except ImportError:
    # ... (error handling)
    logger.critical("Host: LangChain/Ollama libraries not found.")
    sys.exit(1)
except Exception as e:
    # ... (error handling)
    logger.critical(f"Host: Connecting to Ollama failed. Details: {e}")
    sys.exit(1)

# --- MCP Client Functions ---
def call_mcp_lookup_service(term: str, source_kg: str, max_results: int) -> List[CandidateInfo]:
    # ... (Keep implementation from previous step) ...
    logger.debug(f"Calling MCP Server for term='{term}', source='{source_kg}'")
    payload = {
        "term": term, "source_kg": source_kg,
        "max_results": max_results, "dbpedia_verify_ssl": DBPEDIA_VERIFY_SSL
    }
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    try:
        response = requests.post(KG_LOOKUP_INVOKE_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return data.get("candidates", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"MCP Client: Error calling KG Lookup Service: {e}")
        return []
    except Exception as e:
        logger.error(f"MCP Client: Unexpected error calling KG Lookup Service: {e}", exc_info=True)
        return []

def call_mcp_sparql_execution(endpoint_url: str, query: str) -> Optional[Dict[str, Any]]:
    # ... (Keep implementation from previous step) ...
    logger.debug(f"Calling MCP Server for SPARQL execution on {endpoint_url}")
    payload = {"endpoint_url": endpoint_url, "query": query}
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    try:
        response = requests.post(SPARQL_EXEC_INVOKE_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT * 2)
        response.raise_for_status()
        data = response.json()
        return data.get("results")
    except requests.exceptions.Timeout:
         logger.error(f"MCP Client: Timeout calling SPARQL Execution Service for {endpoint_url}")
         return {"error": "Timeout executing query via MCP server."}
    except requests.exceptions.HTTPError as e:
         error_detail = f"HTTP Error {e.response.status_code}"
         try: server_error = e.response.json(); error_detail += f": {server_error.get('detail', e.response.reason)}"
         except: error_detail += f": {e.response.reason}"
         logger.error(f"MCP Client: HTTP Error calling SPARQL Execution Service: {error_detail}")
         return {"error": error_detail}
    except requests.exceptions.RequestException as e:
        logger.error(f"MCP Client: Error calling SPARQL Execution Service: {e}")
        return {"error": f"Failed to connect to MCP server for execution: {e}"}
    except Exception as e:
        logger.error(f"MCP Client: Unexpected error calling SPARQL Execution Service: {e}", exc_info=True)
        return {"error": f"Unexpected client error during execution call: {e}"}


# --- LLM Functions ---

def extract_focused_terms_llm(natural_query: str) -> List[str]:
    # ... (Keep implementation from previous step - already includes logging) ...
    if not llm: logger.error("LLM not available..."); return []
    logger.info(f"Host Step 1: LLM Extracting *focused* terms from: '{natural_query}'")
    # ... (prompt and chain) ...
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
    """Uses LLM to choose the best candidate based on NLQ context."""
    # ... (Input checks remain the same) ...
    if not candidates: return None
    if len(candidates) == 1: logger.info(f"   Only one candidate for '{term}', selecting directly."); return candidates[0]
    if not llm: logger.error(f"LLM not available for disambiguation of '{term}'."); return None

    logger.info(f"   LLM Disambiguating '{term}' for query '{nlq}' using {len(candidates)} candidates from {source_kg}...") # Log NLQ context being used
    # ... (Prepare candidate_text remains the same) ...
    candidate_text = ""
    for i, cand in enumerate(candidates):
        ident = cand.get('id') or cand.get('uri')
        label = cand.get('label')
        desc = cand.get('description')
        candidate_text += f"{i+1}. Identifier: {ident}\n   Label: {label}\n   Description: {desc}\n\n"

    # Prompt remains the same - it already includes NLQ context
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

        # **** MODIFIED PARSING LOGIC ****
        response_text = response.strip()
        choice_str = None
        # 1. Try to find a standalone number (more reliable)
        match = re.search(r'\b(\d+)\b', response_text)
        if match:
            choice_str = match.group(1)
            logger.debug(f"Disambiguation: Found number '{choice_str}' using regex.")
        else:
            # 2. Fallback: find any sequence of digits if regex fails
            digits = ''.join(filter(str.isdigit, response_text))
            if digits:
                 # Heuristic: If multiple digits found, take the first one assuming it's the intended choice.
                 # This handles cases like "I choose 2 because..." or "2 is the best."
                 choice_str = digits[0] # Or potentially parse the sequence if multi-digit options are expected
                 logger.debug(f"Disambiguation: Found digits '{digits}', taking first '{choice_str}' using basic filter (fallback).")
            else:
                 logger.warning(f"Disambiguation: Could not extract *any* number from LLM response: '{response_text}'")
        # **** END OF MODIFIED PARSING LOGIC ****

        if choice_str and choice_str.isdigit():
            choice_index = int(choice_str) - 1
            if 0 <= choice_index < len(candidates):
                selected_candidate = candidates[choice_index]
                logger.info(f"      LLM selected candidate {choice_index + 1} for '{term}'.")
                return selected_candidate
            elif choice_index == -1: # LLM explicitly chose 0 -> "None"
                 logger.info(f"      LLM indicated no suitable candidate for '{term}'.")
                 return None
            else:
                logger.warning(f"LLM chose number {choice_str} which is out of bounds for '{term}' (max index: {len(candidates)-1}).")
                return None
        else:
            logger.warning(f"Failed to parse a valid number from LLM disambiguation response for '{term}': '{response_text}'")
            return None

    except Exception as e:
        logger.error(f"Error during LLM disambiguation for '{term}': {e}", exc_info=True); return None


def generate_sparql_queries_llm(nlq: str, resolved_terms: Dict[str, Dict[str, Optional[CandidateInfo]]]) -> Tuple[Optional[str], Optional[str]]:
    # ... (Keep implementation from previous step - already includes logging) ...
    if not llm: logger.error("LLM not available..."); return None, None
    logger.info("Host Step 4: LLM Generating SPARQL queries...")
    # ... (context formatting) ...
    context = "Use the following resolved entities and properties:\n"
    if not resolved_terms: context += "No specific entities/properties were resolved...\n"
    else:
        for term, kgs in resolved_terms.items():
            context += f"- Term: '{term}'\n"
            wd_info = kgs.get('wikidata'); db_info = kgs.get('dbpedia')
            if wd_info: wd_id = wd_info.get('id'); wd_label = wd_info.get('label'); prefix = 'wdt:' if wd_id and wd_id.startswith('P') else 'wd:'; context += f"  - Wikidata: {wd_label} ({prefix}{wd_id})\n"
            if db_info: db_uri = db_info.get('uri'); db_label = db_info.get('label'); context += f"  - DBpedia: {db_label} (<{db_uri}>)\n"
        context += "\n"
    # ... (prompt and chain) ...
    prompt_template = PromptTemplate.from_template(
        "Based on the natural language query and the resolved entities/properties provided below, generate TWO SPARQL queries: one for Wikidata and one for DBpedia.\n\n"
        "Natural Language Query: {nlq}\n\n"
        "Resolved Context:\n{context}\n\n"
        "Instructions for Queries:\n"
        "1.  **Wikidata Query:** Use `wd:`/`wdt:` prefixes. Select relevant variables. **Include labels** for selected variables using `?varLabel` and the standard `SERVICE wikibase:label` block.\n"
        "2.  **DBpedia Query:** Use full URIs provided. Use common prefixes (`dbo:`, `dbp:`, `dbr:`, `rdfs:`, `rdf:`). Select relevant variables. Try to select labels using `rdfs:label` and filter by language (`LANG(?label) = 'en'`).\n"
        "3.  **Output Format:** Provide ONLY the two queries. Separate them clearly using `--- WIKIDATA ---` on one line, followed by the Wikidata query in a ```sparql block, then `--- DBPEDIA ---` on another line, followed by the DBpedia query in a ```sparql block. Do not add any other text.\n\n"
        "--- WIKIDATA ---\n"
        "```sparql\n"
        "# Wikidata Query Here\n"
        "```\n\n"
        "--- DBPEDIA ---\n"
        "```sparql\n"
        "# DBpedia Query Here\n"
        "```\n"
    )
    chain = prompt_template | llm | StrOutputParser()
    try:
        response = chain.invoke({"nlq": nlq, "context": context})
        # ... (regex parsing - unchanged) ...
        wd_query, db_query = None, None
        wd_match = re.search(r"--- WIKIDATA ---\s*```sparql\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if wd_match: wd_query = wd_match.group(1).strip(); logger.info("   LLM Generated Wikidata Query.")
        db_match = re.search(r"--- DBPEDIA ---\s*```sparql\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if db_match: db_query = db_match.group(1).strip(); logger.info("   LLM Generated DBpedia Query.")
        # ... (logging warnings if missing) ...
        if not wd_query and not db_query: logger.warning(f"LLM failed to generate SPARQL queries in expected format.")
        elif not wd_query: logger.warning("LLM failed to generate Wikidata query.")
        elif not db_query: logger.warning("LLM failed to generate DBpedia query.")
        return wd_query, db_query
    except Exception as e:
        logger.error(f"Error during LLM SPARQL generation: {e}", exc_info=True)
        return None, None


# --- NEW: Agent 3: Result Consolidation (Local LLM) ---
def summarize_results_llm(nlq: str, wd_results_str: str, db_results_str: str) -> str:
    """Generates a natural language summary based on KG results."""
    if not llm:
        logger.error("LLM not available for result summarization.")
        return "Error: LLM not available for summarization."

    # Basic check if we actually got any results beyond errors/no-results messages
    meaningful_wd = "Query executed successfully, but returned no results." not in wd_results_str and "Query not generated" not in wd_results_str and "Execution Error" not in wd_results_str
    meaningful_db = "Query executed successfully, but returned no results." not in db_results_str and "Query not generated" not in db_results_str and "Execution Error" not in db_results_str

    if not meaningful_wd and not meaningful_db:
        logger.info("No meaningful results from either KG to summarize.")
        # Return the most relevant status message
        if "Execution Error" in wd_results_str or "Execution Error" in db_results_str:
            return "Could not generate a final answer due to errors during query execution."
        if "Query not generated" in wd_results_str and "Query not generated" in db_results_str:
             return "Could not generate queries to retrieve an answer."
        return "No information found in Wikidata or DBpedia for this query."

    logger.info("Step 6: LLM Summarizing results...")

    # Prepare context for summarization prompt
    context = ""
    if meaningful_wd:
        context += f"Wikidata Findings (Triples Summary):\n{wd_results_str}\n\n"
    else:
         context += "Wikidata Findings: No results or query failed.\n\n"

    if meaningful_db:
        context += f"DBpedia Findings (Triples Summary):\n{db_results_str}\n\n"
    else:
         context += "DBpedia Findings: No results or query failed.\n\n"


    # Summarization Prompt
    prompt_template = PromptTemplate.from_template(
        "Based on the original question and the following structured findings retrieved from knowledge graphs, provide a concise, natural language answer. Synthesize the information from both sources if applicable and relevant. If the results seem contradictory or insufficient to answer the question directly, state that clearly.\n\n"
        "Original Question: \"{nlq}\"\n\n"
        "Knowledge Graph Findings:\n{context}\n"
        "Focus on directly answering the question based *only* on the provided findings.\n\n"
        "Final Consolidated Answer:"
    )
    chain = prompt_template | llm | StrOutputParser()

    try:
        summary = chain.invoke({"nlq": nlq, "context": context})
        logger.info("   LLM generated consolidated summary.")
        return summary.strip()
    except Exception as e:
        logger.error(f"Error during LLM result summarization: {e}", exc_info=True)
        return "Error: Failed to generate a consolidated summary."


# --- Result Formatting (Unchanged from previous step) ---
def format_results_for_display(results_dict: Optional[Dict[str, Any]]) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[Any]]:
    # ... (Keep implementation from previous step) ...
    if not results_dict: return None, "No results received.", None
    if results_dict.get("error"): return None, f"Execution Error: {results_dict['error']}", None
    bindings = results_dict.get("results", {}).get("bindings", [])
    if not bindings: return pd.DataFrame(), "Query executed successfully, but returned no results.", None
    # ... (DataFrame processing logic) ...
    processed_data = []; variables = results_dict.get("head", {}).get("vars", []); label_vars = {v + "Label" for v in variables}
    for binding in bindings:
        row = {}; processed_simple_vars = set()
        for var in variables:
            if var in processed_simple_vars: continue
            label_var = f"{var}Label"; value_info = binding.get(var); label_info = binding.get(label_var)
            display_value = None; raw_value = value_info.get("value") if value_info else None
            if label_info and label_info.get("value"): display_value = label_info.get("value"); processed_simple_vars.add(var); processed_simple_vars.add(label_var)
            elif value_info:
                display_value = raw_value
                if value_info.get("type") == "uri":
                    if "wikidata.org/entity/" in display_value: display_value = f"wd:{display_value.rsplit('/', 1)[-1]}"
                    elif "dbpedia.org/resource/" in display_value: display_value = f"dbr:{display_value.rsplit('/', 1)[-1]}"
                    elif "dbpedia.org/ontology/" in display_value: display_value = f"dbo:{display_value.rsplit('/', 1)[-1]}"
                processed_simple_vars.add(var)
            row[var] = display_value
        processed_data.append(row)
    df = pd.DataFrame(processed_data)
    display_vars = [v for v in variables if v not in label_vars or v+'Label' not in df.columns]
    try: df = df[[v for v in display_vars if v in df.columns]]
    except KeyError: pass
    # ... (Triple extraction logic) ...
    triples_str = ""; graph = None; added_triples = set()
    if GRAPHING_ENABLED: graph = nx.DiGraph()
    subj_var, pred_var, obj_var = None, None, None
    if 's' in variables and 'p' in variables and 'o' in variables: subj_var, pred_var, obj_var = 's', 'p', 'o'
    elif 'item' in variables and ('property' in variables or 'prop' in variables) and 'value' in variables: subj_var, pred_var, obj_var = 'item', variables[1], 'value' # Heuristic
    elif len(variables) >= 3: subj_var, pred_var, obj_var = variables[0], variables[1], variables[2]
    if subj_var and pred_var and obj_var:
        triples_list = []
        for row_data in processed_data:
            s = row_data.get(subj_var); p = row_data.get(pred_var); o = row_data.get(obj_var)
            if s and p and o:
                s_disp = str(s); p_disp = str(p).split('/')[-1].split('#')[-1]; o_disp = str(o)
                triple = (s_disp, p_disp, o_disp)
                if triple not in added_triples:
                    triples_list.append(f"({s_disp}) --[{p_disp}]--> ({o_disp})"); added_triples.add(triple)
                    if GRAPHING_ENABLED and graph is not None: graph.add_edge(s_disp, o_disp, label=p_disp)
        if triples_list: triples_str = f"Found {len(triples_list)} unique triples:\n" + "\n".join(triples_list)
        else: triples_str = "Could not automatically extract triples."
    else: triples_str = "Could not identify subject/predicate/object variables."
    # Ensure DF is returned even if empty
    df_to_return = df if df is not None else pd.DataFrame()
    return df_to_return, triples_str, graph


def render_graph_to_base64(graph) -> Optional[str]:
    # ... (Keep implementation from previous step) ...
    if not GRAPHING_ENABLED or graph is None or graph.number_of_nodes() == 0: return None
    try:
        plt.figure(figsize=(10, 8)); pos = nx.spring_layout(graph, k=0.6, iterations=50)
        nx.draw_networkx_nodes(graph, pos, node_size=2500, node_color='lightblue', alpha=0.9)
        nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.6, arrowsize=15, connectionstyle='arc3,rad=0.1')
        nx.draw_networkx_labels(graph, pos, font_size=10)
        edge_labels = nx.get_edge_attributes(graph, 'label')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, label_pos=0.3, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
        plt.title("Result Subgraph"); plt.axis('off')
        buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); plt.close(); buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    except Exception as e: logger.error(f"Failed to render graph: {e}", exc_info=True); return None


# --- Main Pipeline Function for Gradio ---
def run_complete_pipeline(natural_language_query: str):
    """Orchestrates the full NLQ -> Resolve -> SPARQL -> Execute -> Format -> Summarize flow."""
    start_time = time.time()
    logger.info(f"\n--- Pipeline Start: '{natural_language_query}' ---")

    # Agent 1: Resolve Entities/Properties via MCP
    resolved_terms = resolve_terms_agent_mcp(natural_language_query) # Internal function orchestrates MCP lookup + local LLM disamb
    resolved_terms_json = json.dumps(resolved_terms, indent=2) if resolved_terms else "{}"
    logger.info("Pipeline: Resolution finished.")

    # Agent 2 Part 1: Generate SPARQL (Local LLM)
    wd_sparql, db_sparql = generate_sparql_queries_llm(natural_language_query, resolved_terms or {}) # Pass empty if None
    logger.info("Pipeline: SPARQL generation finished.")

    # Agent 2 Part 2: Execute Wikidata Query via MCP
    wd_results_raw = None; wd_results_df = pd.DataFrame(); wd_triples = "Query not generated or execution failed."; wd_graph_img = None
    if wd_sparql:
        logger.info("Pipeline: Executing Wikidata query via MCP...")
        wd_results_raw = call_mcp_sparql_execution(WIKIDATA_ENDPOINT, wd_sparql)
        wd_results_df, wd_triples, wd_graph = format_results_for_display(wd_results_raw)
        if wd_graph: wd_graph_img = render_graph_to_base64(wd_graph)
    else: wd_sparql = "# No Wikidata query generated."

    # Agent 2 Part 3: Execute DBpedia Query via MCP
    db_results_raw = None; db_results_df = pd.DataFrame(); db_triples = "Query not generated or execution failed."; db_graph_img = None
    if db_sparql:
        logger.info("Pipeline: Executing DBpedia query via MCP...")
        db_results_raw = call_mcp_sparql_execution(DBPEDIA_ENDPOINT, db_sparql)
        db_results_df, db_triples, db_graph = format_results_for_display(db_results_raw)
        if db_graph: db_graph_img = render_graph_to_base64(db_graph)
    else: db_sparql = "# No DBpedia query generated."

    # Agent 3: Consolidate Results (Local LLM)
    consolidated_answer = summarize_results_llm(natural_language_query, wd_triples, db_triples)
    logger.info("Pipeline: Result summarization finished.")

    end_time = time.time()
    logger.info(f"--- Pipeline End. Total Time: {end_time - start_time:.2f} seconds ---")

    # Return results for Gradio components - ADD consolidated_answer
    return (
        resolved_terms_json,
        wd_sparql,
        db_sparql,
        wd_results_df if wd_results_df is not None else pd.DataFrame(),
        db_results_df if db_results_df is not None else pd.DataFrame(),
        wd_triples,
        db_triples,
        f'<img src="{wd_graph_img}" alt="Wikidata Graph Visualization">' if wd_graph_img else "Graph visualization not available or no data.",
        f'<img src="{db_graph_img}" alt="DBpedia Graph Visualization">' if db_graph_img else "Graph visualization not available or no data.",
        consolidated_answer # <-- Added output
    )

# --- Internal Orchestration for Agent 1 ---
def resolve_terms_agent_mcp(natural_query: str) -> Optional[Dict[str, Dict[str, Optional[CandidateInfo]]]]:
    """Internal function for Agent 1 logic (extraction, MCP lookup, disambiguation)."""
    # 1. Focused term extraction (Local LLM call)
    extracted_terms = extract_focused_terms_llm(natural_query)
    if not extracted_terms:
        logger.warning("Agent 1: No terms extracted by LLM.")
        return None # Return None if extraction fails

    # 2. & 3. Get candidates via MCP and disambiguate locally
    final_resolution: Dict[str, Dict[str, Optional[CandidateInfo]]] = {}
    logger.info("Agent 1: Getting candidates via MCP and disambiguating terms...")

    for term in extracted_terms:
        logger.info(f"--- Agent 1: Processing term: '{term}' ---")
        term_results: Dict[str, Optional[CandidateInfo]] = {"wikidata": None, "dbpedia": None}

        # Wikidata Resolution Path
        wd_candidates = call_mcp_lookup_service(term, "wikidata", CANDIDATE_LIMIT)
        if wd_candidates:
            selected_wikidata = disambiguate_term_llm(natural_query, term, wd_candidates, "Wikidata")
            term_results["wikidata"] = selected_wikidata

        # DBpedia Resolution Path
        db_candidates = call_mcp_lookup_service(term, "dbpedia", CANDIDATE_LIMIT)
        if db_candidates:
            selected_dbpedia = disambiguate_term_llm(natural_query, term, db_candidates, "DBpedia")
            term_results["dbpedia"] = selected_dbpedia

        # Only add term to results if at least one KG found something
        if term_results["wikidata"] or term_results["dbpedia"]:
            final_resolution[term] = term_results
        else:
            logger.info(f"   No resolution found for term '{term}' in either KG.")


    logger.info("Agent 1: Resolution finished.")
    return final_resolution if final_resolution else None # Return None if no terms were resolved


# --- Gradio Interface Definition ---
def create_gradio_interface():
    logger.info("Creating Gradio interface...")
    with gr.Blocks(theme=gr.themes.Soft(), title="NLQ to Knowledge Graph Agent v2") as demo:
        gr.Markdown("# Natural Language Query to Knowledge Graph Agent (v2)")
        gr.Markdown("Enter your question. The system resolves terms, generates/executes SPARQL queries, and synthesizes a final answer.")

        with gr.Row():
            nlq_input = gr.Textbox(label="Enter your Natural Language Query:", placeholder="e.g., Who is the prime minister of Canada?", lines=2, scale=4)
            submit_button = gr.Button("Run Query", variant="primary", scale=1)

        with gr.Accordion("See Detailed Steps & Results", open=False): # Make details collapsible
            with gr.Tabs():
                with gr.TabItem("1. Resolved Terms"):
                     resolved_output = gr.JSON(label="Resolved Terms (Agent 1 Output)")
                with gr.TabItem("2. SPARQL Queries"):
                     with gr.Row():
                         wd_sparql_output = gr.Code(label="Wikidata SPARQL", language="sparql", lines=15)
                         db_sparql_output = gr.Code(label="DBpedia SPARQL", language="sparql", lines=15)
                with gr.TabItem("3. Wikidata Results"):
                     with gr.Column():
                         wd_results_df_output = gr.DataFrame(label="Wikidata Results Table", wrap=True, height=300)
                         wd_triples_output = gr.Textbox(label="Wikidata Results (Triples)", lines=10, interactive=False)
                         wd_graph_output = gr.HTML(label="Wikidata Graph Visualization")
                with gr.TabItem("4. DBpedia Results"):
                     with gr.Column():
                         db_results_df_output = gr.DataFrame(label="DBpedia Results Table", wrap=True, height=300)
                         db_triples_output = gr.Textbox(label="DBpedia Results (Triples)", lines=10, interactive=False)
                         db_graph_output = gr.HTML(label="DBpedia Graph Visualization")

        # Final Answer section always visible below the input
        gr.Markdown("## Final Consolidated Answer")
        consolidated_output = gr.Markdown(label="Consolidated Answer") # Use Markdown for better formatting

        # Define outputs list in the correct order expected by run_complete_pipeline
        # Match the return statement of run_complete_pipeline
        outputs = [
            resolved_output, wd_sparql_output, db_sparql_output,
            wd_results_df_output, db_results_df_output,
            wd_triples_output, db_triples_output,
            wd_graph_output, db_graph_output,
            consolidated_output # <-- Added output component
        ]

        submit_button.click(
            fn=run_complete_pipeline,
            inputs=nlq_input,
            outputs=outputs,
            api_name="query_kg_agent_v2"
        )
        gr.Examples(
            examples=[
                "Who is the president of India?",
                "What is the population of Paris, France?",
                "Which actors starred in the movie Inception?",
                "Show me the chemical formula for caffeine.",
                "What are the main ingredients in Aspirin?"
            ],
            inputs=nlq_input,
            # outputs=outputs, # Outputs list must match for examples too
            fn=run_complete_pipeline,
            # cache_examples=False, # Recommended to disable caching
        )
    return demo


# --- Main Execution ---
if __name__ == "__main__":
    # ... (Keep health check from previous step) ...
    try:
        health_url = f"{MCP_SERVER_URL}/health"; health_check = requests.get(health_url, timeout=5); health_check.raise_for_status()
        logger.info(f"MCP Server health check OK: {health_check.json()}")
    except requests.exceptions.RequestException as e:
        logger.error(f"CRITICAL: MCP Server at {MCP_SERVER_URL} is unreachable! Start it first.")
        logger.error(f"Error details: {e}")
        sys.exit(1)

    # Launch the Gradio UI
    interface = create_gradio_interface()
    interface.launch()
    logger.info("Gradio interface launched.")
