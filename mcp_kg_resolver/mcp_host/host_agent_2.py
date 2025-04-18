import requests
import json
import logging
import sys
import time
import re # For splitting SPARQL queries
import pandas as pd
import gradio as gr
from typing import List, Dict, Optional, Any, Tuple

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


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MCPHostAgent")

# --- Configuration ---
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "deepseek-llm:latest" # Model used for generation/disambiguation

MCP_SERVER_URL: str = "http://localhost:8100" # Local MCP Server URL
KG_LOOKUP_INVOKE_URL = f"{MCP_SERVER_URL}/invoke/knowledge-graph-lookup"
SPARQL_EXEC_INVOKE_URL = f"{MCP_SERVER_URL}/invoke/sparql-execution"

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
DBPEDIA_ENDPOINT = "https://dbpedia.org/sparql" # Official endpoint

# Agent Configuration
CANDIDATE_LIMIT: int = 5
REQUEST_TIMEOUT: int = 30 # Slightly longer timeout for host calls
DBPEDIA_VERIFY_SSL: bool = True # Set based on your environment needs

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
    logger.critical("Host: LangChain/Ollama libraries not found.")
    sys.exit(1)
except Exception as e:
    logger.critical(f"Host: Connecting to Ollama failed. Details: {e}")
    sys.exit(1)


# --- MCP Client Functions ---

def call_mcp_lookup_service(term: str, source_kg: str, max_results: int) -> List[CandidateInfo]:
    """Calls MCP Server for KG lookup candidates."""
    # ... (Keep implementation from previous step, using DBPEDIA_VERIFY_SSL) ...
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
    # ... (Keep robust error handling from previous step, returning []) ...
    except requests.exceptions.RequestException as e:
        logger.error(f"MCP Client: Error calling KG Lookup Service: {e}")
        return []
    except Exception as e: # Catch JSONDecodeError etc.
        logger.error(f"MCP Client: Unexpected error calling KG Lookup Service: {e}", exc_info=True)
        return []


def call_mcp_sparql_execution(endpoint_url: str, query: str) -> Optional[Dict[str, Any]]:
    """Calls the MCP server's SPARQL execution service."""
    logger.debug(f"Calling MCP Server for SPARQL execution on {endpoint_url}")
    payload = {"endpoint_url": endpoint_url, "query": query}
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

    try:
        response = requests.post(SPARQL_EXEC_INVOKE_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT * 2) # Longer timeout for queries
        response.raise_for_status()
        data = response.json()
        # The server response nests results under 'results' key
        return data.get("results")
    except requests.exceptions.Timeout:
         logger.error(f"MCP Client: Timeout calling SPARQL Execution Service for {endpoint_url}")
         return {"error": "Timeout executing query via MCP server."} # Return error structure
    except requests.exceptions.HTTPError as e:
         # Try to get detail from server's response if possible
         error_detail = f"HTTP Error {e.response.status_code}"
         try:
            server_error = e.response.json()
            error_detail += f": {server_error.get('detail', e.response.reason)}"
         except:
            error_detail += f": {e.response.reason}"
         logger.error(f"MCP Client: HTTP Error calling SPARQL Execution Service: {error_detail}")
         return {"error": error_detail}
    except requests.exceptions.RequestException as e:
        logger.error(f"MCP Client: Error calling SPARQL Execution Service: {e}")
        return {"error": f"Failed to connect to MCP server for execution: {e}"}
    except Exception as e:
        logger.error(f"MCP Client: Unexpected error calling SPARQL Execution Service: {e}", exc_info=True)
        return {"error": f"Unexpected client error during execution call: {e}"}


# --- LLM Functions (Extraction, Disambiguation - Same as before) ---
def extract_focused_terms_llm(natural_query: str) -> List[str]:
    # ... (Keep implementation from previous step) ...
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
    # ... (Keep implementation from previous step) ...
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
            if 0 <= choice_index < len(candidates): return candidates[choice_index]
            elif choice_index == -1: return None
            else: logger.warning(f"LLM chose invalid number {choice_str} for '{term}'."); return None
        else: logger.warning(f"LLM response for '{term}' was not a clear number: '{response}'."); return None
    except Exception as e:
        logger.error(f"Error during LLM disambiguation for '{term}': {e}", exc_info=True); return None

# --- Agent 2: SPARQL Generation (Local LLM) ---
def generate_sparql_queries_llm(nlq: str, resolved_terms: Dict[str, Dict[str, Optional[CandidateInfo]]]) -> Tuple[Optional[str], Optional[str]]:
    """Generates SPARQL queries for Wikidata and DBpedia using LLM."""
    if not llm:
        logger.error("LLM not available for SPARQL generation.")
        return None, None

    logger.info("Step 4: LLM Generating SPARQL queries...")

    # Format resolved terms for the prompt
    context = "Use the following resolved entities and properties:\n"
    if not resolved_terms:
        context += "No specific entities/properties were resolved. Generate general queries based on the NLQ.\n"
    else:
        for term, kgs in resolved_terms.items():
            context += f"- Term: '{term}'\n"
            wd_info = kgs.get('wikidata')
            db_info = kgs.get('dbpedia')
            if wd_info:
                wd_id = wd_info.get('id')
                wd_label = wd_info.get('label')
                prefix = 'wdt:' if wd_id and wd_id.startswith('P') else 'wd:'
                context += f"  - Wikidata: {wd_label} ({prefix}{wd_id})\n"
            if db_info:
                db_uri = db_info.get('uri')
                db_label = db_info.get('label')
                context += f"  - DBpedia: {db_label} (<{db_uri}>)\n"
        context += "\n"

    # SPARQL Generation Prompt
    prompt_template = PromptTemplate.from_template(
        "Based on the natural language query and the resolved entities/properties provided below, generate TWO SPARQL queries: one for Wikidata and one for DBpedia.\n\n"
        "Natural Language Query: {nlq}\n\n"
        "Resolved Context:\n{context}\n\n"
        "Instructions for Queries:\n"
        "1.  **Wikidata Query:**\n"
        "    - Use `wd:` prefix for items (Q...) and `wdt:` for properties (P...). Use the exact IDs provided.\n"
        "    - Select relevant variables needed to answer the query.\n"
        "    - **Crucially: Include labels** for all selected item/property variables using `?varLabel` naming convention.\n"
        "    - Include the standard `SERVICE wikibase:label {{ bd:serviceParam wikibase:language \"[AUTO_LANGUAGE],en\". }}` block before the final closing `}}`.\n"
        "2.  **DBpedia Query:**\n"
        "    - Use full URIs provided in the context (e.g., `<http://dbpedia.org/resource/Berlin>`).\n"
        "    - Use common DBpedia prefixes like `dbo:` (ontology), `dbp:` (property), `dbr:` (resource), `rdfs:`, `rdf:`.\n"
        "    - Select relevant variables.\n"
        "    - Try to select labels using `rdfs:label` where appropriate (e.g., `?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel) = 'en')`).\n"
        "3.  **Output Format:** Provide ONLY the two queries. Separate them clearly using `--- WIKIDATA ---` on one line, followed by the Wikidata query, then `--- DBPEDIA ---` on another line, followed by the DBpedia query. Do not add any other text, comments, or explanations.\n\n"
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

        # Parse the response to split queries
        wd_query = None
        db_query = None

        wd_match = re.search(r"--- WIKIDATA ---\s*```sparql\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if wd_match:
            wd_query = wd_match.group(1).strip()
            logger.info("   LLM Generated Wikidata Query.")
            logger.debug(f"\n{wd_query}\n")


        db_match = re.search(r"--- DBPEDIA ---\s*```sparql\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if db_match:
            db_query = db_match.group(1).strip()
            logger.info("   LLM Generated DBpedia Query.")
            logger.debug(f"\n{db_query}\n")

        if not wd_query and not db_query:
             logger.warning(f"LLM did not generate SPARQL queries in the expected format. Response:\n{response}")
        elif not wd_query:
             logger.warning("LLM failed to generate Wikidata query in expected format.")
        elif not db_query:
             logger.warning("LLM failed to generate DBpedia query in expected format.")

        return wd_query, db_query

    except Exception as e:
        logger.error(f"Error during LLM SPARQL generation: {e}", exc_info=True)
        return None, None

# --- Result Formatting ---
def format_results_for_display(results_dict: Optional[Dict[str, Any]]) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[Any]]:
    """Formats SPARQL results into DataFrame, Triples String, and optionally a Graph object."""
    if not results_dict:
        return None, "No results received.", None
    if results_dict.get("error"):
        return None, f"Execution Error: {results_dict['error']}", None

    bindings = results_dict.get("results", {}).get("bindings", [])
    if not bindings:
        return pd.DataFrame(), "Query executed successfully, but returned no results.", None

    # 1. DataFrame
    processed_data = []
    variables = results_dict.get("head", {}).get("vars", [])
    label_vars = {v + "Label" for v in variables} # Potential label variables

    for binding in bindings:
        row = {}
        processed_simple_vars = set()
        for var in variables:
            if var in processed_simple_vars: continue # Skip if already processed via label

            label_var = f"{var}Label"
            value_info = binding.get(var)
            label_info = binding.get(label_var)

            display_value = None
            raw_value = value_info.get("value") if value_info else None

            # Prioritize label
            if label_info and label_info.get("value"):
                display_value = label_info.get("value")
                processed_simple_vars.add(var)
                processed_simple_vars.add(label_var)
            elif value_info:
                display_value = raw_value
                # Simple cleanup for URIs if no label found
                if value_info.get("type") == "uri":
                    if "wikidata.org/entity/" in display_value:
                         display_value = f"wd:{display_value.rsplit('/', 1)[-1]}"
                    elif "dbpedia.org/resource/" in display_value:
                         display_value = f"dbr:{display_value.rsplit('/', 1)[-1]}"
                    elif "dbpedia.org/ontology/" in display_value:
                         display_value = f"dbo:{display_value.rsplit('/', 1)[-1]}"
                processed_simple_vars.add(var)

            row[var] = display_value # Assign to original var name

        processed_data.append(row)

    df = pd.DataFrame(processed_data)
    # Reorder columns to match original variable order, excluding labels we folded in
    display_vars = [v for v in variables if v not in label_vars or v+'Label' not in df.columns] # Adjust logic if needed
    try:
        df = df[[v for v in display_vars if v in df.columns]] # Show only available display vars
    except KeyError:
        logger.warning("Could not reorder DataFrame columns perfectly.")
        pass # Keep original order if error


    # 2. Triples String (Simple heuristic: look for standard ?s ?p ?o vars)
    triples_str = ""
    graph = None
    if GRAPHING_ENABLED:
        graph = nx.DiGraph()

    subj_var, pred_var, obj_var = None, None, None
    # Try to guess standard triple variables
    if 's' in variables and 'p' in variables and 'o' in variables:
        subj_var, pred_var, obj_var = 's', 'p', 'o'
    elif 'item' in variables and 'property' in variables and 'value' in variables:
         subj_var, pred_var, obj_var = 'item', 'property', 'value'
    elif len(variables) >= 3: # Fallback: assume first 3 are s, p, o
        subj_var, pred_var, obj_var = variables[0], variables[1], variables[2]

    added_triples = set()
    if subj_var and pred_var and obj_var:
        triples_list = []
        for i, row_data in enumerate(processed_data): # Use already processed data with labels
            s = row_data.get(subj_var)
            p = row_data.get(pred_var)
            o = row_data.get(obj_var)

            if s and p and o:
                # Simplify labels/URIs for display
                s_disp = str(s)
                p_disp = str(p).split('/')[-1].split('#')[-1] # Get local name
                o_disp = str(o)

                triple = (s_disp, p_disp, o_disp)
                if triple not in added_triples:
                    triples_list.append(f"({s_disp}) --[{p_disp}]--> ({o_disp})")
                    added_triples.add(triple)
                    if GRAPHING_ENABLED and graph is not None:
                        graph.add_edge(s_disp, o_disp, label=p_disp) # Add edge to graph

        if triples_list:
            triples_str = f"Found {len(triples_list)} unique triples:\n" + "\n".join(triples_list)
        else:
            triples_str = "Could not automatically extract triples (variables might not be ?s ?p ?o)."
    else:
        triples_str = "Could not identify subject/predicate/object variables for triple extraction."


    return df, triples_str, graph


def render_graph_to_base64(graph) -> Optional[str]:
    """Renders a NetworkX graph to a base64 encoded PNG image string."""
    if not GRAPHING_ENABLED or graph is None or graph.number_of_nodes() == 0:
        return None
    try:
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(graph, k=0.5, iterations=50) # Position nodes

        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_size=2000, node_color='skyblue', alpha=0.8)

        # Draw edges
        nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5, arrowsize=15)

        # Draw labels
        nx.draw_networkx_labels(graph, pos, font_size=9)

        # Draw edge labels
        edge_labels = nx.get_edge_attributes(graph, 'label')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, label_pos=0.3)

        plt.title("Result Subgraph")
        plt.axis('off') # Turn off axis

        # Save to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close() # Close the plot to free memory
        buf.seek(0)

        # Encode to base64
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"

    except Exception as e:
        logger.error(f"Failed to render graph: {e}", exc_info=True)
        return None

# --- Main Pipeline Function for Gradio ---
def run_complete_pipeline(natural_language_query: str):
    """Orchestrates the full NLQ -> Resolve -> SPARQL -> Execute -> Format flow."""
    start_time = time.time()
    logger.info(f"\n--- Pipeline Start: '{natural_language_query}' ---")

    # 1. Resolve Entities/Properties (Agent 1 via MCP)
    resolved_terms = resolve_terms_agent_mcp(natural_language_query) # Uses MCP lookup internally
    resolved_terms_json = json.dumps(resolved_terms, indent=2)
    logger.info("Pipeline: Resolution finished.")

    # 2. Generate SPARQL Queries (Agent 2 - Local LLM)
    wd_sparql, db_sparql = generate_sparql_queries_llm(natural_language_query, resolved_terms)
    logger.info("Pipeline: SPARQL generation finished.")

    # 3. Execute Wikidata Query (Agent 2 via MCP)
    wd_results_raw = None
    wd_results_df = pd.DataFrame()
    wd_triples = "Query not generated or execution failed."
    wd_graph_img = None
    if wd_sparql:
        logger.info("Pipeline: Executing Wikidata query via MCP...")
        wd_results_raw = call_mcp_sparql_execution(WIKIDATA_ENDPOINT, wd_sparql)
        wd_results_df, wd_triples, wd_graph = format_results_for_display(wd_results_raw)
        if wd_graph:
            wd_graph_img = render_graph_to_base64(wd_graph)
    else:
        wd_sparql = "# No Wikidata query generated."
        wd_triples = "No Wikidata query generated."


    # 4. Execute DBpedia Query (Agent 2 via MCP)
    db_results_raw = None
    db_results_df = pd.DataFrame()
    db_triples = "Query not generated or execution failed."
    db_graph_img = None
    if db_sparql:
        logger.info("Pipeline: Executing DBpedia query via MCP...")
        db_results_raw = call_mcp_sparql_execution(DBPEDIA_ENDPOINT, db_sparql)
        db_results_df, db_triples, db_graph = format_results_for_display(db_results_raw)
        if db_graph:
            db_graph_img = render_graph_to_base64(db_graph)
    else:
        db_sparql = "# No DBpedia query generated."
        db_triples = "No DBpedia query generated."

    end_time = time.time()
    logger.info(f"--- Pipeline End. Total Time: {end_time - start_time:.2f} seconds ---")

    # Return results for Gradio components
    # Order: Resolved Terms (JSON str), WD SPARQL, DB SPARQL, WD Results (DF), DB Results (DF), WD Triples (str), DB Triples (str), WD Graph (HTML img), DB Graph (HTML img)
    return (
        resolved_terms_json,
        wd_sparql,
        db_sparql,
        wd_results_df if wd_results_df is not None else pd.DataFrame(), # Ensure DF for Gradio
        db_results_df if db_results_df is not None else pd.DataFrame(), # Ensure DF for Gradio
        wd_triples,
        db_triples,
        f'<img src="{wd_graph_img}" alt="Wikidata Graph Visualization">' if wd_graph_img else "Graph visualization not available or no data.",
        f'<img src="{db_graph_img}" alt="DBpedia Graph Visualization">' if db_graph_img else "Graph visualization not available or no data."
    )


# --- Gradio Interface Definition ---
def create_gradio_interface():
    logger.info("Creating Gradio interface...")
    with gr.Blocks(theme=gr.themes.Soft(), title="NLQ to Knowledge Graph Agent") as demo:
        gr.Markdown("# Natural Language Query to Knowledge Graph Agent")
        gr.Markdown("Enter your question below. The system will resolve entities/properties, generate SPARQL queries for Wikidata and DBpedia, execute them, and display the results.")

        with gr.Row():
            nlq_input = gr.Textbox(label="Enter your Natural Language Query:", placeholder="e.g., Who is the prime minister of Canada?", lines=2)

        submit_button = gr.Button("Run Query", variant="primary")

        with gr.Tabs():
            with gr.TabItem("1. Resolved Entities/Properties"):
                 resolved_output = gr.JSON(label="Resolved Terms (from Agent 1)")
            with gr.TabItem("2. Generated SPARQL"):
                 with gr.Row():
                     wd_sparql_output = gr.Code(label="Wikidata SPARQL Query", language="sparql")
                     db_sparql_output = gr.Code(label="DBpedia SPARQL Query", language="sparql")
            with gr.TabItem("3. Wikidata Results"):
                 with gr.Column():
                     wd_results_df_output = gr.DataFrame(label="Wikidata Results Table", wrap=True, height=300)
                     wd_triples_output = gr.Textbox(label="Wikidata Results (Triples)", lines=10, interactive=False)
                     wd_graph_output = gr.HTML(label="Wikidata Graph Visualization") # Use HTML for img tag
            with gr.TabItem("4. DBpedia Results"):
                 with gr.Column():
                     db_results_df_output = gr.DataFrame(label="DBpedia Results Table", wrap=True, height=300)
                     db_triples_output = gr.Textbox(label="DBpedia Results (Triples)", lines=10, interactive=False)
                     db_graph_output = gr.HTML(label="DBpedia Graph Visualization") # Use HTML for img tag

        # Define outputs list in the correct order expected by run_complete_pipeline
        outputs = [
            resolved_output, wd_sparql_output, db_sparql_output,
            wd_results_df_output, db_results_df_output,
            wd_triples_output, db_triples_output,
            wd_graph_output, db_graph_output
        ]

        submit_button.click(
            fn=run_complete_pipeline,
            inputs=nlq_input,
            outputs=outputs,
            api_name="query_knowledge_graphs" # Optional: for API access
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
            outputs=outputs, # Outputs need to be defined for examples too
            fn=run_complete_pipeline, # Function to run for examples
            cache_examples=False, # Disable caching for dynamic results
        )

    return demo


# --- Main Execution ---
if __name__ == "__main__":
    # Perform health check before launching Gradio
    try:
        health_url = f"{MCP_SERVER_URL}/health"
        health_check = requests.get(health_url, timeout=5)
        health_check.raise_for_status()
        logger.info(f"MCP Server health check OK: {health_check.json()}")
    except requests.exceptions.RequestException as e:
        logger.error(f"CRITICAL: MCP Server at {MCP_SERVER_URL} is unreachable!")
        logger.error("Please start the MCP Server in a separate terminal using:")
        logger.error("cd mcp_server && uvicorn server:app --host 0.0.0.0 --port 8100")
        logger.error(f"Error details: {e}")
        sys.exit(1) # Exit if server isn't running

    # Launch the Gradio UI
    interface = create_gradio_interface()
    interface.launch() # Share=False for local access by default
    logger.info("Gradio interface launched. Access it in your browser.")
