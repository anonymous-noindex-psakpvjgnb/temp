# ... (Imports, Logging, Config, Type Defs - largely same, update types if needed) ...
import requests
import json
import logging
import sys
import time
import re
import pandas as pd
import gradio as gr
from typing import List, Dict, Optional, Any, Tuple, Literal, TypedDict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MCPHostAgentV5")
# ... (Config Constants remain largely the same) ...
OLLAMA_BASE_URL="http://localhost:11434"; OLLAMA_MODEL="deepseek-llm:latest"
MCP_SERVER_URL="http://localhost:8100"; KG_LOOKUP_INVOKE_URL=f"{MCP_SERVER_URL}/invoke/knowledge-graph-lookup"; SPARQL_EXEC_INVOKE_URL=f"{MCP_SERVER_URL}/invoke/sparql-execution"
WIKIDATA_ENDPOINT="https://query.wikidata.org/sparql"; DBPEDIA_ENDPOINT="https://dbpedia.org/sparql"; WIKIDATA_API_ENDPOINT="https://www.wikidata.org/w/api.php"
CANDIDATE_LIMIT=5; REQUEST_TIMEOUT=30; DBPEDIA_VERIFY_SSL=True; USER_AGENT="AdvancedResolverAgentV5/Host"

# --- Type Definitions ---
CandidateInfo = Dict[str, Optional[str]]
# Removed ExtractedTerm TypedDict as structure changes

class NlqAnalysisResult(TypedDict):
    entities: List[str] # List of entity strings
    target_relation_phrase: Optional[str] # The core relation/attribute phrase

class ResolvedTermData(TypedDict):
    wikidata: Optional[CandidateInfo]
    dbpedia: Optional[CandidateInfo]

ResolvedEntitiesMap = Dict[str, ResolvedTermData] # Maps entity term string to resolved data
ResolvedPropertyInfo = ResolvedTermData # Reuse structure for the single resolved property

# --- Initialize LLM (Same) ---
try:
    # ... (LLM Init same as V4) ...
    from langchain_community.llms import Ollama
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
    llm = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    logger.info(f"Host: Successfully connected to Ollama model: {OLLAMA_MODEL}")
except ImportError: logger.critical("Host: LangChain/Ollama libraries not found."); sys.exit(1)
except Exception as e: logger.critical(f"Host: Connecting to Ollama failed. Details: {e}"); sys.exit(1)


# --- MCP Client Functions (call_mcp_lookup_service, call_mcp_sparql_execution - Same) ---
# ... (Keep implementations from V3/V4) ...
def call_mcp_lookup_service(term: str, source_kg: str, max_results: int) -> List[CandidateInfo]: #...(same)...
def call_mcp_sparql_execution(endpoint_url: str, query: str, context: str = "SPARQL Execution") -> Optional[Dict[str, Any]]: #...(same)...

# --- LLM Functions ---

# --- NEW/REVISED: Initial NLQ Analysis ---
def analyze_nlq_structure_llm(nlq: str) -> Optional[NlqAnalysisResult]:
    """
    REVISED: Analyzes NLQ to extract entities and the target relation/attribute phrase.
    """
    if not llm: logger.error("LLM not available..."); return None
    logger.info(f"Step 1: LLM Analyzing NLQ structure: '{nlq}'")

    # Using StrOutputParser first, then parsing JSON, as JSON output can be flaky
    prompt = PromptTemplate(
        template=(
            "Analyze the user's question. Identify:\n"
            "1. The main specific entities involved (people, places, organizations, named concepts).\n"
            "2. The single, core relationship or attribute being asked about (e.g., 'president', 'spouse', 'capital', 'population', 'director', 'boiling point', 'chemical formula'). If the question is just asking for a description (e.g., 'Tell me about Berlin'), output null for the relation.\n\n"
            "Output ONLY a JSON object with exactly two keys:\n"
            "- \"entities\": A JSON list of the entity strings found.\n"
            "- \"target_relation_phrase\": A JSON string representing the core relationship/attribute, or null if none is clearly identified.\n\n"
            "Example Question: 'Who is the wife of Barack Obama?'\n"
            "Example Output: {{\"entities\": [\"Barack Obama\"], \"target_relation_phrase\": \"wife\"}}\n\n"
            "Example Question: 'What is the population of Berlin?'\n"
            "Example Output: {{\"entities\": [\"Berlin\"], \"target_relation_phrase\": \"population\"}}\n\n"
            "Example Question: 'Tell me about the Eiffel Tower.'\n"
            "Example Output: {{\"entities\": [\"Eiffel Tower\"], \"target_relation_phrase\": null}}\n\n"
            "User Question: \"{query}\"\n\n"
            "JSON Output:"
        ),
        input_variables=["query"],
    )
    chain = prompt | llm | StrOutputParser()

    try:
        response_str = chain.invoke({"query": nlq})
        logger.debug(f"LLM Raw NLQ Analysis Output: {response_str}")
        # Attempt to parse the string response as JSON
        try:
            # Clean potential markdown ```json ... ``` and other noise
            match = re.search(r'\{.*\}', response_str, re.DOTALL)
            if not match:
                logger.error(f"Could not find JSON object in LLM NLQ analysis response: {response_str}")
                return None
            json_str = match.group(0)

            data = json.loads(json_str)

            # Validate structure
            if isinstance(data, dict) and \
               'entities' in data and isinstance(data['entities'], list) and \
               'target_relation_phrase' in data: # Key must exist, value can be null

                # Ensure entities are strings
                entities_list = [str(e) for e in data['entities'] if isinstance(e, (str, int, float))] # Basic validation
                relation_phrase = data['target_relation_phrase']
                if relation_phrase is not None and not isinstance(relation_phrase, str):
                    logger.warning(f"target_relation_phrase was not a string or null, converting: {relation_phrase}")
                    relation_phrase = str(relation_phrase) # Convert just in case

                result: NlqAnalysisResult = {
                    "entities": entities_list,
                    "target_relation_phrase": relation_phrase if isinstance(relation_phrase, str) else None # Ensure null if not string
                }
                logger.info(f"   LLM NLQ Analysis Result: {result}")
                return result
            else:
                logger.error(f"LLM NLQ analysis output JSON lacks required structure: {data}")
                return None
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse LLM NLQ analysis output as JSON: {json_err}. Response: {response_str}")
            return None
    except Exception as e:
        logger.error(f"   Error during LLM NLQ analysis: {e}", exc_info=True)
        return None

# --- Specific KG Search Functions (find_wikidata_properties, find_dbpedia_properties - Same as V4) ---
# These now directly use the 'target_relation_phrase'
def find_wikidata_properties(term_phrase: str, limit: int) -> List[CandidateInfo]: #...(same V4 implementation)...
def find_dbpedia_properties(term_phrase: str, limit: int) -> List[CandidateInfo]: #...(same V4 implementation)...

# --- LLM Disambiguation (Updated Context) ---
def disambiguate_term_llm(
    nlq: str,
    term_being_resolved: str, # Can be entity or property phrase
    is_entity: bool, # Flag to know what we are resolving
    candidates: List[CandidateInfo],
    source_kg: str,
    target_relation_phrase: Optional[str] = None, # Context for entities
    resolved_entities: Optional[ResolvedTermsMap] = None # Context for properties
) -> Optional[CandidateInfo]:
    """Uses LLM for disambiguation, using relation for entities, entities for relations."""
    if not candidates: return None
    if len(candidates) == 1: logger.info(f"   Only one candidate for '{term_being_resolved}', selecting directly."); return candidates[0]
    if not llm: logger.error(f"LLM not available for disambiguation of '{term_being_resolved}'."); return None

    logger.info(f"   LLM Disambiguating {'ENTITY' if is_entity else 'PROPERTY'} '{term_being_resolved}' for query '{nlq}' using {len(candidates)} candidates from {source_kg}...")

    # Prepare candidate list text (same as before)
    candidate_text = "" #...(same loop)...
    for i, cand in enumerate(candidates): ident = cand.get('id') or cand.get('uri'); label = cand.get('label'); desc = cand.get('description'); candidate_text += f"{i+1}. Id: {ident}\n   Lbl: {label}\n   Desc: {desc}\n\n"

    # Prepare specific context based on what's being resolved
    extra_context = ""
    if is_entity and target_relation_phrase:
        extra_context = f"Context: The user is asking about the relation '{target_relation_phrase}' involving this entity.\nChoose the entity sense that fits this relationship.\n"
    elif not is_entity and resolved_entities: # Resolving property phrase
        extra_context += "Context Entities:\n" #...(same loop as V4 to list resolved entities)...
        for entity_term, entity_data in resolved_entities.items():
             entity_kg_info = entity_data.get(source_kg.lower());
             if entity_kg_info: ident = entity_kg_info.get('id') or entity_kg_info.get('uri'); label = entity_kg_info.get('label'); extra_context += f"- '{entity_term}': {label} ({ident})\n"
        extra_context += "\nChoose the property that best represents the relation requested in the query and connects/describes these entities.\n"

    # Disambiguation Prompt (Simplified, more direct)
    prompt_template = PromptTemplate.from_template(
        "Select the most relevant item from the list below for the term '{term}' based on the user's query.\n"
        "Query: \"{nlq}\"\n"
        "Term to Disambiguate: \"{term}\"\n\n"
        "{extra_context}" # Context based on whether it's entity or property
        "Candidates from {source_kg}:\n{candidate_list}\n"
        "Instructions: Consider the query and context. Choose the best fit.\n\n"
        "Which candidate number is most appropriate? Respond ONLY with the number (e.g., '1', '3'). If none fit, respond with '0'."
    )
    chain = prompt_template | llm | StrOutputParser()
    try:
        response = chain.invoke({
            "nlq": nlq, "term": term_being_resolved,
            "source_kg": source_kg, "candidate_list": candidate_text,
            "extra_context": extra_context
        })
        # ... (Use robust number parsing logic from V4) ...
        response_text = response.strip(); choice_str = None; match = re.search(r'\b(\d+)\b', response_text) # ...(same parsing logic)...
        if match: choice_str = match.group(1); logger.debug(f"Disambiguation: Found number '{choice_str}' using regex.")
        else: digits = ''.join(filter(str.isdigit, response_text));
        if digits: choice_str = digits[0]; logger.debug(f"Disambiguation: Found digits '{digits}', taking first '{choice_str}' (fallback).")
        if choice_str and choice_str.isdigit(): #...(same index checking logic)...
             choice_index = int(choice_str) - 1
             if 0 <= choice_index < len(candidates): logger.info(f"LLM selected candidate {choice_index + 1}."); return candidates[choice_index]
             elif choice_index == -1: logger.info("LLM indicated no suitable candidate."); return None
             else: logger.warning(f"LLM chose out-of-bounds number {choice_str}."); return None
        else: logger.warning(f"Failed to parse number from LLM disambiguation: '{response_text}'."); return None
    except Exception as e:
        logger.error(f"Error during LLM disambiguation for '{term_being_resolved}': {e}", exc_info=True); return None


# --- SPARQL Generation (Simplified Prompt) ---
def generate_sparql_queries_llm_v5(nlq: str, resolved_entities: ResolvedTermsMap, resolved_property: Optional[ResolvedPropertyInfo]) -> Tuple[Optional[str], Optional[str]]:
    """Generates SPARQL using resolved entities and the single target property."""
    if not llm: logger.error("LLM not available..."); return None, None
    logger.info("Step 4: LLM Generating SPARQL queries (V5 - Direct Property)...")

    # Format context
    entity_context = "Entities:\n"
    if not resolved_entities: entity_context += "  (None found)\n"
    else:
        for term, data in resolved_entities.items():
            wd_info = data.get('wikidata'); db_info = data.get('dbpedia')
            entity_context += f"- '{term}':\n"
            if wd_info: entity_context += f"  - WD: {wd_info.get('label')} (wd:{wd_info.get('id')})\n"
            if db_info: entity_context += f"  - DB: {db_info.get('label')} (<{db_info.get('uri')}>)\n"

    prop_context = "Target Relation/Attribute:\n"
    if not resolved_property: prop_context += "  (None identified or resolved)\n"
    else:
        wd_info = resolved_property.get('wikidata'); db_info = resolved_property.get('dbpedia')
        if wd_info: prop_context += f"  - WD: {wd_info.get('label')} (wdt:{wd_info.get('id')})\n" # Assume property, use wdt:
        if db_info: prop_context += f"  - DB: {db_info.get('label')} (<{db_info.get('uri')}>)\n"

    # Simplified SPARQL Generation Prompt
    prompt_template = PromptTemplate.from_template(
        "Generate TWO SPARQL queries (Wikidata, DBpedia) to answer the query using the provided entities and target relation.\n\n"
        "Natural Language Query: {nlq}\n\n"
        "Context:\n{entity_context}{prop_context}\n"
        "Instructions:\n"
        "1. Construct queries connecting the entities using the target relation/attribute.\n"
        "2. If the relation is missing, try to formulate a query based only on the entities (e.g., retrieve basic facts or description).\n"
        "3. For Wikidata: Use `wd:`/`wdt:`. Include labels (`?varLabel`, `SERVICE wikibase:label`).\n"
        "4. For DBpedia: Use full URIs. Include `rdfs:label` (`LANG='en'`).\n"
        "5. Structure the output ONLY as two queries separated by `--- WIKIDATA ---` and `--- DBPEDIA ---` in ```sparql blocks.\n\n"
        "--- WIKIDATA ---\n```sparql\n# Wikidata Query Here\n```\n\n"
        "--- DBPEDIA ---\n```sparql\n# DBpedia Query Here\n```\n"
    )
    chain = prompt_template | llm | StrOutputParser()
    try:
        response = chain.invoke({
            "nlq": nlq, "entity_context": entity_context, "prop_context": prop_context
        })
        # ... (Regex parsing remains the same) ...
        wd_query, db_query = None, None; #...(same regex parsing)...
        wd_match = re.search(r"--- WIKIDATA ---\s*```sparql\n(.*?)```", response, re.DOTALL | re.IGNORECASE); db_match = re.search(r"--- DBPEDIA ---\s*```sparql\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if wd_match: wd_query = wd_match.group(1).strip(); logger.info("   LLM Generated Wikidata Query (V5).")
        if db_match: db_query = db_match.group(1).strip(); logger.info("   LLM Generated DBpedia Query (V5).")
        #...(warning logs)...
        return wd_query, db_query
    except Exception as e: logger.error(f"Error during LLM SPARQL generation V5: {e}", exc_info=True); return None, None


# --- Fallback Funcs (fetch_wikidata_abstract, fetch_dbpedia_abstract, answer_from_abstracts_llm - Same as V4) ---
# ... (Keep V3/V4 implementations) ...
def fetch_wikidata_abstract(qid: str) -> Optional[str]: #...(same)...
def fetch_dbpedia_abstract(uri: str) -> Optional[str]: #...(same)...
def answer_from_abstracts_llm(nlq: str, wd_abstract: Optional[str], db_abstract: Optional[str]) -> str: #...(same)...

# --- Result Formatting (format_results_for_display, render_graph_to_base64 - Same as V4) ---
# ... (Keep V3/V4 implementations) ...
def format_results_for_display(results_dict: Optional[Dict[str, Any]]) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[Any]]: #...(same)...
def render_graph_to_base64(graph) -> Optional[str]: #...(same)...

# --- Agent 1 Internal Orchestration (REVISED) ---
def resolve_terms_agent_v5(natural_query: str, analysis: NlqAnalysisResult) -> Tuple[ResolvedEntitiesMap, Optional[ResolvedPropertyInfo]]:
    """REVISED: Resolves entities and the single target property."""
    resolved_entities: ResolvedEntitiesMap = {}
    resolved_property: Optional[ResolvedPropertyInfo] = None

    if not analysis:
        logger.warning("Agent 1 (V5): NLQ Analysis failed, cannot resolve terms.")
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


# --- Main Pipeline Function for Gradio (REVISED) ---
def run_complete_pipeline_v5(natural_language_query: str):
    """Orchestrates the full V5 flow: NLQ Analysis -> Resolve E/P -> SPARQL -> Exec -> Format/Fallback/Summarize."""
    start_time = time.time()
    logger.info(f"\n--- Pipeline V5 Start: '{natural_language_query}' ---")

    # 1. Analyze NLQ Structure (Local LLM)
    nlq_analysis = analyze_nlq_structure_llm(natural_language_query)
    target_relation_phrase_display = nlq_analysis.get("target_relation_phrase", "N/A") if nlq_analysis else "Analysis Failed"

    # 2. Resolve Entities & Target Property (Agent 1 via MCP + Local LLM Disambiguation)
    resolved_entities, resolved_property = resolve_terms_agent_v5(natural_language_query, nlq_analysis)
    # Combine for JSON display
    combined_resolved = {
        "entities": resolved_entities,
        "target_property": resolved_property if resolved_property else "Not Resolved"
    }
    resolved_terms_json = json.dumps(combined_resolved, indent=2)
    logger.info("Pipeline V5: Resolution finished.")

    # 3. Generate SPARQL Queries (Agent 2 - Local LLM)
    wd_sparql, db_sparql = generate_sparql_queries_llm_v5(natural_language_query, resolved_entities, resolved_property)
    logger.info("Pipeline V5: SPARQL generation finished.")

    # 4. Execute Queries & Format Results (Same as V4)
    # ... (wd_ok, db_ok logic remains the same) ...
    wd_results_raw, db_results_raw = None, None; wd_results_df, db_results_df = pd.DataFrame(), pd.DataFrame(); wd_triples, db_triples = "Exec pending.", "Exec pending."; wd_graph_img, db_graph_img = None, None; wd_ok, db_ok = False, False
    if wd_sparql: wd_results_raw = call_mcp_sparql_execution(WIKIDATA_ENDPOINT, wd_sparql, "WD Query Exec"); wd_results_df, wd_triples, wd_graph = format_results_for_display(wd_results_raw); #... (graph rendering, set wd_ok)...
    else: wd_sparql = "# N/A"; wd_triples = "Query not generated."
    if db_sparql: db_results_raw = call_mcp_sparql_execution(DBPEDIA_ENDPOINT, db_sparql, "DB Query Exec"); db_results_df, db_triples, db_graph = format_results_for_display(db_results_raw); #... (graph rendering, set db_ok)...
    else: db_sparql = "# N/A"; db_triples = "Query not generated."


    # 5. Determine Final Answer: Summarize or Fallback (Same logic as V4)
    consolidated_answer = ""
    if wd_ok or db_ok: #...(Summarize successful results)...
    else: #...(Attempt fallback using abstracts, find primary entity from resolved_entities)...
        # ...(Fallback logic remains the same as V4)...


    end_time = time.time()
    logger.info(f"--- Pipeline V5 End. Total Time: {end_time - start_time:.2f} seconds ---")

    # Return results for Gradio - Replace Intent with Target Relation Phrase
    return (
        target_relation_phrase_display, # <-- Changed from Intent
        resolved_terms_json,
        wd_sparql, db_sparql,
        wd_results_df if wd_results_df is not None else pd.DataFrame(),
        db_results_df if db_results_df is not None else pd.DataFrame(),
        wd_triples, db_triples,
        f'<img src="{wd_graph_img}" alt="Wikidata Graph Viz">' if wd_graph_img else "Graph not available.",
        f'<img src="{db_graph_img}" alt="DBpedia Graph Viz">' if db_graph_img else "Graph not available.",
        consolidated_answer
    )

# --- Gradio Interface Definition (Updated) ---
def create_gradio_interface_v5():
    logger.info("Creating Gradio interface V5...")
    with gr.Blocks(theme=gr.themes.Soft(), title="NLQ to Knowledge Graph Agent v5") as demo:
        gr.Markdown("# NLQ to Knowledge Graph Agent (v5 - Direct Relation Focus)")
        gr.Markdown("Enter query -> Analyze Structure (Entities/Relation) -> Resolve -> Generate/Execute SPARQL -> Summarize/Fallback.")
        with gr.Row(): nlq_input = gr.Textbox(label="Enter your Natural Language Query:", placeholder="e.g., Who is the prime minister of Canada?", lines=2, scale=4); submit_button = gr.Button("Run Query", variant="primary", scale=1)
        gr.Markdown("## Final Consolidated Answer"); consolidated_output = gr.Markdown(label="Consolidated Answer")

        with gr.Accordion("See Detailed Steps & Results", open=False):
            with gr.Tabs():
                # Changed Tab 0 from Intent to Relation Phrase
                with gr.TabItem("0. NLQ Analysis"):
                    relation_output = gr.Textbox(label="Identified Target Relation/Attribute Phrase", interactive=False) # Use Textbox

                with gr.TabItem("1. Resolved Terms"): resolved_output = gr.JSON(label="Resolved Entities & Property")
                with gr.TabItem("2. SPARQL Queries"): # ...(Same layout as V4)...
                with gr.TabItem("3. Wikidata Results"): # ...(Same layout as V4)...
                with gr.TabItem("4. DBpedia Results"): # ...(Same layout as V4)...

        # Update outputs list order to match run_complete_pipeline_v5 return
        outputs = [
            relation_output, # <-- Changed from intent_output
            resolved_output, wd_sparql_output, db_sparql_output,
            wd_results_df_output, db_results_df_output,
            wd_triples_output, db_triples_output,
            wd_graph_output, db_graph_output,
            consolidated_output
        ]

        submit_button.click( fn=run_complete_pipeline_v5, inputs=nlq_input, outputs=outputs, api_name="query_kg_agent_v5" )
        gr.Examples( examples=["Who is the president of India?", "What is the population of Paris, France?", "Which actors starred in the movie Inception?", "Show me the chemical formula for caffeine.", "What are the main ingredients in Aspirin?"], inputs=nlq_input, fn=run_complete_pipeline_v5, )
    return demo

# --- Main Execution (Updated function call) ---
if __name__ == "__main__":
    # ... (Keep health check) ...
    interface = create_gradio_interface_v5() # Call new function
    interface.launch()
    logger.info("Gradio interface V5 launched.")
