"""
Query Routing Prompts
Prompts for routing queries to appropriate data sources
"""

ROUTING_PROMPT_TEMPLATE = """Route the query to the best data source.

Available sources:
{sources_list}

Guidelines:
- DOCUMENTS: PRIORITY - always try first for uploaded files, knowledge base content
- ARCHIVAL_MEMORY: personal info, learned facts, past conversations
- CONVERSATION_HISTORY: context from current conversation
- HYBRID: when multiple sources beneficial
- WEB_SEARCH: LAST RESORT - only for current events, real-time data not in documents

IMPORTANT: Always prefer DOCUMENTS over WEB_SEARCH unless query explicitly asks for current/recent information.
Respond:
SOURCE: [source name]
CONFIDENCE: [0.0-1.0]
REASONING: [explanation]"""

SOURCE_DESCRIPTIONS = {
    "ARCHIVAL_MEMORY": "Long-term memory from past interactions",
    "DOCUMENTS": "Recently uploaded documents (PDFs, texts)",
    "WEB_SEARCH": "Real-time web search for current info",
    "CONVERSATION_HISTORY": "Recent conversation context",
    "HYBRID": "Multiple sources combined"
}
