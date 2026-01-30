"""
Knowledge Graph Extraction Prompts
Prompts for extracting triples from text
"""

KG_EXTRACTION_SYSTEM_PROMPT = "You are a knowledge graph extraction system. Extract precise factual relationships."

KG_EXTRACTION_PROMPT_TEMPLATE = """Extract factual relationships from this text as knowledge graph triples.

Text:
{text_sample}

Instructions:
1. Identify key entities (people, organizations, concepts, technologies)
2. Extract relationships between entities
3. Format as: Subject | Relation | Object
4. Focus on factual, verifiable relationships
5. Return up to {max_triples} most important triples

Example:
Python | is_a | programming_language
Python | used_for | web_development
Django | is_framework_for | Python

Now extract triples from the text above. One triple per line.
Format: Subject | Relation | Object"""
