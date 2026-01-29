"""
Knowledge Graph Extraction and Retrieval
Extracts subject-predicate-object triples for multi-hop reasoning
"""

import logging
import json
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


class KnowledgeGraphExtractor:
    """
    Extract knowledge graph triples from text
    
    Uses LLM to identify entities and relationships, creating structured knowledge
    that enables complex multi-hop queries beyond simple semantic search.
    
    """
    
    def __init__(self, llm, min_confidence: float = 0.5):
        """
        Initialize KG extractor
        
        Args:
            llm: LangChain LLM for extraction
            min_confidence: Minimum confidence threshold for triples
        """
        self.llm = llm
        self.min_confidence = min_confidence
        logger.info(f"Initialized KG extractor (min_confidence={min_confidence})")
    
    def extract_triples(
        self,
        text: str,
        source_doc_id: Optional[int] = None,
        max_triples: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Extract knowledge triples from text
        
        Args:
            text: Text content to analyze
            source_doc_id: Document ID for provenance
            max_triples: Maximum triples to extract
        
        Returns:
            List of triples with subject, relation, object, confidence
        """
        if not text or len(text.strip()) < 50:
            return []
        
        try:
            # Truncate very long text
            text_sample = text[:3000] if len(text) > 3000 else text
            
            extraction_prompt = f"""Extract factual relationships from this text as knowledge graph triples.

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

            messages = [
                SystemMessage(content="You are a knowledge graph extraction system. Extract precise factual relationships."),
                HumanMessage(content=extraction_prompt)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse triples
            triples = []
            for line in response_text.split('\n'):
                line = line.strip()
                if '|' in line and not line.startswith('#'):
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) == 3:
                        subject, relation, obj = parts
                        
                        # Basic validation
                        if (len(subject) > 2 and len(relation) > 2 and len(obj) > 2 and
                            len(subject) < 100 and len(relation) < 50 and len(obj) < 100):
                            
                            triple = {
                                'subject': subject,
                                'relation': relation,
                                'object': obj,
                                'source_doc_id': source_doc_id,
                                'confidence': 0.8,  # Default confidence
                                'metadata': {}
                            }
                            triples.append(triple)
            
            logger.info(f"Extracted {len(triples)} knowledge triples")
            return triples[:max_triples]
            
        except Exception as e:
            logger.error(f"Triple extraction failed: {e}")
            return []
    
    def store_triples(
        self,
        triples: List[Dict[str, Any]],
        agent_id: str
    ) -> int:
        """
        Store triples in knowledge_graph table
        
        Args:
            triples: List of extracted triples
            agent_id: Agent identifier
        
        Returns:
            Number of triples stored
        """
        if not triples:
            return 0
        
        try:
            from database.connection import db
            
            stored_count = 0
            with db.get_cursor() as cur:
                for triple in triples:
                    if triple.get('confidence', 0) < self.min_confidence:
                        continue
                    
                    cur.execute("""
                        INSERT INTO knowledge_graph
                        (agent_id, subject, relation, object, source_doc_id, confidence, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (
                        agent_id,
                        triple['subject'],
                        triple['relation'],
                        triple['object'],
                        triple.get('source_doc_id'),
                        triple.get('confidence', 0.8),
                        json.dumps(triple.get('metadata', {}))
                    ))
                    stored_count += 1
                
                cur.connection.commit()
            
            logger.info(f"Stored {stored_count} triples in knowledge graph")
            return stored_count
            
        except Exception as e:
            logger.error(f"Failed to store triples: {e}")
            return 0


class KnowledgeGraphRetriever:
    """
    Multi-hop retrieval using knowledge graph (Paper-compliant)
    
    Traverses knowledge graph to find indirect relationships,
    enabling complex reasoning beyond direct semantic similarity.
    """
    
    def __init__(self, max_hops: int = 2, min_confidence: float = 0.5):
        """
        Initialize KG retriever
        
        Args:
            max_hops: Maximum traversal depth
            min_confidence: Minimum triple confidence
        """
        self.max_hops = max_hops
        self.min_confidence = min_confidence
        logger.info(f"Initialized KG retriever (max_hops={max_hops})")
    
    def find_related_entities(
        self,
        agent_id: str,
        entity: str,
        max_hops: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Find entities related through knowledge graph traversal
        
        Args:
            agent_id: Agent identifier
            entity: Starting entity
            max_hops: Maximum hops (overrides default)
        
        Returns:
            List of related entities with paths
        """
        hops = max_hops if max_hops is not None else self.max_hops
        
        try:
            from database.connection import db
            
            related = []
            visited = set()
            current_entities = [entity]
            
            for hop in range(hops):
                if not current_entities:
                    break
                
                next_entities = []
                
                with db.get_cursor() as cur:
                    for current in current_entities:
                        if current in visited:
                            continue
                        visited.add(current)
                        
                        # Find outgoing relations
                        cur.execute("""
                            SELECT subject, relation, object, source_doc_id, confidence
                            FROM knowledge_graph
                            WHERE agent_id = %s
                              AND (subject ILIKE %s OR object ILIKE %s)
                              AND confidence >= %s
                            LIMIT 50
                        """, (agent_id, f"%{current}%", f"%{current}%", self.min_confidence))
                        
                        for row in cur.fetchall():
                            subject, relation, obj, doc_id, conf = row
                            
                            # Determine next entity
                            if current.lower() in subject.lower():
                                next_entity = obj
                                direction = 'forward'
                            else:
                                next_entity = subject
                                direction = 'backward'
                            
                            related.append({
                                'entity': next_entity,
                                'relation': relation,
                                'direction': direction,
                                'source_doc_id': doc_id,
                                'confidence': float(conf),
                                'hop_distance': hop + 1,
                                'path': f"{current} --[{relation}]--> {next_entity}" if direction == 'forward' else f"{next_entity} --[{relation}]--> {current}"
                            })
                            
                            next_entities.append(next_entity)
                
                current_entities = list(set(next_entities))
            
            logger.info(f"Found {len(related)} related entities for '{entity}' ({len(visited)} nodes visited)")
            return related
            
        except Exception as e:
            logger.error(f"KG traversal failed: {e}")
            return []
    
    def query_knowledge_graph(
        self,
        agent_id: str,
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Query knowledge graph with natural language
        
        Args:
            agent_id: Agent identifier
            query: Natural language query
        
        Returns:
            Relevant triples and entities
        """
        try:
            # Clean query: remove metadata prefixes and punctuation
            import string
            clean_query = query.replace("Refined Query:", "").replace("Original:", "").strip('"\'')
            clean_query = clean_query.translate(str.maketrans('', '', string.punctuation))
            
            # Extract potential entities from query (simple tokenization)
            words = clean_query.lower().split()
            # Basic stopwords to skip
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now', 'quais', 'sao', 'como', 'que', 'para', 'com', 'uma', 'das', 'dos', 'pela', 'pelo', 'esta', 'esse', 'essa'}
            entities = [w for w in words if len(w) > 3 and w not in stopwords]
            
            all_results = []
            for entity in entities[:3]:  # Limit to top 3 query terms
                results = self.find_related_entities(agent_id, entity, max_hops=1)
                all_results.extend(results)
            
            # Deduplicate and sort by confidence
            seen = set()
            unique_results = []
            for r in all_results:
                key = (r['entity'], r['relation'])
                if key not in seen:
                    seen.add(key)
                    unique_results.append(r)
            
            unique_results.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.info(f"KG query returned {len(unique_results)} unique results")
            return unique_results[:20]
            
        except Exception as e:
            logger.error(f"KG query failed: {e}")
            return []
