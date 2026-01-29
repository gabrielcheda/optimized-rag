QUERY_REWRITER_PROMPT="""Perform the following query transformations ONLY where indicated:
	- SIMPLIFY: {simplify}
	- CONTEXTUALIZE: {contextualize}
	- REFORMULATE: {reformulate}
	- CORRECT: {correct}

	Original Query: "{query}"
	Recent History: {history_text}

	Guidelines:
	1. Maintain the ORIGINAL language of the query.
	2. If a transformation is not needed, return null for that field.
	3. For REFORMULATE, focus on keywords that would help a vector database.
	"""