from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# Initialize
model = SentenceTransformer('all-MiniLM-L6-v2')
driver = GraphDatabase.driver("localhost")

def semantic_search(query_text, limit=5):
    """Find similar concepts using vector search"""
    query_embedding = model.encode(query_text).tolist()

    with driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes('concept_embeddings', $limit, $embedding)
            YIELD node, score
            RETURN node.rid as rid, 
                   node.label as label, 
                   coalesce(node.definition, '') as definition, 
                   score
            ORDER BY score DESC
        """, embedding=query_embedding, limit=limit)

        return [dict(record) for record in result]

def get_concept_context(rid, depth=2):
    """Get graph context around a concept"""
    with driver.session() as session:
        result = session.run("""
            MATCH path = (c:RadLexConcept {rid: $rid})-[:SUBCLASS_OF*0..%d]-(related:RadLexConcept)
            RETURN DISTINCT 
                related.rid as rid, 
                related.label as label,
                coalesce(related.definition, '') as definition,
                length(path) as distance
            ORDER BY distance
            LIMIT 20
        """ % depth, rid=rid)

        return [dict(record) for record in result]

def graphrag_query(user_question, top_k=3, depth=2):
    """Complete GraphRAG pipeline"""
    print(f"\n{'='*60}")
    print(f"Question: {user_question}")
    print('='*60)

    # 1. Find relevant concepts using vector search
    print("\n Finding relevant concepts...")
    relevant_concepts = semantic_search(user_question, limit=top_k)

    print("\nTop matching concepts:")
    for i, concept in enumerate(relevant_concepts, 1):
        print(f"{i}. [{concept['rid']}] {concept['label']}")
        print(f"   Similarity: {concept['score']:.3f}")
        if concept['definition']:
            print(f"   Definition: {concept['definition'][:150]}...")

    # 2. Get graph context for each concept
    print(f"\n Expanding graph context (depth={depth})...")
    all_context = {}

    for concept in relevant_concepts:
        graph_context = get_concept_context(concept['rid'], depth=depth)
        for item in graph_context:
            if item['rid'] not in all_context:
                all_context[item['rid']] = item

    print(f"Found {len(all_context)} related concepts in the knowledge graph")

    # 3. Build context for LLM
    print("\n Building context for LLM...")
    context_text = "# RadLex Medical Knowledge Base Context\n\n"
    context_text += f"Query: {user_question}\n\n"
    context_text += "## Relevant Medical Concepts:\n\n"

    for concept in relevant_concepts:
        context_text += f"### {concept['label']} (RID: {concept['rid']})\n"
        if concept['definition']:
            context_text += f"{concept['definition']}\n"
        context_text += f"Relevance Score: {concept['score']:.3f}\n\n"

    context_text += "## Related Concepts in Knowledge Graph:\n\n"
    for rid, item in list(all_context.items())[:15]:  # Limit to top 15
        context_text += f"- **{item['label']}** ({item['rid']})"
        if item['definition']:
            context_text += f": {item['definition'][:100]}..."
        context_text += "\n"

    print("CONTEXT TO SEND TO LLM:")
    print(context_text)

    return context_text

# Test queries
if __name__ == "__main__":
    # Example 1: Imaging modality question
    graphrag_query("What imaging modality is best for detecting brain tumors?")

    # Example 2: Anatomical question
    graphrag_query("What are the parts of the cerebellum?")

    # Example 3: Technical imaging question
    graphrag_query("What is the difference between T1 and T2 weighted MRI?")

driver.close()