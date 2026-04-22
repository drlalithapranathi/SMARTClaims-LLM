from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import time

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

driver = GraphDatabase.driver("localhost")

def create_vector_index(tx):
    # Create vector index for semantic search
    query = """
    CREATE VECTOR INDEX concept_embeddings IF NOT EXISTS
    FOR (c:RadLexConcept)
    ON c.embedding
    OPTIONS {indexConfig: {
        `vector.dimensions`: 384,
        `vector.similarity_function`: 'cosine'
    }}
    """
    tx.run(query)

def get_concepts_without_embeddings(tx, limit=100):
    query = """
    MATCH (c:RadLexConcept)
    WHERE c.embedding IS NULL AND c.label IS NOT NULL
    RETURN c.rid as rid, c.label as label, 
           coalesce(c.definition, '') as definition
    LIMIT $limit
    """
    result = tx.run(query, limit=limit)
    return [dict(record) for record in result]

def update_embeddings(tx, embeddings_batch):
    query = """
    UNWIND $batch AS item
    MATCH (c:RadLexConcept {rid: item.rid})
    SET c.embedding = item.embedding
    """
    tx.run(query, batch=embeddings_batch)

print("Creating vector index...")
with driver.session() as session:
    session.execute_write(create_vector_index)

print("Adding vector embeddings...")
total_processed = 0
start_time = time.time()

with driver.session() as session:
    while True:
        # Get batch of concepts without embeddings
        concepts = session.execute_read(get_concepts_without_embeddings, limit=100)

        if not concepts:
            break

        # Generate embeddings
        embeddings_batch = []
        for concept in concepts:
            # Combine label and definition for richer embedding
            text = concept['label']
            if concept['definition']:
                text += ": " + concept['definition']

            embedding = model.encode(text)
            embeddings_batch.append({
                "rid": concept['rid'],
                "embedding": embedding.tolist()
            })

        # Update in Neo4j
        session.execute_write(update_embeddings, embeddings_batch)
        total_processed += len(embeddings_batch)

        elapsed = time.time() - start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        print(f"Processed {total_processed} embeddings... ({rate:.1f} concepts/sec)")

print(f"\n Embeddings complete! Processed {total_processed} concepts in {elapsed:.1f} seconds")
driver.close()