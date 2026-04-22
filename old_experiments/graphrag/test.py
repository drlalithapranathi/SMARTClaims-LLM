from neo4j import GraphDatabase

driver = GraphDatabase.driver("localhost")

print("Cleaning up old embeddings and indexes...")

with driver.session() as session:
    # 1. Drop the old vector index
    print("Dropping old vector index...")
    try:
        session.run("DROP INDEX concept_embeddings IF EXISTS")
        print("  Old index dropped")
    except Exception as e:
        print(f"  Note: {e}")

    # 2. Remove old embedding properties from any remaining nodes
    print("Removing old embedding properties...")
    result = session.run("""
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        REMOVE n.embedding
        RETURN count(n) as removed_count
    """)
    removed = result.single()["removed_count"]
    print(f" Removed embeddings from {removed} nodes")

    # 3. Check if there are any old Class nodes (from the failed import)
    print("Checking for old Class nodes...")
    result = session.run("""
        MATCH (n:Class)
        RETURN count(n) as old_nodes
    """)
    old_nodes = result.single()["old_nodes"]

    if old_nodes > 0:
        print(f"  Found {old_nodes} old Class nodes. Deleting...")
        session.run("MATCH (n:Class) DETACH DELETE n")
        print("  ✓ Old Class nodes deleted")
    else:
        print("  ✓ No old Class nodes found")

    # 4. Verify RadLexConcept nodes are clean
    result = session.run("""
        MATCH (c:RadLexConcept)
        WHERE c.embedding IS NOT NULL
        RETURN count(c) as concepts_with_embeddings
    """)
    remaining = result.single()["concepts_with_embeddings"]
    print(f"\nRadLexConcept nodes with embeddings: {remaining}")

    # 5. Count total RadLexConcept nodes
    result = session.run("""
        MATCH (c:RadLexConcept)
        RETURN count(c) as total_concepts
    """)
    total = result.single()["total_concepts"]
    print(f"Total RadLexConcept nodes ready for embedding: {total}")

print("\n Cleanup complete. Ready to create fresh embeddings.")
driver.close()