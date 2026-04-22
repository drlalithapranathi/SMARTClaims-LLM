from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef
from neo4j import GraphDatabase

print("Loading RadLex 4.2...")
g = Graph()
g.parse("Radlex.owl", format="xml")
print(f"Loaded {len(g)} triples")

# Define RadLex namespace
RADLEX = Namespace("http://www.radlex.org/RID/")
RADLEX_PROPS = Namespace("http://radlex.org/RID/")

# Connect to Neo4j
driver = GraphDatabase.driver("localhost")

def clear_database(tx):
    tx.run("MATCH (n) DETACH DELETE n")

def create_constraints(tx):
    tx.run("CREATE CONSTRAINT concept_uri IF NOT EXISTS FOR (c:RadLexConcept) REQUIRE c.uri IS UNIQUE")
    tx.run("CREATE CONSTRAINT concept_rid IF NOT EXISTS FOR (c:RadLexConcept) REQUIRE c.rid IS UNIQUE")

def import_concepts(tx, concepts_batch):
    query = """
    UNWIND $concepts AS concept
    MERGE (c:RadLexConcept {uri: concept.uri})
    SET c.rid = concept.rid,
        c.label = concept.label,
        c.preferredName = concept.preferredName,
        c.definition = concept.definition,
        c.synonyms = concept.synonyms,
        c.fmaid = concept.fmaid
    """
    tx.run(query, concepts=concepts_batch)

def import_relationships(tx, rels_batch):
    query = """
    UNWIND $rels AS rel
    MATCH (child:RadLexConcept {uri: rel.child})
    MATCH (parent:RadLexConcept {uri: rel.parent})
    MERGE (child)-[:SUBCLASS_OF]->(parent)
    """
    tx.run(query, rels=rels_batch)

# Extract concepts - ONLY named URIs (not blank nodes)
print("Extracting RadLex concepts...")
concepts = []
relationships = []

# Get all subjects with RID URIs
rid_subjects = set()
for s in g.subjects():
    s_str = str(s)
    if s_str.startswith("http://www.radlex.org/RID/RID"):
        rid_subjects.add(s)

print(f"Found {len(rid_subjects)} RID subjects")

# Extract properties for each RID subject
for subject in rid_subjects:
    subject_str = str(subject)
    rid = subject_str.split("/")[-1]

    concept_data = {
        "uri": subject_str,
        "rid": rid,
        "label": None,
        "preferredName": None,
        "definition": None,
        "synonyms": [],
        "fmaid": None
    }

    # Get Preferred_name (primary label in RadLex)
    pref_name_uri = URIRef("http://radlex.org/RID/Preferred_name")
    for pref_name in g.objects(subject, pref_name_uri):
        concept_data["preferredName"] = str(pref_name)
        concept_data["label"] = str(pref_name)  # Use as main label
        break

    # Fallback to rdfs:label if no Preferred_name
    if not concept_data["label"]:
        for label in g.objects(subject, RDFS.label):
            concept_data["label"] = str(label)
            break

    # Get definition
    definition_uri = URIRef("http://radlex.org/RID/Definition")
    for definition in g.objects(subject, definition_uri):
        concept_data["definition"] = str(definition)
        break

    # Get synonyms
    synonym_uri = URIRef("http://radlex.org/RID/Synonym")
    for synonym in g.objects(subject, synonym_uri):
        concept_data["synonyms"].append(str(synonym))

    # Get FMAID (FMA cross-reference)
    fmaid_uri = URIRef("http://radlex.org/RID/FMAID")
    for fmaid in g.objects(subject, fmaid_uri):
        concept_data["fmaid"] = str(fmaid)
        break

    # Only add if we have at least a label
    if concept_data["label"]:
        concepts.append(concept_data)

    # Extract subclass relationships (only between named RID classes)
    for parent in g.objects(subject, RDFS.subClassOf):
        if isinstance(parent, URIRef):
            parent_str = str(parent)
            if parent_str.startswith("http://www.radlex.org/RID/RID"):
                relationships.append({
                    "child": subject_str,
                    "parent": parent_str
                })

print(f"Found {len(concepts)} concepts with labels")
print(f"Found {len(relationships)} subclass relationships")

# Show samples
print("\nSample concepts:")
for concept in concepts[:10]:
    print(f"  {concept['rid']}: {concept['label']}")
    if concept['synonyms']:
        print(f"    Synonyms: {', '.join(concept['synonyms'][:3])}")

# Import to Neo4j
print("\nImporting to Neo4j...")
with driver.session() as session:
    print("Clearing database...")
    session.execute_write(clear_database)

    print("Creating constraints...")
    session.execute_write(create_constraints)

    print("Importing concepts...")
    batch_size = 1000
    for i in range(0, len(concepts), batch_size):
        batch = concepts[i:i+batch_size]
        session.execute_write(import_concepts, batch)
        print(f"  Imported {min(i+batch_size, len(concepts))}/{len(concepts)} concepts")

    print("Importing relationships...")
    for i in range(0, len(relationships), batch_size):
        batch = relationships[i:i+batch_size]
        session.execute_write(import_relationships, batch)
        print(f"  Imported {min(i+batch_size, len(relationships))}/{len(relationships)} relationships")

print("\n Import complete!")

# Verify
with driver.session() as session:
    result = session.run("""
        MATCH (c:RadLexConcept)
        WHERE c.label IS NOT NULL
        RETURN count(c) as count
    """)
    count = result.single()["count"]
    print(f"\nTotal concepts with labels: {count}")

    result = session.run("""
        MATCH (c:RadLexConcept)
        WHERE c.rid STARTS WITH 'RID'
        RETURN c.rid as rid, c.label as label
        ORDER BY c.rid
        LIMIT 20
    """)
    print("\nSample concepts:")
    for record in result:
        print(f"  {record['rid']}: {record['label']}")

    # Check for specific medical concepts
    result = session.run("""
        MATCH (c:RadLexConcept)
        WHERE toLower(c.label) CONTAINS 'mri' 
           OR toLower(c.label) CONTAINS 'magnetic resonance'
        RETURN c.rid as rid, c.label as label
        LIMIT 5
    """)
    print("\nMRI-related concepts:")
    for record in result:
        print(f"  {record['rid']}: {record['label']}")

driver.close()