from rdflib import Graph, URIRef
from neo4j import GraphDatabase
from collections import defaultdict, Counter
import os

def get_neo4j_credentials():
    neo4j_uri = os.getenv('NEO4J_URI')
    neo4j_user = os.getenv('NEO4J_USER')
    neo4j_password = os.getenv('NEO4J_PASSWORD')

    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        exit(1)

    return neo4j_uri, neo4j_user, neo4j_password

neo4j_uri, neo4j_user, neo4j_password = get_neo4j_credentials()
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

confirm = input("WARNING: This will DELETE all data. Type 'YES' to continue: ")
if confirm != 'YES':
    exit(0)

with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")

g = Graph()
g.parse("RadLex.owl", format="xml")

rel_types_found = Counter()

for s, p, o in g:
    if isinstance(o, URIRef):
        s_str = str(s)
        o_str = str(o)

        if s_str.startswith("http://www.radlex.org/RID/RID") and \
                o_str.startswith("http://www.radlex.org/RID/RID"):
            p_str = str(p)

            if 'radlex.org/RID/' in p_str or p_str == "http://www.w3.org/2000/01/rdf-schema#subClassOf":
                rel_types_found[p_str] += 1

RELATIONSHIP_PROPERTIES = {}
for rel_uri in rel_types_found.keys():
    rel_name = rel_uri.split('/')[-1]
    neo4j_name = rel_name.upper().replace(' ', '_').replace('-', '_').replace('#', '_')
    RELATIONSHIP_PROPERTIES[rel_uri] = neo4j_name

rid_subjects = set()
for s in g.subjects():
    s_str = str(s)
    if s_str.startswith("http://www.radlex.org/RID/RID"):
        rid_subjects.add(s)

concepts = []
relationships_by_type = defaultdict(list)

for idx, subject in enumerate(rid_subjects):
    subject_str = str(subject)
    rid = subject_str.split("/")[-1]

    concept = {
        "uri": subject_str,
        "rid": rid,
        "label": None,
        "preferredName": None,
        "definition": None,
        "synonyms": [],
        "fmaid": None,
        "umlsId": None,
        "umlsTerm": None,
    }

    for pred, obj in g.predicate_objects(subject):
        pred_str = str(pred)

        if pred_str == "http://www.radlex.org/RID/Preferred_name":
            concept["preferredName"] = str(obj)
            concept["label"] = str(obj)
        elif pred_str == "http://www.w3.org/2000/01/rdf-schema#label" and not concept["label"]:
            concept["label"] = str(obj)
        elif pred_str == "http://www.radlex.org/RID/Definition":
            concept["definition"] = str(obj)
        elif pred_str == "http://www.radlex.org/RID/Synonym":
            concept["synonyms"].append(str(obj))
        elif pred_str == "http://www.radlex.org/RID/FMAID":
            concept["fmaid"] = str(obj)
        elif pred_str == "http://www.radlex.org/RID/UMLS_ID":
            concept["umlsId"] = str(obj)
        elif pred_str == "http://www.radlex.org/RID/UMLS_Term":
            concept["umlsTerm"] = str(obj)

        elif pred_str in RELATIONSHIP_PROPERTIES:
            if isinstance(obj, URIRef):
                obj_str = str(obj)
                if obj_str.startswith("http://www.radlex.org/RID/RID"):
                    rel_type = RELATIONSHIP_PROPERTIES[pred_str]
                    relationships_by_type[rel_type].append({
                        "source": subject_str,
                        "target": obj_str
                    })

    if concept["label"]:
        concepts.append(concept)

def create_constraints(tx):
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:RadLexConcept) REQUIRE c.uri IS UNIQUE")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (c:RadLexConcept) ON (c.rid)")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (c:RadLexConcept) ON (c.label)")

def import_concepts_batch(tx, batch):
    tx.run("""
        UNWIND $batch AS c
        CREATE (n:RadLexConcept {
            uri: c.uri,
            rid: c.rid,
            label: c.label,
            preferredName: c.preferredName,
            definition: c.definition,
            synonyms: c.synonyms,
            fmaid: c.fmaid,
            umlsId: c.umlsId,
            umlsTerm: c.umlsTerm
        })
    """, batch=batch)

def make_rel_importer(rel_type):
    def import_rels(tx, batch):
        query = f"""
            UNWIND $batch AS r
            MATCH (a:RadLexConcept {{uri: r.source}})
            MATCH (b:RadLexConcept {{uri: r.target}})
            CREATE (a)-[:{rel_type}]->(b)
        """
        tx.run(query, batch=batch)
    return import_rels

with driver.session() as session:
    session.execute_write(create_constraints)

    batch_size = 1000
    for i in range(0, len(concepts), batch_size):
        batch = concepts[i:i+batch_size]
        session.execute_write(import_concepts_batch, batch)

    for rel_type, rels in sorted(relationships_by_type.items(), key=lambda x: -len(x[1])):
        importer = make_rel_importer(rel_type)
        for i in range(0, len(rels), batch_size):
            batch = rels[i:i+batch_size]
            session.execute_write(importer, batch)

driver.close()