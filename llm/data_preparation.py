import pandas as pd
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

# Config - UPDATE THESE
DATA_PATH = "YOUR_DATA_PATH.csv"
OUTPUT_DIR = Path("YOUR_OUTPUT_DIR")
NEO4J_URI = "YOUR_NEO4J_URI"
NEO4J_USER = "YOUR_NEO4J_USER"
NEO4J_PASSWORD = "YOUR_NEO4J_PASSWORD"
TRAIN_RATIO = 0.8
RADLEX_TOP_K = 10
GRAPH_DEPTH = 2

OUTPUT_DIR.mkdir(exist_ok=True)
model = SentenceTransformer('all-MiniLM-L6-v2')
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def get_all_relationship_types():
    with driver.session() as session:
        result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
        return [r['relationshipType'] for r in result]

ALL_RELATIONSHIPS = get_all_relationship_types()

def semantic_search_radlex(query_text, limit=10):
    embedding = model.encode(query_text).tolist()
    with driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes('concept_embeddings', $limit, $embedding)
            YIELD node, score
            RETURN node.rid as rid, coalesce(node.preferredName, node.label) as name,
                   coalesce(node.definition, '') as definition, score
        """, embedding=embedding, limit=limit)
        return [dict(r) for r in result]

def get_concept_context(rid, depth=2):
    rel_pattern = "|".join(ALL_RELATIONSHIPS)
    with driver.session() as session:
        h = session.run(f"""
            MATCH (c:RadLexConcept {{rid: $rid}})-[:RDF_SCHEMA_SUBCLASSOF*1..{depth}]-(related:RadLexConcept)
            RETURN DISTINCT coalesce(related.preferredName, related.label) as name, 'hierarchy' as rel_type LIMIT 5
        """, rid=rid)
        hierarchy = [dict(r) for r in h]
        o = session.run(f"""
            MATCH (c:RadLexConcept {{rid: $rid}})-[r:{rel_pattern}]-(related:RadLexConcept)
            WHERE type(r) <> 'RDF_SCHEMA_SUBCLASSOF'
            RETURN DISTINCT coalesce(related.preferredName, related.label) as name, type(r) as rel_type LIMIT 10
        """, rid=rid)
        return hierarchy + [dict(r) for r in o]

def build_radlex_context(report_text, top_k=10, depth=2):
    concepts = semantic_search_radlex(report_text, limit=top_k)
    if not concepts:
        return "RadLex Context: None"
    lines = ["RadLex Knowledge Graph Context:"]
    for i, c in enumerate(concepts, 1):
        lines.append(f"\n{i}. {c['name']} (RID: {c['rid']})")
        if c['definition']:
            lines.append(f"   Definition: {c['definition'][:200]}...")
        related = get_concept_context(c['rid'], depth)
        rel_groups = {}
        for item in related:
            rel_groups.setdefault(item['rel_type'].replace('_', ' ').lower(), []).append(item['name'])
        for rel_type, names in rel_groups.items():
            lines.append(f"   {rel_type.title()}: {', '.join(names[:5])}")
    return "\n".join(lines)

def create_training_text(report_text, note_id):
    context = build_radlex_context(report_text, RADLEX_TOP_K, GRAPH_DEPTH)
    return {"note_id": note_id, "text": f"{context}\n\nRadiology Report:\n{report_text}\n"}

def process_dataframe(df, desc):
    examples = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        try:
            examples.append(create_training_text(row['text'], row['note_id']))
        except Exception as e:
            print(f"Error {row['note_id']}: {e}")
    return examples

def main():
    df = pd.read_csv(DATA_PATH)
    df = df[df['text'].notna()]
    train_df, val_df = train_test_split(df, test_size=(1 - TRAIN_RATIO), random_state=42)

    train_examples = process_dataframe(train_df, "Training")
    val_examples = process_dataframe(val_df, "Validation")

    pd.DataFrame(train_examples).to_json(OUTPUT_DIR / "train_dataset.jsonl", orient='records', lines=True)
    pd.DataFrame(val_examples).to_json(OUTPUT_DIR / "val_dataset.jsonl", orient='records', lines=True)

    print(f"Saved: train ({len(train_examples)}), val ({len(val_examples)})")
    driver.close()

if __name__ == "__main__":
    main()