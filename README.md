# SmartClaims: A GraphRAG solution for radiology billing validation
AI-powered SMART on FHIR app using GraphRAG to detect billing discrepancies in radiology reports

## Project Structure

```
SMARTClaims-LLM/
├── app/                          # SMART on FHIR web app
│   ├── index.html                # Patient viewer
│   ├── standalone.html           # EHR selector
│   ├── launch.html               # EHR launch handler
│   ├── css/styles.css
│   └── js/
│       ├── app-config.js         # Your config (gitignored)
│       ├── app-config.example.js # Config template
│       ├── patient-viewer.js
│       └── standalone.js
│
├── graphrag/                     # Knowledge graph
│   └── vector_embeddings.py      # RadLex embeddings in Neo4j
│
├── llm/                          # LLM training
│   ├── data_preparation.py       # Prepare training data with GraphRAG
│   └── unsloth_medgemma27b_1k.py # Fine-tune MedGemma-27B
│
└── ablation_study/               # Prove RadLex helps
    ├── no_radlex_baseline.py     # Train without RadLex
    └── compare_models.py         # Compare models
```

## Preliminary Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  EHR System │────▶│ SMART on    │────▶│  GraphRAG   │────▶│  MedGemma   │
│ Epic/Cerner │     │ FHIR App    │     │  (RadLex)   │     │    LLM      │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           │                   │                   │
                    Get radiology       Add RadLex          Extract billable
                       reports           context              procedures
```
## Preliminary Test Results

| Model | Training Loss |
|-------|---------------|
| WITH RadLex | 1.70 |
| WITHOUT RadLex | 2.27 |

RadLex context improves model performance.

## Tech Stack

- **Frontend**: HTML/CSS/JS, SMART on FHIR
- **EHRs**: Epic, Cerner (FHIR R4), OpenEMR
- **Knowledge Graph**: Neo4j, RadLex ontology, vector embeddings
- **LLM**: MedGemma-27B, Unsloth, LoRA fine-tuning
