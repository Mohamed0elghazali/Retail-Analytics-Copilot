# Retail-Analytics-Copilot
## Overview
Retail Analytics Copilot is a hybrid retrieval-augmented generation (RAG) system designed for retail business intelligence and analytics. The system combines advanced language models with structured business data to provide accurate, actionable insights for retail operations, financial analysis, and customer behavior understanding.

## Summary

* **Retrieval (**`TF-IDF Search`**)**:
  Uses `sklearn.feature_extraction.text.TfidfVectorizer` to vectorize text chunks from docs folder.

* **LLM Clients (**``**)**:
  Provides interfaces to query:

  * **Ollama Gemma 1B** – lightweight, fast testing.
  * **Groq LLaMA 3.1 8B** – larger, higher-quality reasoning.

* **Data Focus**:
  Only the three tables (`orders`, `order_details`, `products`) from the Northwind database are used to simplify testing and evaluation.

* **Evaluation Integration**:
  Can work directly with `sample_questions_hybrid_eval.jsonl` for testing retrieval and response quality.

* **Next Step**:
  Using DSPy to optimize prompts.


## Project Structure

```
RETAIL_ANALYTICS_COPILOT/
├── agent/                     # Core agent implementation
│   ├── rag/                   # Retrieval-Augmented Generation components
│   ├── tools/                 # Analytical tools and utilities
│   ├── graph_hybrid.py        # Hybrid graph implementation
│   ├── models.py              # Data models and schemas
│   └── prompts.py             # Prompt templates and management
├── data/                      # Retail datasets and business data
├── docs/                      # Documentation for RAG
├── helper/                    # Utility functions and helpers
├── logs/
├── notebooks/
├── .env.example               # Environment template
├── ALAssignment_DSPy.pdf      # Project documentation
├── outputs_hybrid.jsonl       # Generated responses and performance metrics
├── requirements.txt           # Python dependencies
├── run_agent_hybrid.py        # Main execution script
└── sample_questions_hybrid_eval.jsonl  # Evaluation dataset
```

## How to Use

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Execute the hybrid agent
python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl

# For specific retail queries, modify the input in the script
# or use the demo notebook for interactive testing
