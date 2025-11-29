### Prompts.py

from langchain_core.prompts import PromptTemplate

ROUTER_PROMPT = PromptTemplate.from_template("""
You are a routing classifier. Your job is to choose EXACTLY one of these options:

- rag
- sql
- hybrid

You classify based on the *type of reasoning required*, not the query surface form.

----------------------------------------------------------------------
### ROUTING PRINCIPLES

#### Choose **rag** when the query requires ONLY unstructured knowledge:
This includes any information that:
- Exists in documents, manuals, policy text, KPI definitions, marketing calendars, catalogs, or narrative descriptions.
- Requires interpreting rules, definitions, formulas, conditions, time periods, campaign names, fiscal calendars, or business concepts.
- Cannot be directly computed from structured tables.

Common signals:
- “According to the policy…”
- “Per the KPI definition…”
- “As defined in the calendar…”
- “What is the return window / rule / policy / description?”
- The task ends after retrieving/understanding text, with NO numeric computation.

----------------------------------------------------------------------
#### Choose **sql** when the query is answered *entirely* by structured data:
This includes:
- Pure numeric retrieval or aggregations
- SUM, COUNT, revenue, totals, top-N rankings, filtering by simple fields
- Queries that refer to date ranges explicitly provided by the user
- No document-defined rules, formulas, or calendar logic are required.

Common signals:
- “Top N products…”
- “Total revenue…”
- “Quantity sold…”
- Dates are directly provided (not named periods like ‘Summer Promo 1997’)
- No need to understand definitions, policies, formulas, or campaigns.

----------------------------------------------------------------------
#### Choose **hybrid** when the query requires BOTH:
1. **Unstructured document lookup**, AND  
2. **Structured SQL computation**

This occurs when:
- A KPI, formula, business rule, derived metric, or interpretation MUST be obtained from documents before SQL can execute.
- A named period (e.g., “Winter Classics 1997”, “Summer Beverages 1997”) requires calendar lookup to translate into date ranges.
- Policy rules, cost assumptions, multipliers, or product metadata must be extracted from text and then applied to SQL tables.
- The query requires mixing knowledge + computation.

Common signals:
- “Using the definition from…”
- “Based on the KPI formula…”
- “During [named campaign]”
- “According to the policy, compute…”
- “Per the calendar… then calculate…”

----------------------------------------------------------------------

### OUTPUT RULES
- Output EXACTLY one word: rag, sql, or hybrid.
- No explanation. No punctuation. No extra text.

User Query:
{query}

Your output:
""")

PLANNER_PROMPT = PromptTemplate.from_template("""
You are a constraint extractor.

Given the retrieved context chunks, extract any constraints that could be used for planning a query.

Extract:
- date ranges (e.g., "last 7 days", "2023-01 to 2023-03")
- KPIs or metrics formulas (e.g., "revenue", "conversion rate", "sum(sales)")
- categories or labels (e.g., "product A", "region = Europe")
- entities (company names, product names, user IDs, etc.)

Return JSON following the exact schema.

Retrieved Chunks:
{chunks}
""")

SQL_PROMPT = PromptTemplate.from_template("""
You are an expert SQL generator for **SQLite**. 
Your job is to produce a **single valid SELECT query** that answers the user question.

You will receive:
- The **database schema**
- Extracted **constraints**
- The **user question**
- (Optional) An **error message** from the previous failed query
- (Optional) The **previous SQL query** that caused the error

Follow the rules carefully.

========================================
### SCHEMA
{schema}

### CONSTRAINTS (date ranges, categories, KPIs, entities)
{constraints}

### USER QUESTION
{question}

### PREVIOUS SQL (optional)
{previous_sql}

### ERROR MESSAGE (optional)
{error}
========================================

### RULES
1. **Output ONLY valid SQLite SQL** — no text, no explanations.
2. Always return **one SELECT statement**, never multiple.
3. Use only tables/columns that exist in the schema.
4. Apply all constraints (filters, date ranges, categories, entities).
5. If KPI metrics exist, compute them inside the SQL using the formula.
6. If this is a retry:
   - **Fix the previous SQL instead of starting from scratch.**
   - The error message describes exactly what to correct.
   - Maintain user intent.
7. Never hallucinate columns or tables.

### YOUR TASK
Generate the corrected SQL query (or the initial SQL if no error exists).
""")

SYNTH_PROMPT = PromptTemplate.from_template("""
You are a synthesizer that must combine RAG results, SQL results, and user instructions.

### Requirements
- Produce a FINAL ANSWER in the requested format: "{format_hint}"
- Use both RAG and SQL results when relevant
- EXPLANATION: max 2 sentences ONLY

### Inputs
Question:
{question}

RAG Output:
{rag_output}

SQL Output:
{sql_output}

### Output JSON schema:
- final_answer: string
- explanation: <= 2 sentences
""")
