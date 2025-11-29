import logging
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from .models import AgentState, RouterState, ConstraintPlan, SQLGeneration, SQLExecutionResult, SynthesizerOutput
from .prompts import ROUTER_PROMPT, PLANNER_PROMPT, SQL_PROMPT, SYNTH_PROMPT
from helper.clients import groq_llm, ollama_llm, retriever, db

logging.basicConfig(
    filename="logs/agentlog.log",
    format='%(asctime)s %(levelname)s %(message)s',
    filemode='a',
    level=logging.INFO)

logger = logging.getLogger()

def router_node(state: AgentState, config: RunnableConfig) -> AgentState:
    question = state["question"]
    llm = config["configurable"].get("llm")

    router_chain = ROUTER_PROMPT | llm.with_structured_output(RouterState)
    out = router_chain.invoke(question)

    state["route"] = out.route
    logger.info(f"router_node: {out=}")
    return state

def retriever_node(state: AgentState, config: RunnableConfig) -> AgentState:
    question = state["question"]
    retriever = config["configurable"].get("retriever") 

    state["retrieved_docs"] = retriever.query(question, 4)
    logger.info(f"retriever_node: {state["retrieved_docs"]=}")
    return state

def planner_node(state: AgentState, config: RunnableConfig) -> AgentState:
    chunks_text = "\n\n".join([doc.page_content for doc in state["retrieved_docs"]])
    llm = config["configurable"].get("llm")

    planner_chain = PLANNER_PROMPT | llm.with_structured_output(ConstraintPlan)
    constraints = planner_chain.invoke({"chunks": chunks_text})
    
    state["constraints"] = constraints.model_dump()
    logger.info(f"retriever_node: {state["constraints"]=}")
    return state

def nl_to_sql_node(state: AgentState, config: RunnableConfig) -> AgentState:
    question = state["question"]
    constraints = state.get("constraints", {})
    sql_error = state.get("sql_result", {}).get("error")
    sql_query = state.get("sql_query", "")

    llm = config["configurable"].get("llm")
    db = config["configurable"].get("db")

    try:
        db.connect()
        schema_str = db.extract_schema(state["table_names"])
    except Exception as e:
        logger.error(f"nl_to_sql_node: {e}")
    finally:
        db.disconnect()

    nl_to_sql_chain = SQL_PROMPT | llm.with_structured_output(SQLGeneration)
    result = nl_to_sql_chain.invoke({
        "schema": schema_str,
        "constraints": constraints,
        "question": question,
        "error": sql_error,
        "previous_sql": sql_query
    })

    state["sql_query"] = result.sql
    logger.info(f"nl_to_sql_node: {state["sql_query"]=}")
    return state

def sql_executor_node(state: AgentState, config: RunnableConfig) -> AgentState:
    sql_query = state["sql_query"]
    db = config["configurable"].get("db")

    try:
        db.connect()
        rows, col_names, error = db.execute_query(sql_query)
        result = SQLExecutionResult(columns=col_names, rows=rows, error=str(error))
    except Exception as e:
        logger.error(f"sql_executor_node: {e}")
        result = SQLExecutionResult(columns=None, rows=None, error=str(e))
    finally:
        db.disconnect()
        state["sql_result"] = result.model_dump()
    logger.info(f"sql_executor_node: {state["sql_result"]=}")
    return state

def retry_counter_node(state: AgentState) -> AgentState:
    state["attempt_count"] += 1
    return state

def Synthesizer_node(state: AgentState, config: RunnableConfig) -> AgentState:
    chunks = state.get("retrieved_docs", [])
    llm = config["configurable"].get("llm")
    synth_chain = SYNTH_PROMPT | llm.with_structured_output(SynthesizerOutput)

    result = synth_chain.invoke({
        "format_hint": state["format_hint"],
        "question": state["question"],
        "rag_output": "\n\n".join(doc.page_content for doc in chunks),
        "sql_output": state.get("sql_result", {}),
    })

    state["final_answer"] = result.final_answer
    state["explanation"] = result.explanation
    logger.info(f"Synthesizer_node: {state["final_answer"]=}")
    return state

def format_output(state: AgentState, config: RunnableConfig) -> AgentState:
    state["citations"] = []
    rag_score = 1
    sql_score = 1
    rows_score = 1

    if state["route"] in ["rag", "hybrid"] and state.get("retrieved_docs", []):
        chunks = state.get("retrieved_docs", [])
        state["citations"] += [f"{chunk.metadata["source"]}:chunk_{chunk.metadata["chunk_id"]}" for chunk in chunks] 
        rag_score = sum([chunk.metadata["score"] for chunk in chunks]) / len(chunks)

    if state["route"] in ["sql", "hybrid"] and state["table_names"]:
        state["citations"] += state["table_names"]
        sql_score = 1 if not state.get("sql_result", {}).get("error") else 0
        rows_score = 1 if state["sql_result"].get("rows", {}) else 0

    state["confidence"] = round((rag_score + sql_score + rows_score) / 3, 3)
    return state

graph_agent = StateGraph(AgentState)

graph_agent.add_node("router", router_node)
graph_agent.add_node("retriever", retriever_node)
graph_agent.add_node("planner", planner_node)
graph_agent.add_node("nl_to_sql", nl_to_sql_node)
graph_agent.add_node("sql_executor", sql_executor_node)
graph_agent.add_node("retry_counter", retry_counter_node)
graph_agent.add_node("Synthesizer", Synthesizer_node)
graph_agent.add_node("format_output", format_output)

graph_agent.set_entry_point("router")

graph_agent.add_conditional_edges(
    "router",
    lambda x: "rag" if x["route"] in ["rag", "hybrid"] else "sql",
    {
        "rag": "retriever",
        "sql": "nl_to_sql"
    }
)

graph_agent.add_conditional_edges(
    "retriever",
    lambda x: "sql" if x["route"] in ["sql", "hybrid"] else "end",
    {
        "sql": "planner",
        "end": "Synthesizer"
    }
)

graph_agent.add_edge("planner", "nl_to_sql")
graph_agent.add_edge("nl_to_sql", "sql_executor")
graph_agent.add_edge("sql_executor", "retry_counter")

def sql_retry(state: AgentState) -> str:
    sql_rows = state["sql_result"].get("rows", [])
    if sql_rows:
        return "success"
    elif state["attempt_count"] <= 2:
        return "retry"
    return "fallback"

graph_agent.add_conditional_edges(
    "retry_counter",
    sql_retry,
    {
        "success": "Synthesizer",
        "retry": "nl_to_sql",
        "fallback": "Synthesizer"
    }
)

graph_agent.add_edge("Synthesizer", "format_output")
graph_agent.add_edge("format_output", END)

northwind_agent = graph_agent.compile()

def invoke_agent(id, question, format_hint):
    config = {
        "configurable": {
            "llm": ollama_llm, # ollama_llm, groq_llm
            "retriever": retriever, 
            "db": db
        },
        "recursion_limit": 15
    }

    input = {
        "id": id,
        "question": question,
        "format_hint": format_hint,
        "table_names": ["demo_orders", "demo_order_details", "demo_products"],
        "attempt_count": 0
    }

    out = northwind_agent.invoke(input, config)
    target_keys = ["id", "final_answer", "sql_query", "confidence", "explanation", "citations"]
    return {key: out.get(key) for key in target_keys}

if __name__ == "__main__":
    id = "rag_policy_beverages_return_days"
    question = "According to the product policy, what is the return window (days) for unopened Beverages? Return an integer."
    format_hint = "int"

    id = "sql_top3_products_by_revenue_alltime"
    question = "Top 3 products by total revenue all-time. Revenue uses Order Details: SUM(UnitPrice*Quantity*(1-Discount)). Return list[{product:str, revenue:float}]."
    format_hint = "list[{product:str, revenue:float}]"

    id = "hybrid_revenue_beverages_summer_1997"
    question = "Total revenue from the 'Beverages' category during 'Summer Beverages 1997' dates. Return a float rounded to 2 decimals."
    format_hint = "float"

    # id = "hybrid_best_customer_margin_1997"
    # question = "Per the KPI definition of gross margin, who was the top customer by gross margin in 1997? Assume CostOfGoods is approximated by 70% of UnitPrice if not available. Return {customer:str, margin:float}.", 
    # format_hint = "{customer:str, margin:float}"

    print(invoke_agent(id, question, format_hint))
