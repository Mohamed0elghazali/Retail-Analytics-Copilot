from pydantic import BaseModel
from typing import Literal, TypedDict, Optional, List, Dict, Any

class AgentState(TypedDict):
    id: str
    question: str
    table_names: List[str]
    route: Optional[Literal["rag", "sql", "hybrid"]]
    retrieved_docs: List[Dict[str, Any]]
    constraints: Dict[str, Any]
    sql_query: Optional[str]
    sql_result: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    error: Optional[str]
    attempt_count: int
    format_hint: Optional[str]
    citations: List[str]
    confidence: float
    explanation: str

class RouterState(BaseModel):
    route: Literal["rag", "sql", "hybrid"]

class ConstraintPlan(BaseModel):
    date_ranges: Optional[List[str]] = None
    kpis: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    
class SQLGeneration(BaseModel):
    sql: str

class SQLExecutionResult(BaseModel):
    columns: Optional[List[str]] = None
    rows: Optional[List[List]] = None
    error: Optional[str] = None

class SynthesizerOutput(BaseModel):
    final_answer: str
    explanation: str