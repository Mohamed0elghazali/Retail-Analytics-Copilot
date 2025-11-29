import os 
from dotenv import dotenv_values
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

from agent.rag.retrieval import MarkdownLoaderAndSplitter, TfidfRetriever
from agent.tools.sqlite_tool import SQLiteClient

app_setting = dotenv_values()

docs = MarkdownLoaderAndSplitter(app_setting.get("DOCS_PATH"))
retriever = TfidfRetriever(docs.chunks, k=app_setting.get("RETRIEVAL_RESULTS"))

db = SQLiteClient(app_setting.get("DATABASE_PATH"))

ollama_llm = ChatOllama(
    model=app_setting.get("OLLAMA_LLM_MODEL_ID"),
    temperature=0,
    num_ctx=1024,
    seed=111
)

groq_llm = ChatGroq(
    model=app_setting.get("GROQ_LLM_MODEL_ID"),
    temperature=0,
    max_tokens=1024,
    api_key=app_setting.get("GROQ_API_KEY")
)
