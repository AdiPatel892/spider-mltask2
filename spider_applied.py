import os
import warnings
import logging


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Accessing __path__.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import pandas as pd
from typing import TypedDict, List
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END

load_dotenv()

df = pd.read_csv("medquad.csv").dropna(subset=["question", "answer"])


documents = [
    Document(
        page_content=f"Q: {row['question']}\nA: {row['answer']}",
        metadata={
            "source": row["source"],
            "focus_area": row.get("focus_area", "General") 
        }
    )
    for _, row in df.iterrows()
]
print("Building FAISS Vector Store...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
print("Retriever ready!")


llm_model = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Flash-DSpark",
    task="text-generation"
)


llm = ChatHuggingFace(llm=llm_model)

# llm = ChatGoogleGenerativeAI(model= "gemini-2.5-flash")

web_search_tool = TavilySearchResults(max_results=3,
                                      search_depth="advanced"
                    
                                      )

class GraphState(TypedDict):
    prompt_relevancy: bool
    question: str
    context: str
    sources: List[str]
    is_good_match: str
    answer: str



def query_node(state: GraphState):
    prompt = (
        f"Does the following  question relates to any medical related query?\n"
        f"Reply ONLY with the word YES or NO.\n\n"
        
        f"Question: {state['question']}"
    )
    result = llm.invoke(prompt).content.strip().upper()
    if "YES" in result:
        return {"prompt_relevancy": True}
    else:
        return {"prompt_relevancy": False, "answer":"Ask something related to medical field"}
    
    
    
    
    

def retrieve_node(state: GraphState):
    """Node 1: Search medquad.csv via FAISS retriever."""
    retrieved_docs = retriever.invoke(state["question"])
    
    
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    source_list = [f"{doc.metadata['source']} ({doc.metadata['focus_area']})" for doc in retrieved_docs]
    
    return {"context": context_text, "sources": source_list}

def grade_node(state: GraphState):
    prompt = (
        f"Does the following medical text contain enough facts to answer the question?\n"
        f"Reply ONLY with the word YES or NO.\n\n"
        f"Text: {state['context']}\n"
        f"Question: {state['question']}"
    )
    
    response = llm.invoke(prompt).content.strip().upper()

    if "YES" in response:
        return {"is_good_match": "yes"}
    else:
        return {"is_good_match": "no"}

def web_search_node(state: GraphState):
   
    
   
    trusted_query = f"{state['question']} site:who.int OR site:cdc.gov OR site:nice.org.uk OR site:nih.gov"
    search_results = web_search_tool.invoke({"query": trusted_query})
    
    web_context = "\n\n".join([f"Snippet: {res['content']}" for res in search_results])
    web_sources = [f"Online Trusted Source: {res['url']}" for res in search_results]
    
    return {"context": web_context, "sources": web_sources}

def generate_node(state: GraphState):
    """Node 4: Generate grounded response with citations."""
    prompt = ChatPromptTemplate.from_template(
        "You are a trustworthy Healthcare Information Assistant.\n"
        "Answer the user question strictly using ONLY the provided context below. "
        "Do not assume or extrapolate. If the context does not fully answer the question, just reply 'I am unable to answer the query'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Provide a clear medical answer and list the supporting evidence."
    )
    generator = prompt | llm
    response = generator.invoke({"context": state["context"], "question": state["question"]})
    
   
    formatted_sources = "\n".join([f"- {src}" for src in set(state["sources"])])
    final_answer = f"{response.content}\n\n**Supporting Sources:**\n{formatted_sources}"
    
    return {"answer": final_answer}

builder = StateGraph(GraphState)
builder.add_node("query", query_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("grade", grade_node)
builder.add_node("web_search", web_search_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "query")
builder.add_conditional_edges(
    "query",
    lambda state: state["prompt_relevancy"],
    {
        True: "retrieve",
        False: END
    }
)
builder.add_edge("retrieve", "grade")


builder.add_conditional_edges(
    "grade",
    lambda state: state["is_good_match"],
    {
        "yes": "generate",
        "no": "web_search"
    }
)

builder.add_edge("web_search", "generate")
builder.add_edge("generate", END)

app = builder.compile()




st.title("Spider ML Task-2")
st.divider()

@st.fragment
def st_queries():
    st_query = st.text_input("Ask your Question here")
    st_go=st.button("Go")

    if st_go or st_query:
        result = app.invoke({"question": st_query})


        st.write(result["answer"])
st_queries()
