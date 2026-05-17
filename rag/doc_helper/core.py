import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10
)


vectorstore = PineconeVectorStore(index_name="langchain-docs-2026", embedding=embeddings)

model = init_chat_model("gpt-5.2", openai_api_key=os.getenv('OPENAI_API_KEY'))


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """retrieve relevant documentation to help answer user queries about langchain"""
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)
    serialized_docs = [f"Source: {doc.metadata['source']}\nContent: {doc.page_content}" for doc in retrieved_docs]
    return "\n\n".join(serialized_docs), retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """Run the RAG pipeline to answer a query using retrieved documentation.

     args:
         query: The user question
     Returns:
         Dictionary : containing the answer
         Context : List of Retrieved documents used to generate the answer.
    """

    system_prompt = """You are a helpful assistant for answering questions about the langchain documentation.
     you have access to a tool that can retrieve relevant documentation.
     Ues the tool to find relevant information before answering questions. 
     Always cite the source you use for your answers.
     if you cannot find the answer in the retrieved documentation, say so."""

    agent = create_agent(model=model, tools=[retrieve_context], system_prompt=system_prompt)
    messages = [{"role": "user", "content": query}]
    response = agent.invoke({"messages": messages})
    answer = response["messages"][-1].content

    context_docs = []
    for msg in response["messages"]:
        if isinstance(msg, ToolMessage) and hasattr(msg, "artifact"):
            if isinstance(msg.artifact, list):
                context_docs.extend(msg.artifact)

    return {
        "answer": answer,
        "context": context_docs
    }

if __name__ == "__main__":
    query = "What are deep agents?"
    result = run_llm(query)
    print("Answer:", result["answer"])
    print("\nContext:")
    for doc in result["context"]:
        print(f"- {doc.metadata['source']}")
