import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = TextLoader('/Users/s0v00ar/agentic/langchain-course/rag/mediumblog1.txt')
    documents = loader.load()

    print(f"splitting into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    print(f"Creating vector store...")
    vectorstore = PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv('INDEX_NAME'))
    print("Ingestion complete!")

