import asyncio
import os
from dotenv import load_dotenv
import ssl
from typing import Any, Dict, List

import certifi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_info, log_success, log_error, log_warning, log_header)

load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                              show_progress_bar=True, chunk_size=50, retry_min_seconds=10)

vectorstore = PineconeVectorStore(index_name="langchain-docs-2026", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_pages=100, max_bread_time=20)
tavily_crawl = TavilyCrawl()


async def main():
    """Main function to orchestrate the crawling, extraction, and ingestion process."""
    log_header("DOCUMENTATION: Crawling and Ingesting Documentation")
    log_info(" TavilyCrwal - Crawling the web documentation from https://python.langchain.com/", Colors.PURPLE)

    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 5,
        "extract_depth": "advanced"
    })


    all_docs=[Document(page_content=result['raw_content'], metadata={"source": result['url']}) for result in res["results"]]


    log_success(f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documentation site")

if __name__ ==  "__main__":
    asyncio.run(main())
