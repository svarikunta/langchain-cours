import asyncio
import os
import ssl
from typing import Any, Dict, List
import json
from datetime import datetime
from dotenv import load_dotenv
import certifi
from langchain_tavily import TavilyMap, TavilyExtract
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress


from logger import log_header, log_info, log_success, log_error, log_warning, Colors

console = Console()

load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=500,
                       tavily_api_key='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')

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10
)

vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
# vectorstore = PineconeVectorStore(
#     index_name="langchain-docs-2026", embedding=embeddings
# )


def chunk_url(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i: i + chunk_size]
        chunks.append(chunk)
    return chunks


async def extract_batch(batch_urls: List[str], batch_number: int) -> List[Dict[str, Any]]:
    """Process a batch of URLs using TavilyExtract."""
    try:
        log_info(f"Processing batch {batch_number} with {len(batch_urls)} URLs", Colors.BLUE)
        tavily_extract = TavilyExtract(tavily_api_key='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')
        extraction_result = await tavily_extract.ainvoke({"urls": batch_urls})
        extracted_docs = extraction_result.get("results", [])
        log_success(f"Batch {batch_number}: Extracted content from {len(extracted_docs)} URLs")
        return extraction_result
    except Exception as e:
        log_error(f"Batch {batch_number}: Error processing batch - {str(e)}")
        return []


async def process_all_batches(url_batches: List[List[str]]):
    """Process all batches of URLs sequentially."""
    log_info(f"Starting sequential processing of all URL batches :{len(url_batches)} with TavilyExtract", Colors.BLUE)
    tasks = [extract_batch(batch, i + 1) for i, batch in enumerate(url_batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            log_error(f"Batch {failed_batches}")
            failed_batches += 1
        else:
            for extracted_page in result["results"]:
                document = Document(
                    page_content=extracted_page["raw_content"],
                    metadata={"source": extracted_page["url"]}
                )
                all_pages.append(document)
    log_success(f"Tavily extraction is completed total pages extracts : {len(all_pages)}")

    if failed_batches > 0:
        log_warning(f"{failed_batches} batches failed during processing")
    return all_pages


async def index_documentts_async(documents: List[Document], batch_size: int = 50):
    """Index documents in batches."""
    log_info(f"Starting indexing of {len(documents)} documents in batches of {batch_size}", Colors.BLUE)
    batches = [
        documents[i: i+batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(f"Total batches to index: {len(batches)} of {batch_size}", Colors.BLUE)

    async def add_batch(batch: List[Document], batch_number: int) -> bool:
        """Add a batch of documents to the vector store."""
        try:
            log_info(f"Indexing batch {batch_number} with {len(batch)} documents", Colors.BLUE)
            # Here you would add the batch to your vector store, e.g.:
            await vectorstore.aadd_documents(batch)
            log_success(f"Batch {batch_number}: Successfully indexed {len(batch)} documents")
        except Exception as e:
            log_error(f"Batch {batch_number}: Error indexing batch - {str(e)}")
            return False
        return True

    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    successful_batches = sum(1 for result in results if result is True)
    failed_batches = len(results) - successful_batches
    log_success(f"Indexing completed: {successful_batches} batches indexed successfully")
    if failed_batches > 0:
        log_warning(f"{failed_batches} batches failed during indexing")





async def main():
    """Main function to orchestrate the mapping and extraction process."""
    log_header("DOCUMENTATION Ingestion pipeline")
    log_info("TavilyMap - Mapping the website structure for https://python.langchain.com/",
             Colors.PURPLE)
    site_map = tavily_map.invoke({"url": "https://python.langchain.com/"})
    urls = site_map.get("results", [])
    log_success(f"TavilyMap: Found {len(urls)} URLs in the site map")

    url_batches = chunk_url(urls, 20)
    log_info(
        f"URL processing - split {len(urls)} URLs in batches of 20 with TavilyExtract into : {len(url_batches)}",
        Colors.PURPLE)

    all_docs = await process_all_batches(url_batches)

    log_info(
        f" Extract completed {len(urls)} URLs in batches of 20 with TavilyExtract into : {len(all_docs)}")

    log_info("Splitting documents into chunks of 4000 characters with RecursiveCharacterTextSplitter", Colors.PURPLE)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splited_docs = text_splitter.split_documents(all_docs)
    log_success(f"Document splitting completed: {len(all_docs)} documents split into {len(splited_docs)} chunks")

    await index_documentts_async(splited_docs, batch_size=500)



    # write main method to invoke main


if __name__ == "__main__":
    asyncio.run(main())
