import asyncio
import os
import ssl
from typing import Any, Dict, List
import json
from datetime import datetime

import certifi
from langchain_tavily import TavilyMap, TavilyExtract
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress


console = Console()


ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

tavily_map = TavilyMap(max_depth=5, max_breadth=15, limit=500,tavily_api_key='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')

demo_url = "https://python.langchain.com/docs/introduction/"

console.print(f"Mapping website structure for {demo_url}", style="bold blue")
site_map = tavily_map.invoke({"url": demo_url})
urls =site_map.get("results", [])

console.print(f"Found {len(urls)} URLs in the site map", style="bold green")

# show first 10 urls in a table
table = Table(title="Mapped URLs")
table.add_column("URL", style="cyan")



for url in urls[:10]:
    console.print(url)

tavily_extract = TavilyExtract(tavily_api_key='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')

async def main():
    sample_urls = [urls[20]]
    console.print(f"Extracting content from {len(sample_urls)} sample URLs", style="bold blue")

    extraction_result = await tavily_extract.ainvoke({"urls": sample_urls})

    extracted_docs = extraction_result.get("results", [])
    console.print(f"Extracted content from {len(extracted_docs)} URLs", style="bold green")

    ## show summary of extracted content in a table
    extraction_table = Table(title="Extracted Content Summary")
    extraction_table.add_column("URL", style="cyan")
    extraction_table.add_column("Content Length", style="magenta")

    for doc in extracted_docs:
        url = doc.get("url", "N/A")
        content_length = str(len(doc.get("raw_content", "")))
        extraction_table.add_row(url, content_length)

    console.print(extraction_table)

    url_batches = chunk_list(urls[:9], chunk_size=3)
    tasks = [extract_batch(batch, i) for i, batch in enumerate(url_batches)]
    batch_results = await asyncio.gather(*tasks)

    all_extracted = []
    for batch in batch_results:
        all_extracted.extend(batch)

    console.print(f"Total extracted docs: {len(all_extracted)}", style="bold green")

def chunk_list(urls, chunk_size) -> List[List[str]]:
    """Utility function to chunk a list into smaller sublists."""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk= urls[i:i + chunk_size]
        chunks.append(chunk)
    return chunks
async def extract_batch(urls, batch_num=5) -> List[Dict[str, Any]]:
    """Extract content from URLs in chunks to manage rate limits and resources."""
    try:
        console.print(f"processing batch :{batch_num}", style="bold blue")
        extraction_result = await tavily_extract.ainvoke({"urls": urls})
        extracted_docs = extraction_result.get("results", [])
        console.print(f"completed batch :{batch_num}", style="bold blue")

        return extracted_docs
    except Exception as e:
        console.print(f"Error processing batch {batch_num}: {str(e)}", style="bold red")
        return []

asyncio.run(main())


