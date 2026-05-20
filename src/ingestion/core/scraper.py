import logging
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


def get_docs_folder_path() -> str:
    current_dir = Path(__file__).parent.parent.parent
    docs_path = str(current_dir / "data" / "docs")
    return docs_path


def get_document_filename(url: str) -> str:
    parsed = urlparse(url)
    filename = parsed.path.strip("/")

    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)

    if not filename:
        filename = "index"

    return filename


async def store_docs_in_files():
    docs_folder = get_docs_folder_path()
    if not Path(docs_folder).exists():
        logger.info("Creating docs folder at %s", docs_folder)
        Path(docs_folder).mkdir(parents=True)

    if len(list(Path(docs_folder).glob("*"))) > 0:
        logger.info("Skipping scraping, docs folder already contains files")
        return

    logger.info("Scraping docs from langchain.com...")

    index_url = "https://docs.langchain.com/llms.txt"

    async with httpx.AsyncClient() as client:
        response = await client.get(index_url, timeout=None)
        response.raise_for_status()
        text = response.text

    urls = []

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- [") and "https://docs.langchain.com/" in line:
            url = line.split("(")[1].split(")")[0]
            urls.append(url)

    for url in urls:
        if "langsmith" in url or "javascript" in url:
            continue
        logger.info("Scraping %s", url)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30)
                response.raise_for_status()
                markdown = response.text

            filename = get_document_filename(url)
            filepath = Path(docs_folder) / f"{filename}"
            filepath.write_text(markdown, encoding="utf-8")

        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.warning("Skipping %s — %s", url, e)
            continue

    logger.info("Scraping completed: %d docs downloaded", len(urls))
