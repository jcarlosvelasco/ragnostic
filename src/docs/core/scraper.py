import re
from pathlib import Path
from urllib.parse import urlparse

import requests


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
        print(f"Creating docs folder at {docs_folder}")
        Path(docs_folder).mkdir(parents=True)

    if len(list(Path(docs_folder).glob("*"))) > 0:
        print("Skipping scraping, docs folder already contains files")
        return

    index_url = "https://docs.langchain.com/llms.txt"
    text = requests.get(index_url, timeout=30).text

    urls = []

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- [") and "https://docs.langchain.com/" in line:
            url = line.split("(")[1].split(")")[0]
            urls.append(url)

    for url in urls[:20]:
        if "langsmith" in url or "javascript" in url:
            continue

        markdown = requests.get(url, timeout=30).text

        filename = get_document_filename(url)

        filepath = Path(docs_folder) / f"{filename}"

        filepath.write_text(markdown, encoding="utf-8")
