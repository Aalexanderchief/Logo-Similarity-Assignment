import aiohttp
import asyncio
import csv
import re
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Input Parquet file with domain column
INPUT_FILE = "logos.snappy.parquet"
# Output CSV file with extracted logo URLs
OUTPUT_FILE = "outputs/logo_urls.csv"
# Concurrency level for async requests
CONCURRENCY = 50
# Request timeout in seconds
TIMEOUT = 8

# Pattern used to identify potential logo elements
LOGO_REGEX = re.compile(r"(logo|logotype)", re.IGNORECASE)

async def fetch_html(session, url):
    """
    Fetch the HTML content of the given URL asynchronously.
    """
    try:
        async with session.get(url, timeout=TIMEOUT) as resp:
            if resp.status == 200:
                return await resp.text()
    except Exception:
        return None

def normalize_url(url):
    """
    Normalize a URL to remove query parameters and trailing slashes.
    """
    if not url:
        return ""
    parsed = urlparse(url)
    return (parsed.scheme + "://" + parsed.netloc + parsed.path).strip().rstrip("/")

def extract_logo_url(html, base_url):
    """
    Extract the most likely logo URL from a webpage.
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1. Check <img> elements that look like logos
    for img in soup.find_all("img"):
        attrs = (img.get("alt", ""), img.get("class", ""), img.get("id", ""))
        if any(LOGO_REGEX.search(str(attr)) for attr in attrs):
            return urljoin(base_url, img.get("src"))

    # 2. Check <meta property="og:image">
    og_image = soup.find("meta", property="og:image")
    if og_image and og_image.get("content"):
        return urljoin(base_url, og_image["content"])

    # 3. Check <link rel="icon"> or <link rel="shortcut icon">
    for rel_type in ["icon", "shortcut icon"]:
        link = soup.find("link", rel=lambda x: x and rel_type in x)
        if link and link.get("href"):
            return urljoin(base_url, link["href"])

    # 4. Fallback: favicon
    return urljoin(base_url, "/favicon.ico")

async def process_domain(session, domain):
    """
    Process one domain: fetch HTML and extract logo URL.
    """
    url = f"http://{domain}"
    html = await fetch_html(session, url)
    if html:
        logo_url = extract_logo_url(html, url)
        return domain, normalize_url(logo_url)
    return domain, ""

async def main():
    """
    Main entry point: read domains, process in parallel, save results.
    """
    # Load Parquet file and extract domain column
    df = pd.read_parquet(INPUT_FILE)
    domains = df["domain"].dropna().astype(str).str.strip().tolist()

    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [process_domain(session, domain) for domain in domains]
        results = await asyncio.gather(*tasks)

    # Save results to CSV
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "logo_url"])
        writer.writerows(results)

if __name__ == "__main__":
    asyncio.run(main())
