import asyncio
import aiohttp
import pandas as pd
from lxml import html
from urllib.parse import urljoin
import re
import os
from aiofiles import open as aioopen
from tqdm import tqdm

CSV_INPUT_PATH = "logos.snappy.parquet"
CSV_OUTPUT_PATH = "outputs/logo_urls.csv"
CONCURRENCY = 100
TIMEOUT = 10

LOGO_PATTERN = re.compile(r"logo|logotype|brand", re.IGNORECASE)

async def fetch(session, url):
    try:
        async with session.get(url, timeout=TIMEOUT) as response:
            if response.status == 200:
                return await response.text()
    except:
        return None

async def extract_logo(session, domain):
    for scheme in ["https", "http"]:
        url = f"{scheme}://{domain}"
        html_content = await fetch(session, url)
        if html_content:
            try:
                tree = html.fromstring(html_content)
                for img in tree.xpath('//img'):
                    for attr in ['@src', '@alt', '@class', '@id']:
                        val = img.xpath(attr)
                        if val and LOGO_PATTERN.search(" ".join(val)):
                            src = img.xpath('@src')
                            if src:
                                return domain, urljoin(url, src[0])
            except:
                continue
    return domain, ""

async def worker(sem, session, domain, results):
    async with sem:
        domain, logo_url = await extract_logo(session, domain)
        results[domain] = logo_url

async def main():
    df = pd.read_parquet(CSV_INPUT_PATH)
    domains = df["domain"].dropna().astype(str).tolist()
    os.makedirs("outputs", exist_ok=True)

    sem = asyncio.Semaphore(CONCURRENCY)
    results = {}

    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [worker(sem, session, domain, results) for domain in domains]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Logo extraction"):
            await f

    df_out = pd.DataFrame(results.items(), columns=["domain", "logo_url"])
    df_out.to_csv(CSV_OUTPUT_PATH, index=False)
    found = df_out["logo_url"].astype(bool).sum()
    total = len(df_out)
    print(f"✅ Logos found for {found} out of {total} websites ({found/total*100:.2f}%)")

if __name__ == "__main__":
    asyncio.run(main())
