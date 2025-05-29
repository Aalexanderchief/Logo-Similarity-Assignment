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
CSV_OUTPUT_PATH = "outputs/logo_urls_static.csv"
CONCURRENCY = 500  
TIMEOUT = 15 

LOGO_PATTERN = re.compile(r"logo|logotype|brand", re.IGNORECASE)

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".svg", ".webp", ".ico")

def is_valid_image(src: str) -> bool:
    if not src:
        return False
    if src.startswith("data:"):
        return False
    if "1x1" in src or "transparent" in src:
        return False
    return src.lower().endswith(VALID_EXTENSIONS)

async def fetch(session, url, retries=1, backoff=2):
    for attempt in range(retries + 1):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            }
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.text()
        except Exception:
            if attempt < retries:
                await asyncio.sleep(backoff)
            else:
                return None

async def extract_logo_from_domain(session, domain):
    """Extract logo URL from a domain"""
    if not domain or pd.isna(domain):
        return None
        
    domain = str(domain).strip()
    if not domain:
        return None
    
    # Try both HTTPS and HTTP
    for scheme in ["https", "http"]:
        url = f"{scheme}://{domain}"
        html_content = await fetch(session, url)
        if html_content:
            try:
                tree = html.fromstring(html_content)
                
                # Look for logo images
                for img in tree.xpath('//img'):
                    for attr in ['@src', '@alt', '@class', '@id']:
                        val = img.xpath(attr)
                        if val and LOGO_PATTERN.search(" ".join(val)):
                            src = img.xpath('@src')
                            if src and is_valid_image(src[0]):
                                return urljoin(url, src[0])
                
                # Also check meta tags for logos (og:image, etc.)
                meta_selectors = [
                    '//meta[@property="og:image"]/@content',
                    '//meta[@property="og:logo"]/@content', 
                    '//meta[@name="twitter:image"]/@content',
                    '//meta[@property="og:image:url"]/@content'
                ]
                
                for selector in meta_selectors:
                    meta_values = tree.xpath(selector)
                    for meta_value in meta_values:
                        if meta_value and ('logo' in meta_value.lower() or is_valid_image(meta_value)):
                            return urljoin(url, meta_value)
                            
            except Exception:
                continue
    
    return None

async def process_domain_with_index(session, idx, domain):
    """Process domain and return with original index"""
    try:
        logo_url = await extract_logo_from_domain(session, domain)
        return (idx, domain, logo_url if logo_url else "")
    except Exception as e:
        return (idx, domain, "")

async def main():
    # Read the parquet file
    df = pd.read_parquet(CSV_INPUT_PATH)
    print(f"Total rows in parquet file: {len(df)}")
    
    # Keep all rows, including duplicates - just filter out null domains
    valid_df = df[df['domain'].notna() & (df['domain'].astype(str).str.strip() != '')]
    print(f"Rows with valid domains: {len(valid_df)}")
    print(f"Rows filtered out (null/empty domains): {len(df) - len(valid_df)}")
    
    # Check for duplicates but don't remove them
    duplicate_count = valid_df['domain'].duplicated().sum()
    if duplicate_count > 0:
        print(f"Found {duplicate_count} duplicate domains (keeping all entries)")
    
    # Get all domains including duplicates with their original indices
    domains_with_indices = [(idx, row['domain']) for idx, row in valid_df.iterrows()]
    print(f"Processing {len(domains_with_indices)} domain entries")
    
    os.makedirs("outputs", exist_ok=True)
    
    # Setup HTTP session
    timeout = aiohttp.ClientTimeout(
        total=TIMEOUT,
        connect=12,
        sock_connect=12,
        sock_read=12
    )
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, limit_per_host=50)
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Create tasks for all domains
        tasks = [process_domain_with_index(session, idx, domain) for idx, domain in domains_with_indices]
        
        # Process all tasks with progress bar
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Extracting logos"):
            result = await coro
            results.append(result)
    
    # Create results dataframe preserving original indices
    results_df = pd.DataFrame(results, columns=['original_index', 'domain', 'logo_url'])
    
    # Merge back with original dataframe to preserve all columns
    final_df = df.copy()
    final_df['logo_url'] = ''  # Initialize logo_url column
    
    # Update logo URLs for processed domains
    for _, row in results_df.iterrows():
        if row['logo_url']:  # Only update if logo was found
            final_df.loc[row['original_index'], 'logo_url'] = row['logo_url']
    
    # Save results with all original data preserved
    final_df.to_csv(CSV_OUTPUT_PATH, index=False)
    
    # Calculate and display statistics
    found_logos = len([r for r in results if r[2]])  # Count non-empty logo URLs
    total_processed = len(domains_with_indices)
    
    print(f"\n✅ Static logo extraction completed!")
    print(f"📊 Results: Found logos for {found_logos}/{total_processed} domains ({found_logos/total_processed*100:.1f}%)")
    print(f"💾 Results saved to: {CSV_OUTPUT_PATH}")
    print(f"📁 Final dataset contains {len(final_df)} rows (including all original data)")

if __name__ == "__main__":
    asyncio.run(main())