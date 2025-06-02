import os
import base64
import asyncio
import requests
import pandas as pd
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm

df = pd.read_parquet("../logos.snappy.parquet")
domains = df.iloc[:, 0].dropna().unique().tolist() #used unique() to avoid duplicates

output_dir = "../logos"
os.makedirs(output_dir, exist_ok=True)

def is_valid_image(content):
    """Check if content is a valid image by checking magic bytes"""
    if len(content) < 10:
        return False
    
    # Check for common image file signatures
    image_signatures = [
        b'\xff\xd8\xff',  # JPEG
        b'\x89PNG\r\n\x1a\n',  # PNG
        b'GIF87a',  # GIF87a
        b'GIF89a',  # GIF89a
        b'RIFF',  # WebP (starts with RIFF)
    ]
    
    for sig in image_signatures:
        if content.startswith(sig):
            return True
    
    # Check for SVG (XML-based)
    try:
        content_str = content.decode('utf-8', errors='ignore').strip().lower()
        if content_str.startswith('<?xml') or content_str.startswith('<svg'):
            return True
    except:
        pass
    
    return False

def generate_advanced_queries(domain):
    """Generate single optimized query for speed"""
    queries = [
        f"{domain} logo",  # Single, simple query
    ]
    return queries

def download_bing_image(query, download_path):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        
        # Single query for speed
        domain = query
        search_query = f"{domain} logo"
        
        print(f"Trying query: {search_query}")
        
        params = {
            "q": search_query,
            "form": "HDRSC2",
            "first": "1",
            "tsc": "ImageBasicHover"
        }
        search_url = "https://www.bing.com/images/search"
        response = requests.get(search_url, headers=headers, params=params, timeout=15)  # Relaxed timeout
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'lxml')
        image_elements = soup.find_all("a", class_="iusc")

        if not image_elements:
            print(f"No images found for {domain}")
            return False

        # Try for first 2 images
        for element in image_elements[:2]:
            try:
                metadata = json.loads(element.get("m"))
                image_url = metadata.get("murl")

                if not image_url:
                    continue

                # Download the image with reduced timeout
                image_response = requests.get(image_url, headers=headers, timeout=12)
                image_response.raise_for_status()

                # Quick validation - just check size
                if len(image_response.content) < 30:  # Skip very small images
                    continue

                # Simple extension detection
                ext = 'jpg'
                content_type = image_response.headers.get('content-type', '')
                if 'png' in content_type:
                    ext = 'png'
                elif 'svg' in content_type:
                    ext = 'svg'

                # Save the image
                image_name = f"{domain.replace('.', '_')}.{ext}"
                image_path = os.path.join(download_path, image_name)

                with open(image_path, "wb") as f:
                    f.write(image_response.content)

                print(f"✓ Downloaded: {domain}")
                return True

            except Exception as e:
                continue

        print(f"✗ Failed: {domain}")
        return False

    except Exception as e:
        print(f"✗ Error: {domain}")
        return False

def main():
    start_time = time.time()
    successful_downloads = 0
    total_domains = len(domains)
    
    print(f"Starting FAST logo extraction for {total_domains} domains...")
    
    def process_domain_wrapper(domain):
        nonlocal successful_downloads
        result = download_bing_image(domain, output_dir)
        if result:
            successful_downloads += 1
        return result
    
    # Increased workers for maximum speed
    with ThreadPoolExecutor(max_workers=135) as executor:
        results = list(tqdm(
            executor.map(process_domain_wrapper, domains),
            total=total_domains,
            desc="Downloading logos",
            unit="domain",
            smoothing=0.1
        ))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n{'='*50}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*50}")
    print(f"Total domains processed: {total_domains}")
    print(f"Successfully downloaded logos: {successful_downloads}")
    print(f"Failed downloads: {total_domains - successful_downloads}")
    print(f"Success rate: {(successful_downloads/total_domains)*100:.1f}%")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average time per domain: {elapsed_time/total_domains:.2f} seconds")

if __name__ == "__main__":
    main()