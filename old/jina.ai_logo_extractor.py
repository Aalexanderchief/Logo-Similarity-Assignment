import requests
import pandas as pd
import re
import os
from urllib.parse import urljoin, urlparse
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import threading

# Read domains from CSV
df = pd.read_parquet("../logos.snappy.parquet")
domains = df.iloc[:, 0].dropna().tolist()

# Create output directory
output_dir = "logos_jina"
os.makedirs(output_dir, exist_ok=True)

# API Key rotation pool with all 4 keys
API_KEYS = [
 
]

# Enhanced settings for better performance
MAX_REQUESTS_PER_KEY_PER_MINUTE = 500  # Per API key limit
KEY_COUNT = len(API_KEYS)
OPTIMAL_WORKERS = 150  # Increased for better stability and performance

# Per-API-key tracking
api_key_usage = {key: {"requests": 0, "minute": time.time() // 60, "lock": threading.Lock()} for key in API_KEYS}
key_index_lock = threading.Lock()
current_key_index = 0

print(f"Using {OPTIMAL_WORKERS} workers with {KEY_COUNT} API keys")
print(f"Maximum requests per key per minute: {MAX_REQUESTS_PER_KEY_PER_MINUTE}")

def get_next_api_key():
    """Get the next API key with rate limiting"""
    global current_key_index
    
    with key_index_lock:
        attempts = 0
        while attempts < KEY_COUNT:
            key = API_KEYS[current_key_index]
            current_key_index = (current_key_index + 1) % KEY_COUNT
            
            with api_key_usage[key]["lock"]:
                now_minute = time.time() // 60
                
                # Reset counter if we're in a new minute
                if now_minute > api_key_usage[key]["minute"]:
                    api_key_usage[key]["minute"] = now_minute
                    api_key_usage[key]["requests"] = 0
                
                # Check if this key has capacity
                if api_key_usage[key]["requests"] < MAX_REQUESTS_PER_KEY_PER_MINUTE:
                    api_key_usage[key]["requests"] += 1
                    return key
            
            attempts += 1
        
        # If all keys are at limit, wait and retry with first key
        time.sleep(1)
        return API_KEYS[0]

def extract_logo_from_page(domain):
    """Extract logo image from a domain using Jina AI"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Get the next available API key
            api_key = get_next_api_key()
            
            url = f"https://r.jina.ai/https://{domain}"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "X-Engine": "browser",
                "X-Remove-Selector": "footer, sidebar",
                "X-Target-Selector": "header, nav, .logo, #logo, .brand, img[alt*='logo' i], img[src*='logo' i]",
                "X-With-Images-Summary": "true",
                "X-Cache-Opt-Out": "true",
                "X-Timeout": "15"
            }

            response = requests.get(url, headers=headers, timeout=10)
            
            # Handle rate limiting specifically
            if response.status_code == 429:
                wait_time = min(2 ** attempt, 30)  # Progressive backoff, max 30s
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            
            page_content = response.text
            
            # Look for logo-related patterns in both HTML and Markdown
            logo_patterns = [
                # HTML img tags with logo in alt or src
                r'<img[^>]*(?:alt=["\'][^"\']*logo[^"\']*["\']|src=["\'][^"\']*logo[^"\']*["\'])[^>]*src=["\']([^"\']+)["\']',
                r'<img[^>]*src=["\']([^"\']+)["\'][^>]*(?:alt=["\'][^"\']*logo[^"\']*["\'])',
                r'<img[^>]*(?:alt=["\'][^"\']*["\']|src=["\'][^"\']*\.(?:png|jpg|jpeg|svg|webp))[^>]*src=["\']([^"\']+)["\']',
                
                # Markdown format
                r'!\[.*?logo.*?\]\((.*?)\)',  # Markdown images with "logo" in alt text
                r'!\[.*?brand.*?\]\((.*?)\)',  # Markdown images with "brand" in alt text
                r'https?://[^\s\)]+(?:logo|brand|header)[^\s\)]*\.(?:png|jpg|jpeg|svg|webp)',  # Direct logo URLs
            ]
            
            image_url = None
            
            # Search for logo patterns
            for pattern in logo_patterns:
                matches = re.findall(pattern, page_content, re.IGNORECASE)
                if matches:
                    image_url = matches[0]
                    break
            
            # If no specific logo pattern found, look for any images in header/nav area
            if not image_url:
                # Try to find any image tag in the content
                general_img_pattern = r'<img[^>]*src=["\']([^"\']+(?:\.png|\.jpg|\.jpeg|\.svg|\.webp))["\'][^>]*>'
                matches = re.findall(general_img_pattern, page_content, re.IGNORECASE)
                if matches:
                    # Take the first image which is likely to be the logo
                    image_url = matches[0]
                else:
                    # Fallback to Markdown format
                    header_img_pattern = r'!\[.*?\]\((https?://[^\s\)]+\.(?:png|jpg|jpeg|svg|webp))\)'
                    matches = re.findall(header_img_pattern, page_content, re.IGNORECASE)
                    if matches:
                        image_url = matches[0]
            
            if image_url:
                return download_image(image_url, domain)
            else:
                return False
                
        except requests.exceptions.Timeout as e:
            if attempt < max_retries - 1:
                time.sleep(1 + attempt)  # Progressive backoff
                continue
            else:
                return False
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            else:
                return False
    
    return False

def download_image(image_url, domain):
    """Download the logo image"""
    try:
        # Make sure URL is absolute
        if not image_url.startswith('http'):
            image_url = f"https://{domain}{image_url}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(image_url, headers=headers, timeout=8)
        response.raise_for_status()
        
        # Determine file extension
        content_type = response.headers.get('content-type', '')
        ext = 'jpg'
        if 'png' in content_type:
            ext = 'png'
        elif 'svg' in content_type:
            ext = 'svg'
        elif 'webp' in content_type:
            ext = 'webp'
        
        # Save the image
        filename = f"{domain.replace('.', '_')}.{ext}"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        return True
        
    except Exception as e:
        return False

def main():
    """Process all domains from CSV"""
    start_time = time.time()
    successful_downloads = 0
    total_domains = len(domains)
    
    print(f"Starting Jina AI logo extraction for {total_domains} domains...")
    
    def process_domain_wrapper(domain):
        nonlocal successful_downloads
        result = extract_logo_from_page(domain)
        if result:
            successful_downloads += 1
        return result
    
    # Optimized worker count for better performance
    with ThreadPoolExecutor(max_workers=OPTIMAL_WORKERS) as executor:
        # Use tqdm for progress bar
        results = list(tqdm(
            executor.map(process_domain_wrapper, domains),
            total=total_domains,
            desc="Processing domains",
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

if __name__ == "__main__":
    main()