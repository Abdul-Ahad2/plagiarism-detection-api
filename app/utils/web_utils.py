from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup
import time
from app.config import API_KEY, SEARCH_ENGINE_ID
import logging
import hashlib
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import re
import signal
from contextlib import contextmanager
from ..logger import logger
logger.info("Scraper started")

# Optional fallback
try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except Exception:
    CLOUDSCRAPER_AVAILABLE = False

# ---- OPTIMIZED Configuration ----
MAX_WORKERS = 8  # Increased for parallelism
REQUEST_TIMEOUT = 5  # Reduced from 6 - fail faster
PER_URL_TIMEOUT = 8  # Reduced from 10
POLITENESS_DELAY = 0.1  # Reduced from 0.25
GOOGLE_NUM_DEFAULT = 10
MIN_TEXT_LENGTH = 700
GOOGLE_API_TIMEOUT = 6
BRAVE_API_KEY = "BSAE_jMY2tpTa_jYwCkcaiddxmzLs7m"
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

# Blacklist slow/unscrapeable domains
BLACKLIST_DOMAINS = {
    'neurips.cc',
    'icml.cc', 
    'jmlr.org',
    'researchgate.net',
    'arxiv.org',
    'springer.com',
    'nature.com',
    'nips.cc',
    'iccv2023.thecvf.com'
}

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scraper")

# ---- Session ----
def _make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=1,  # Only 1 retry - fail fast
        backoff_factor=0.1,  # Minimal backoff
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        respect_retry_after_header=False  # Don't wait for Retry-After
    )
    s.mount("https://", HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20))
    s.mount("http://", HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20))
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    })
    return s

_SESSION = _make_session()

# ---- Helpers ----
def _normalize_whitespace(s: str) -> str:
    return " ".join(s.split())

def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _clean_soup(soup: BeautifulSoup, prefer_main: bool = True, max_chars: Optional[int] = None) -> str:
    for junk in soup(["script", "style", "nav", "footer", "noscript", "header"]):
        junk.decompose()
    parts = []
    if prefer_main:
        main = soup.find(["main", "article", "section"])
        if main:
            elems = main.find_all(["p", "h1", "h2", "h3", "li"])
        else:
            elems = soup.find_all(["p", "h1", "h2", "h3"])
    else:
        elems = soup.find_all(["p", "li"])
    for el in elems:
        t = el.get_text(separator=" ", strip=True)
        if t and len(t) > 30:
            parts.append(t)
    text = _normalize_whitespace(" ".join(parts))
    if max_chars and len(text) > max_chars:
        return text[:max_chars]
    return text

# ---- Google Search ----
@lru_cache(maxsize=256)
def google_search(query: str, num_results: int = 10):
    if not API_KEY or not SEARCH_ENGINE_ID:
        logger.warning("Missing API_KEY or SEARCH_ENGINE_ID in config.py")
        return []

    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": API_KEY, "cx": SEARCH_ENGINE_ID, "q": query, "num": num_results}
    try:
        r = _SESSION.get(url, params=params, timeout=GOOGLE_API_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", []) or []
        out = []
        for i in items:
            out.append({
                "link": i.get("link"),
                "title": i.get("title", ""),
                "snippet": i.get("snippet", "")
            })
        logger.info(f"google_search: got {len(out)} items for '{query[:60]}'")
        return out
    except Exception as e:
        logger.warning(f"google_search failed: {e}")
        return []

# ---- IMPROVED: Cloudscraper with early exit ----
def scrape_with_cloudscraper(url: str, timeout: int = 8):
    if not CLOUDSCRAPER_AVAILABLE:
        return ""
    try:
        logger.debug(f"cloudscraper: GET {url}")
        scraper = cloudscraper.create_scraper()
        r = scraper.get(url, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        text = _clean_soup(soup, prefer_main=True, max_chars=20000)  # Reduced from 25k
        if text and len(text) > MIN_TEXT_LENGTH:
            logger.info(f"   ✅ Scraped {len(text)} chars (cloudscraper) for {url}")
            return text
        return ""
    except Exception as e:
        logger.debug(f"cloudscraper error for {url}: {e}")
        return ""

# ---- IMPROVED: Playwright with hard timeout ----
def scrape_with_playwright(url: str, timeout: int = 8):
    """Scrape JS-heavy pages with aggressive waiting and content extraction."""
    try:
        logger.debug(f"Playwright: GET {url}")
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True, 
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-gpu"
                ]
            )
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            context.set_default_timeout(timeout * 1000)
            page = context.new_page()
            
            try:
                # Navigate with longer timeout for JS-heavy sites
                page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
                
                # Wait for common content containers to load
                try:
                    page.wait_for_selector("article, [role='article'], .post-content, .blog-content, main", timeout=3000)
                except:
                    pass  # Selector may not exist, continue anyway
                
                # Extended wait for dynamic content
                page.wait_for_timeout(3000)  # 3 seconds for rendering
                
                # Scroll down to load lazy-loaded images/content
                page.evaluate("""
                    async () => {
                        await new Promise((resolve) => {
                            let totalHeight = 0;
                            const distance = 100;
                            const timer = setInterval(() => {
                                window.scrollBy(0, distance);
                                totalHeight += distance;
                                if (totalHeight >= document.body.scrollHeight) {
                                    clearInterval(timer);
                                    resolve();
                                }
                            }, 100);
                        });
                    }
                """)
                
                # Wait after scrolling
                page.wait_for_timeout(1500)
                
                # Scroll back to top
                page.evaluate("window.scrollTo(0, 0)")
                page.wait_for_timeout(500)
                
                html = page.content()
            finally:
                context.close()
                browser.close()
            
            soup = BeautifulSoup(html, "html.parser")
            text = _clean_soup(soup, prefer_main=True, max_chars=25000)
            
            if text and len(text) > MIN_TEXT_LENGTH:
                logger.info(f"   ✅ Scraped {len(text)} chars (Playwright) for {url}")
                return text
            
            # If we got very little content, try a less aggressive cleanup
            logger.debug(f"Initial extraction got {len(text)} chars, trying aggressive extraction")
            text_aggressive = _clean_soup(soup, prefer_main=False, max_chars=25000)
            if text_aggressive and len(text_aggressive) > MIN_TEXT_LENGTH and len(text_aggressive) > len(text):
                logger.info(f"   ✅ Scraped {len(text_aggressive)} chars (Playwright aggressive) for {url}")
                return text_aggressive
            
            logger.debug(f"Playwright extraction minimal for {url}: {len(text)} chars")
            return text if text else ""
            
    except (PlaywrightTimeoutError, Exception) as e:
        logger.debug(f"Playwright error for {url}: {e}")
        return ""

# ---- IMPROVED: Smart fallback strategy ----
def scrape_page(url: str, timeout: int = 5):
    """Scrape pages in order: requests → cloudscraper → Playwright."""
    try:
        domain = urlparse(url).netloc.lower()

        # Skip blacklisted domains entirely
        if any(bd in domain for bd in BLACKLIST_DOMAINS):
            logger.info(f"Skipping blacklisted domain: {domain}")
            return ""

        # --- 1️⃣ Requests (fastest) ---
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        try:
            logger.debug(f"requests: GET {url}")
            r = _SESSION.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            text = _clean_soup(soup, prefer_main=True, max_chars=20000)
            if text and len(text) > MIN_TEXT_LENGTH:
                logger.info(f"   ✅ Scraped {len(text)} chars (requests) for {url}")
                return text
        except Exception as e:
            logger.debug(f"requests failed for {url}: {e}")

        # --- 2️⃣ Cloudscraper (medium) ---
        if CLOUDSCRAPER_AVAILABLE:
            try:
                logger.debug(f"cloudscraper: GET {url}")
                scraper = cloudscraper.create_scraper()
                r = scraper.get(url, timeout=timeout)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                text = _clean_soup(soup, prefer_main=True, max_chars=20000)
                if text and len(text) > MIN_TEXT_LENGTH:
                    logger.info(f"   ✅ Scraped {len(text)} chars (cloudscraper) for {url}")
                    return text
            except Exception as e:
                logger.debug(f"cloudscraper failed for {url}: {e}")

        # --- 3️⃣ Playwright (heaviest) ---
        try:
            logger.debug(f"Playwright: GET {url}")
            res = scrape_with_playwright(url, timeout=12)  # Hard timeout for all URLs
            if res and len(res) > MIN_TEXT_LENGTH:
                return res
        except Exception as e:
            logger.debug(f"Playwright failed for {url}: {e}")

    except Exception as e:
        logger.debug(f"All scrapers failed for {url}: {e}")

    return ""


# ---- IMPROVED: Parallel multi-query fetch ----
def fetch_sources(query: str, num_results: int = 10):
    logger.info(f"Fetching sources for query: '{query[:60]}'")
    items = google_search(query, num_results=num_results)
    if not items:
        logger.warning("No URLs returned from Google Search")
        return []

    urls = [it["link"] for it in items if it.get("link")]
    snippets = {it["link"]: it.get("snippet", "") for it in items if it.get("link")}
    domain_last_hit: Dict[str, float] = {}

    def _scrape_task(u: str) -> Dict:
        try:
            dom = urlparse(u).netloc
            last = domain_last_hit.get(dom, 0.0)
            now = time.time()
            delta = now - last
            if delta < POLITENESS_DELAY:
                time.sleep(POLITENESS_DELAY - delta)
            domain_last_hit[dom] = time.time()
            logger.info(f"Scraping URL: {u}")

            text = scrape_page(u, timeout=REQUEST_TIMEOUT)
            if text:
                return {"url": u, "content": text, "method": "scraped", "len": len(text), "hash": _text_hash(text[:4000])}
            else:
                return {"url": u, "content": "", "method": "failed", "len": 0, "hash": ""}
        except Exception as e:
            logger.debug(f"Exception in _scrape_task for {u}: {e}")
            return {"url": u, "content": "", "method": "error", "len": 0, "hash": ""}


    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_to_url = {ex.submit(_scrape_task, u): u for u in urls}
        for fut in as_completed(future_to_url):
            u = future_to_url[fut]
            try:
                r = fut.result(timeout=PER_URL_TIMEOUT)
                if r.get("content"):
                    results.append(r)
            except FuturesTimeoutError:
                logger.warning(f"Timeout scraping {u}")
            except Exception as e:
                logger.warning(f"Error scraping {u}: {e}")

    if not results:
        logger.info("No pages scraped; using snippets")
        for u in urls:
            snip = snippets.get(u, "")
            if snip:
                results.append({"url": u, "content": snip, "method": "google_snippet", "len": len(snip), "hash": _text_hash(snip)})
        return results

    # Deduplicate
    seen: Dict[str, Dict] = {}
    for r in results:
        h = r.get("hash") or _text_hash(r.get("content", ""))
        cur = seen.get(h)
        if not cur or (r.get("len", 0) > cur.get("len", 0)):
            seen[h] = r

    deduped = list(seen.values())

    # Fill from snippets if needed
    if len(deduped) < num_results:
        for u in urls:
            if any(x["url"] == u for x in deduped):
                continue
            snip = snippets.get(u, "")
            if snip:
                deduped.append({"url": u, "content": snip, "method": "google_snippet", "len": len(snip), "hash": _text_hash(snip)})
            if len(deduped) >= num_results:
                break

    logger.info(f"Fetched {len(deduped)} sources for query")
    return deduped

# ---- IMPROVED: Multi-query fetch - parallel processing ----
def fetch_sources_multi_query(text: str, num_results: int = 10) -> List[Dict[str, str]]:
    from app.utils.lexical_utils import get_meaningful_sentences

    logger.info("Building sentence-pair-based queries for multi-query fetch")
    sentences = get_meaningful_sentences(text)
    if not sentences or len(sentences) < 2:
        logger.warning("Not enough sentences; falling back to single query")
        return fetch_sources(text, num_results=num_results)

    queries = [
        " ".join(sentences[:2]),
        " ".join(sentences[len(sentences)//2 - 1 : len(sentences)//2 + 1]),
        " ".join(sentences[-2:])
    ]
    logger.info(f"Built {len(queries)} queries from document")

    all_sources: Dict[str, Dict] = {}
    DOC_EXTENSIONS = [".pdf", ".doc", ".docx", ".odf", ".xls", ".xlsx", ".ppt", ".pptx"]

    def _process_url(u: str, items: List) -> Optional[Dict]:
        """Process single URL with better retry logic."""
        if u in all_sources:
            return None
        if any(ext in u.lower() for ext in DOC_EXTENSIONS):
            logger.info(f"Skipping document URL: {u}")
            return None
        
        logger.info(f"Scraping URL: {u}")
        
        try:
            # Try scraping the page
            text_content = scrape_page(u, timeout=REQUEST_TIMEOUT)
            if text_content and len(text_content) > MIN_TEXT_LENGTH:
                logger.info(f"   ✅ Scraped {len(text_content)} chars for {u}")
                return {"url": u, "content": text_content, "source_url": u}
            
            # If scraping failed, try snippet as fallback
            snippet = next((it.get("snippet", "") for it in items if it.get("link") == u), "")
            if snippet and len(snippet) > 50:
                logger.info(f"   ⚠️ Using snippet ({len(snippet)} chars) for {u}")
                return {"url": u, "content": snippet, "source_url": u}
            
            # If no content at all, log and skip
            logger.warning(f"   ❌ No content extracted for {u}")
            return None
            
        except Exception as e:
            logger.debug(f"Error scraping {u}: {e}")
            return None

    # Process queries in parallel
    with ThreadPoolExecutor(max_workers=3) as query_ex:
        query_futures = {}
        for qi, q in enumerate(queries):
            def _fetch_query_urls(query_text: str, qi_val: int):
                logger.info(f"Query {qi_val}/{len(queries)}: '{query_text[:60]}'")
                items = google_search(query_text, num_results=num_results)
                if not items:
                    return []
                
                urls = [it["link"] for it in items if it.get("link")]
                
                # Process URLs for this query in parallel
                with ThreadPoolExecutor(max_workers=4) as url_ex:
                    url_futures = {url_ex.submit(_process_url, u, items): u for u in urls}
                    for fut in as_completed(url_futures):
                        try:
                            result = fut.result(timeout=PER_URL_TIMEOUT)
                            if result:
                                all_sources[result['url']] = result
                                logger.info(f"   ✅ Added: {result['url'][:50]}")
                        except FuturesTimeoutError:
                            logger.warning(f"Timeout on URL")
                        except Exception as e:
                            logger.debug(f"Error: {e}")
                        time.sleep(0.05)
                
                time.sleep(0.15)
                return list(all_sources.values())
            
            query_futures[query_ex.submit(_fetch_query_urls, q, qi)] = qi

        try:
            for fut in as_completed(query_futures):
                try:
                    fut.result()
                except Exception as e:
                    logger.warning(f"Query processing error: {e}")
        except TimeoutError:
            logger.warning("Multi-query fetch timeout - returning partial results")
            for fut in query_futures:
                fut.cancel()

    res = list(all_sources.values())
    logger.info(f"Total unique sources from multi-query: {len(res)}")
    return res


def fetch_brave_sources(query: str, num_results: int = 5) -> list[dict]:
    """Fetch from Brave Search API."""
    headers = {
        "Accept": "application/json",
        "X-API-KEY": BRAVE_API_KEY
    }
    params = {"q": query, "count": num_results}

    try:
        resp = requests.get(BRAVE_ENDPOINT, headers=headers, params=params, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("webPages", []):
            results.append({
                "title": item.get("url") or "Unknown",
                "content": item.get("snippet") or "",
            })
        return results
    except Exception as e:
        logging.warning(f"Brave search failed for query '{query}': {e}")
        return []

def prepare_brave_query(text: str, min_len: int = 20, max_len: int = 200) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    if len(t) < min_len:
        return None
    if len(t) > max_len:
        t = t[:max_len].rsplit(" ", 1)[0]
    return t