"""Kiwix HTTP client and article fetching."""

import os
import subprocess
import sys
import time
from typing import List, Optional, Tuple
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from kiwix_chat.config import KIWIX_BASE_URL
from kiwix_chat.models import ArticleLink
from kiwix_chat.kiwix.parser import HTMLParserWithLinks, KiwixSearchParser

# Import caching module
try:
    from kiwix_cache import (
        get_cached_article, cache_article,
        get_cached_search, cache_search
    )
    CACHING_ENABLED = True
except ImportError:
    CACHING_ENABLED = False
    # Define no-op functions if caching not available
    def get_cached_article(*args): return None
    def cache_article(*args): pass
    def get_cached_search(*args): return None
    def cache_search(*args): pass


# Module-level variable to store ZIM file path from CLI args
_global_zim_file_path: Optional[str] = None


def set_global_zim_file_path(path: Optional[str]) -> None:
    """Set the global ZIM file path."""
    global _global_zim_file_path
    _global_zim_file_path = path


def http_get(url: str, timeout: float = 20.0) -> str:
    """Make HTTP GET request."""
    req = Request(url)
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _detect_language_from_zim(zim_file: str) -> Optional[str]:
    """Detect language from ZIM filename (simple heuristic)."""
    # Extract language code from filename if present
    # Common pattern: wikipedia_XX_all.zim or similar
    filename = os.path.basename(zim_file)
    parts = filename.lower().split('_')
    for part in parts:
        if len(part) == 2 and part.isalpha():
            # Could be a language code
            return part.upper()
    return None


def _detect_content_type_from_zim(zim_file: str) -> str:
    """Detect content type from ZIM filename (wikipedia, wiktionary, gutenberg, etc.).
    Returns a user-friendly description of the content type."""
    filename = os.path.basename(zim_file).lower()
    
    # Common patterns in ZIM filenames
    if 'wikipedia' in filename:
        return 'Wikipedia articles'
    elif 'wiktionary' in filename:
        return 'Wiktionary entries'
    elif 'gutenberg' in filename:
        return 'Project Gutenberg books'
    elif 'stackexchange' in filename or 'stack_exchange' in filename:
        return 'Stack Exchange content'
    elif 'ted' in filename:
        return 'TED talks'
    elif 'wikibooks' in filename:
        return 'Wikibooks content'
    elif 'wikisource' in filename:
        return 'Wikisource content'
    elif 'wikiquote' in filename:
        return 'Wikiquote content'
    elif 'wikivoyage' in filename:
        return 'Wikivoyage content'
    elif 'wikiversity' in filename:
        return 'Wikiversity content'
    else:
        # Generic fallback - extract first meaningful word or use generic term
        parts = filename.replace('.zim', '').split('_')
        for part in parts:
            if part and len(part) > 3 and part.isalpha():
                return f'{part.capitalize()} content'
        return 'Kiwix content'


def get_current_zim_file_path() -> Optional[str]:
    """Get the current ZIM file path being used.
    Returns the global path if set, otherwise tries to auto-detect.
    Prioritizes ZIM files in the app folder (where kiwix_chat.py is located)."""
    global _global_zim_file_path
    
    if _global_zim_file_path and os.path.isfile(_global_zim_file_path):
        return _global_zim_file_path
    
    # Try to auto-detect - prioritize app folder first
    # Find the app folder (where kiwix_chat.py is located)
    app_folder = None
    try:
        # Method 1: Go up from this file to find project root
        # kiwix_chat/kiwix/client.py -> kiwix_chat/kiwix -> kiwix_chat -> project root
        current_file = os.path.abspath(__file__)
        script_dir = os.path.dirname(current_file)  # kiwix_chat/kiwix
        script_dir = os.path.dirname(script_dir)    # kiwix_chat
        candidate_folder = os.path.dirname(script_dir)  # project root (where kiwix_chat.py should be)
        
        # Verify kiwix_chat.py exists in this folder
        if os.path.isfile(os.path.join(candidate_folder, "kiwix_chat.py")):
            app_folder = candidate_folder
        else:
            # Method 2: Check if we're already in the project root
            # Look for kiwix_chat directory to confirm
            if os.path.isdir(os.path.join(candidate_folder, "kiwix_chat")):
                app_folder = candidate_folder
            else:
                # Method 3: Try current working directory
                cwd = os.getcwd()
                if os.path.isfile(os.path.join(cwd, "kiwix_chat.py")):
                    app_folder = cwd
                elif os.path.isdir(os.path.join(cwd, "kiwix_chat")):
                    app_folder = cwd
                else:
                    # Fallback: use candidate folder anyway
                    app_folder = candidate_folder
    except Exception:
        # Fallback: use current working directory
        app_folder = os.getcwd()
    
    # Search order: app folder first, then other locations
    search_dirs = []
    if app_folder and os.path.isdir(app_folder):
        search_dirs.append(app_folder)
    search_dirs.extend([os.path.expanduser("~"), "/usr/share/kiwix"])
    
    # Search for ZIM files
    for search_dir in search_dirs:
        if os.path.isdir(search_dir):
            try:
                # Get all .zim files and sort (for consistent selection)
                zim_files = [
                    os.path.join(search_dir, filename)
                    for filename in os.listdir(search_dir)
                    if filename.endswith('.zim') and os.path.isfile(os.path.join(search_dir, filename))
                ]
                if zim_files:
                    # Return first ZIM file found (sorted for consistency)
                    zim_files.sort()
                    return zim_files[0]
            except (OSError, PermissionError):
                continue
    
    return None


def get_zim_content_description() -> str:
    """Get a user-friendly description of the current ZIM content type.
    Returns a generic description if no ZIM file is found."""
    zim_file = get_current_zim_file_path()
    if zim_file:
        return _detect_content_type_from_zim(zim_file)
    return 'Kiwix content'


def _score_relevance(href: str, query: str) -> float:
    """Score how relevant an article href is to the query.
    Returns a score from 0.0 to 1.0, higher is more relevant."""
    # Extract article name from href (format: /A/Article_Name or /wiki/Article_Name)
    article_name = href.split('/')[-1].replace('_', ' ').lower()
    query_lower = query.lower()
    
    score = 0.0
    
    # Exact match gets highest score
    if article_name == query_lower:
        return 1.0
    
    # Check word overlap
    query_words = set(query_lower.split())
    article_words = set(article_name.split())
    
    if query_words and article_words:
        overlap = len(query_words & article_words)
        word_score = overlap / max(len(query_words), len(article_words))
        score += word_score * 0.6
    
    # Check substring match
    if query_lower in article_name or article_name in query_lower:
        score += 0.3
    
    # Check if query words appear in order in article name
    query_word_list = query_lower.split()
    article_word_list = article_name.split()
    if len(query_word_list) > 1 and len(article_word_list) >= len(query_word_list):
        # Check if query words appear consecutively in article
        for i in range(len(article_word_list) - len(query_word_list) + 1):
            if article_word_list[i:i+len(query_word_list)] == query_word_list:
                score += 0.1
                break
    
    return min(score, 1.0)


def _auto_start_kiwix(zim_file_path: Optional[str] = None) -> bool:
    """Attempt to auto-start Kiwix server if not running.
    Returns True if Kiwix is now available, False otherwise.
    
    Args:
        zim_file_path: Optional path to ZIM file. If None, uses global _global_zim_file_path
                      or auto-detects first .zim file found.
    """
    # Check if already running
    try:
        test_html = http_get(f"{KIWIX_BASE_URL}/")
        return True
    except Exception:
        pass
    
    # Use provided path, global path, or auto-detect
    zim_file = zim_file_path or _global_zim_file_path
    
    # If not specified, use get_current_zim_file_path() which prioritizes app folder
    if not zim_file:
        zim_file = get_current_zim_file_path()
    
    if not zim_file:
        return False
    
    # Check if kiwix-serve is available
    try:
        subprocess.run(["kiwix-serve", "--version"], capture_output=True, check=True, timeout=5)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False
    
    # Kill any existing kiwix-serve on port 8081
    try:
        subprocess.run(["pkill", "-f", "kiwix-serve.*8081"], capture_output=True, timeout=2)
        time.sleep(1)
    except Exception:
        pass
    
    # Start kiwix-serve
    try:
        # Detect language for better UX
        detected_lang = _detect_language_from_zim(zim_file)
        lang_info = f" ({detected_lang})" if detected_lang else ""
        print(f"[kiwix] Auto-starting Kiwix server{lang_info}...", file=sys.stderr)
        process = subprocess.Popen(
            ["kiwix-serve", "--port=8081", zim_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        # Wait for server to start (up to 30 seconds)
        for _ in range(30):
            time.sleep(1)
            try:
                test_html = http_get(f"{KIWIX_BASE_URL}/")
                lang_msg = f" with {detected_lang} content" if detected_lang else ""
                print(f"[kiwix] ✓ Kiwix server auto-started successfully{lang_msg}", file=sys.stderr)
                return True
            except Exception:
                # Check if process is still running
                if process.poll() is not None:
                    # Process died
                    break
        
        # If we get here, it didn't start
        try:
            process.terminate()
            process.wait(timeout=2)
        except Exception:
            pass
        return False
    except Exception as e:
        print(f"[kiwix] Failed to auto-start: {e}", file=sys.stderr)
        return False


def kiwix_search_first_href(query: str) -> Optional[str]:
    """Search Kiwix and return the best matching article href.
    Tries multiple search variations and scores results by relevance."""
    # Check cache first
    if CACHING_ENABLED:
        cached_href = get_cached_search(query)
        if cached_href is not None:
            return cached_href
    
    # Try multiple search variations (prioritize more likely matches first)
    search_variations = [
        query,  # Original query
        query.title(),  # Title case: "michael jackson" -> "Michael Jackson"
        query.replace(" ", "_"),  # Underscores: "michael_jackson"
        query.replace(" ", "_").title(),  # Title case with underscores
        query.replace(" ", ""),  # No spaces: "basketball"
        query.replace(" ", "").title(),  # No spaces, title case
        # Additional variations
        query.capitalize(),  # First word capitalized
        query.upper(),  # All caps (less common but sometimes works)
    ]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for var in search_variations:
        if var not in seen and var:
            seen.add(var)
            unique_variations.append(var)
    
    # Check Kiwix connectivity once (static variable)
    if not hasattr(kiwix_search_first_href, '_kiwix_checked'):
        try:
            test_html = http_get(f"{KIWIX_BASE_URL}/")
            print(f"[kiwix] Kiwix server is accessible at {KIWIX_BASE_URL}", file=sys.stderr)
            kiwix_search_first_href._kiwix_checked = True
            kiwix_search_first_href._kiwix_available = True
        except Exception as e:
            print(f"[kiwix] ERROR: Kiwix server not accessible at {KIWIX_BASE_URL}: {e}", file=sys.stderr)
            # Try to auto-start
            if _auto_start_kiwix():
                kiwix_search_first_href._kiwix_checked = True
                kiwix_search_first_href._kiwix_available = True
            else:
                print(f"[kiwix] Start it manually with: kiwix-serve --port=8081 <path-to-zim-file>", file=sys.stderr)
                kiwix_search_first_href._kiwix_checked = True
                kiwix_search_first_href._kiwix_available = False
                return None
    
    if not kiwix_search_first_href._kiwix_available:
        return None
    
    # Collect all results with scores
    scored_results = []
    
    for search_query in unique_variations:
        try:
            # Properly URL-encode the query
            encoded_query = quote_plus(search_query)
            search_url = f"{KIWIX_BASE_URL}/search?pattern={encoded_query}"
            html = http_get(search_url)
            
            # Use proper HTML parser
            try:
                parser = KiwixSearchParser()
                parser.feed(html)
                
                if parser.hrefs:
                    # Score each result
                    for href in parser.hrefs:
                        score = _score_relevance(href, query)
                        scored_results.append((score, href, search_query))
            except Exception as parse_error:
                print(f"[kiwix] Error parsing search results for '{search_query}': {parse_error}", file=sys.stderr)
                continue  # Try next variation
        except Exception as e:
            continue  # Try next variation
    
    if not scored_results:
        return None
    
    # Sort by score (highest first), then by search order (earlier variations preferred)
    scored_results.sort(key=lambda x: (-x[0], x[2]))  # Negative score for descending
    
    # Remove duplicates (keep first occurrence)
    seen_hrefs = set()
    best_href = None
    best_score = 0.0
    best_query = query
    
    for score, href, search_query in scored_results:
        if href not in seen_hrefs:
            seen_hrefs.add(href)
            best_href = href
            best_score = score
            best_query = search_query
            break
    
    # Minimum relevance threshold - filter out low-relevance matches
    MIN_RELEVANCE_THRESHOLD = 0.25
    
    if best_href and best_score >= MIN_RELEVANCE_THRESHOLD:
        print(f"[kiwix] Found match for '{query}' via '{best_query}' (relevance: {best_score:.2f}): {best_href}", file=sys.stderr)
        # Cache the result
        if CACHING_ENABLED:
            cache_search(query, best_href)
        return best_href
    
    # Fallback to first result if all were duplicates, but only if above threshold
    if scored_results:
        best_score, best_href, best_query = scored_results[0]
        if best_score >= MIN_RELEVANCE_THRESHOLD:
            print(f"[kiwix] Using first result for '{query}' (relevance: {best_score:.2f}): {best_href}", file=sys.stderr)
            # Cache the result
            if CACHING_ENABLED:
                cache_search(query, best_href)
            return best_href
        else:
            print(f"[kiwix] No relevant match for '{query}' (best relevance: {best_score:.2f} < {MIN_RELEVANCE_THRESHOLD})", file=sys.stderr)
    
    # Cache None result (not found)
    if CACHING_ENABLED:
        cache_search(query, None)
    return None


def kiwix_fetch_article(query: str, max_chars: int) -> Optional[Tuple[str, List[ArticleLink], str]]:
    """Fetch article text and links from Kiwix. Returns (text, links, href) or None."""
    # Check cache first
    if CACHING_ENABLED:
        cached_result = get_cached_article(query)
        if cached_result is not None:
            return cached_result
    
    href = kiwix_search_first_href(query)
    if not href:
        return None
    try:
        html = http_get(f"{KIWIX_BASE_URL}{href}")
    except Exception:
        return None
    try:
        parser = HTMLParserWithLinks()
        parser.feed(html)
        text = parser.get_text()
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[truncated]"
        links = parser.get_links()
        result = (text, links, href) if text else None
        
        # Cache the result
        if result and CACHING_ENABLED:
            cache_article(query, result)
        
        return result
    except Exception as e:
        print(f"[kiwix] Error parsing HTML for '{query}': {e}", file=sys.stderr)
        return None

