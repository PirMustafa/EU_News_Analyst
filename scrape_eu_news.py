"""
EU Commission News Scraper
==========================
Pure HTML scraping with pagination - NO RSS feeds.
Visits each page, grabs article links, continues until articles are too old.

How it works:
1. Visit Page 1 of the news listing
2. Grab all article links on that page
3. Calculate URL for Page 2 (?page=1), visit it
4. Repeat until we hit the cutoff date (e.g., Dec 1st)
"""

import json
import os
import requests
from datetime import datetime
from bs4 import BeautifulSoup
import time
import re

# --- CONFIGURATION ---
OUTPUT_FILE = r"D:\NLP_Projects\EU_News_Analyst\eu_news_data.json"
BASE_URL = "https://commission.europa.eu"
NEWS_PAGE = "https://commission.europa.eu/news-and-media/news_en"

# Date cutoff - stop scraping when articles are older than this
CUTOFF_DATE = datetime(2025, 1, 1)  # January 1st, 2025 - Get full year of articles
MAX_PAGES = 200  # Increased limit for more articles

# Headers to mimic a real browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}


def get_page_url(page_num):
    """
    Calculate the URL for a specific page number.
    
    The EU Commission site uses 0-indexed pagination:
    - Page 1 (first) = no parameter
    - Page 2 = ?page=1
    - Page 3 = ?page=2
    """
    if page_num == 1:
        return NEWS_PAGE
    else:
        return f"{NEWS_PAGE}?page={page_num - 1}"


def parse_date(date_str):
    """Parse various date formats from the website."""
    if not date_str:
        return None
    
    date_str = date_str.strip()
    
    # Try different formats the EU site might use
    formats = [
        '%d %B %Y',      # "02 February 2026"
        '%B %d, %Y',     # "February 02, 2026"
        '%Y-%m-%d',      # "2026-02-02"
        '%d/%m/%Y',      # "02/02/2026"
        '%d.%m.%Y',      # "02.02.2026"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None


def scrape_article_content(article_url):
    """
    Scrape the full content of an individual article.
    Returns a dict with title, date, and content.
    """
    try:
        response = requests.get(article_url, headers=HEADERS, timeout=15)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # --- Extract title ---
        title_elem = soup.find('h1')
        title = title_elem.get_text().strip() if title_elem else None
        
        # --- Extract date ---
        date_str = None
        date_selectors = [
            'time',
            '.ecl-date-block',
            '.date',
            'meta[property="article:published_time"]',
            '.field--name-created',
        ]
        
        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                date_str = date_elem.get('datetime') or date_elem.get('content') or date_elem.get_text()
                if date_str:
                    break
        
        # Parse the date
        article_date = None
        if date_str:
            # Handle ISO format (2026-02-02T10:30:00+00:00)
            if 'T' in str(date_str):
                try:
                    date_only = date_str.split('T')[0]
                    article_date = datetime.fromisoformat(date_only)
                except:
                    pass
            if not article_date:
                article_date = parse_date(date_str)
        
        # --- Extract content ---
        content = ''
        content_selectors = [
            '.ecl-paragraph p',
            '.field--name-body p',
            'article p',
            '.content p',
            '.main-content p',
        ]
        
        for selector in content_selectors:
            content_elems = soup.select(selector)
            if content_elems:
                content = ' '.join([elem.get_text().strip() for elem in content_elems])
                if len(content) > 100:
                    break
        
        # Clean content (normalize whitespace)
        content = re.sub(r'\s+', ' ', content).strip()
        
        return {
            'title': title,
            'date': article_date,
            'content': content[:5000] if content else ''
        }
        
    except Exception as e:
        print(f"     ❌ Error scraping article: {e}")
        return None


def scrape_news_page(page_num):
    """
    Scrape a single page of the news listing.
    Returns a list of article link dicts and a success boolean.
    """
    page_url = get_page_url(page_num)
    print(f"\n📄 Page {page_num}: {page_url}")
    
    try:
        response = requests.get(page_url, headers=HEADERS, timeout=15)
        if response.status_code != 200:
            print(f"   ❌ HTTP {response.status_code}")
            return [], False
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all news article links
        article_links = []
        all_links = soup.find_all('a', href=True)
        
        for link in all_links:
            href = link.get('href', '')
            
            # Filter for actual news articles (contain /news/ in URL)
            if '/news/' not in href:
                continue
            
            # Skip navigation/category links
            if 'news_en' in href or href.endswith('/news/'):
                continue
            
            # Build full URL
            if href.startswith('/'):
                full_url = BASE_URL + href
            elif href.startswith('http'):
                full_url = href
            else:
                continue
            
            # Get link text as potential title preview
            link_text = link.get_text().strip()
            
            # Only include if it looks like an article (has a real title)
            if len(link_text) > 20:
                article_links.append({
                    'url': full_url,
                    'preview_title': link_text[:100]
                })
        
        # Remove duplicates (keep first occurrence)
        seen_urls = set()
        unique_links = []
        for link in article_links:
            if link['url'] not in seen_urls:
                seen_urls.add(link['url'])
                unique_links.append(link)
        
        print(f"   📰 Found {len(unique_links)} article links")
        return unique_links, True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return [], False


def main():
    """
    Main scraping function with pagination.
    
    Strategy:
    1. Start at Page 1
    2. Get all article links from that page
    3. Scrape each article for full content
    4. Check article date - if older than CUTOFF_DATE, stop
    5. Move to Page 2 (calculated as ?page=1), repeat
    """
    print("=" * 60)
    print("🇪🇺 EU COMMISSION NEWS SCRAPER")
    print("=" * 60)
    print(f"Source: {NEWS_PAGE}")
    print(f"Cutoff Date: {CUTOFF_DATE.strftime('%B %d, %Y')}")
    print(f"Max Pages: {MAX_PAGES}")
    print("=" * 60)
    
    all_articles = []
    scraped_urls = set()
    reached_cutoff = False
    oldest_date_seen = datetime.now()
    
    # --- PAGINATION LOOP ---
    for page_num in range(1, MAX_PAGES + 1):
        
        # Step 1: Get article links from this page
        article_links, success = scrape_news_page(page_num)
        
        if not success or not article_links:
            print(f"\n⚠️  No more articles found. Stopping pagination.")
            break
        
        # Step 2: Scrape each article on this page
        page_article_count = 0
        
        for i, link_info in enumerate(article_links):
            article_url = link_info['url']
            
            # Skip if already scraped (from earlier pages)
            if article_url in scraped_urls:
                continue
            
            scraped_urls.add(article_url)
            
            # Rate limiting - be respectful to the server
            time.sleep(0.5)
            
            # Scrape the article
            print(f"   [{i+1}/{len(article_links)}] {link_info['preview_title'][:50]}...")
            
            article_data = scrape_article_content(article_url)
            
            if article_data and article_data.get('title') and article_data.get('content'):
                article_date = article_data.get('date')
                
                # Step 3: Check if article is too old
                if article_date:
                    if article_date < CUTOFF_DATE:
                        print(f"\n   ⏹️  Article from {article_date.strftime('%Y-%m-%d')} is before cutoff.")
                        print(f"   📅 Cutoff date reached! Stopping scraper.")
                        reached_cutoff = True
                        break
                    
                    # Track oldest date we've seen
                    if article_date < oldest_date_seen:
                        oldest_date_seen = article_date
                    
                    date_str = article_date.strftime("%A, %d %B %Y")
                else:
                    # No date found - use today
                    date_str = datetime.now().strftime("%A, %d %B %Y")
                
                # Only add if content is substantial (>200 chars)
                if len(article_data['content']) > 200:
                    all_articles.append({
                        'title': article_data['title'],
                        'date': date_str,
                        'source': 'European Commission',
                        'link': article_url,
                        'content': article_data['content']
                    })
                    page_article_count += 1
        
        print(f"\n   ✅ Scraped {page_article_count} articles from page {page_num}")
        print(f"   📊 Total so far: {len(all_articles)} articles")
        
        # Check if we hit the cutoff
        if reached_cutoff:
            break
        
        # Delay between pages to be nice to the server
        time.sleep(1)
    
    # --- FINAL CLEANUP ---
    
    # Remove duplicates by title (some articles might appear on multiple pages)
    seen_titles = set()
    unique_articles = []
    for article in all_articles:
        if article['title'] not in seen_titles:
            seen_titles.add(article['title'])
            unique_articles.append(article)
    
    # --- SAVE RESULTS ---
    print("\n" + "=" * 60)
    print("📊 SCRAPING COMPLETE")
    print("=" * 60)
    print(f"Total Unique Articles: {len(unique_articles)}")
    
    if unique_articles:
        # Show date range
        dates = [a['date'] for a in unique_articles]
        print(f"Date Range: {min(dates)} to {max(dates)}")
    
    # Save to JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(unique_articles, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved to: {OUTPUT_FILE}")
    print("=" * 60)
    
    return unique_articles


if __name__ == "__main__":
    main()
