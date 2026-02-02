import json
import os
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup, SoupStrainer
import time
import re

# --- CONFIGURATION ---
OUTPUT_FILE = r"D:\NLP_Projects\EU_News_Analyst\eu_news_data.json"
TARGET_TOTAL_ARTICLES = 500

# Dynamic Dates
TODAY = datetime.now()
START_DATE = TODAY - timedelta(days=60)  # EXTENDED from 30 to 60 days for more articles

# --- EU NEWS SOURCES (Hybrid: RSS + Direct Scraping) ---
EU_SOURCES = [
    {
        'name': 'European Commission',
        'type': 'rss',
        'rss_url': 'https://ec.europa.eu/commission/presscorner/api/rss?language=en',
        'base_url': 'https://ec.europa.eu'
    },
    {
        'name': 'European Parliament',
        'type': 'rss',
        'rss_url': 'https://www.europarl.europa.eu/rss/doc/press-releases/en.xml',
        'base_url': 'https://www.europarl.europa.eu'
    },
    {
        'name': 'EU Council',
        'type': 'direct',
        'news_url': 'https://www.consilium.europa.eu/en/press/press-releases/',
        'base_url': 'https://www.consilium.europa.eu',
        'article_selector': '.list-item, .press-release',
        'title_selector': 'h2, .title a',
        'content_selector': '.summary, .description',
        'date_selector': '.date, time',
        'link_selector': 'a[href*="/press/"]'
    },
    {
        'name': 'Euronews EU',
        'type': 'rss',
        'rss_url': 'https://www.euronews.com/rss?level=vertical&name=my-europe',
        'base_url': 'https://www.euronews.com'
    },
    {
        'name': 'Politico EU',
        'type': 'rss',
        'rss_url': 'https://www.politico.eu/feed/',
        'base_url': 'https://www.politico.eu'
    },
    {
        'name': 'EU Observer',
        'type': 'rss',
        'rss_url': 'https://euobserver.com/rss.xml',
        'base_url': 'https://euobserver.com'
    },
    {
        'name': 'DW EU News',
        'type': 'rss',
        'rss_url': 'https://rss.dw.com/xml/rss-en-eu',
        'base_url': 'https://www.dw.com'
    },
    {
        'name': 'Reuters EU',
        'type': 'rss',
        'rss_url': 'https://feeds.reuters.com/Reuters/worldNews',
        'base_url': 'https://www.reuters.com'
    }
]

# --- 1. SCRAPE EU NEWS DIRECTLY FROM HTML ---
def scrape_eu_news():
    """Scrape real EU news directly from official EU websites using BeautifulSoup."""
    articles = []
    print("--- SCRAPING REAL EU NEWS FROM OFFICIAL SOURCES ---")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for source in EU_SOURCES:
        try:
            print(f"Scraping {source['name']}...")

            if source['type'] == 'rss':
                # Use BeautifulSoup to parse RSS XML/HTML directly instead of feedparser
                try:
                    rss_response = requests.get(source['rss_url'], headers=headers, timeout=10)
                    rss_soup = BeautifulSoup(rss_response.content, 'xml')
                except:
                    print(f"Failed to fetch RSS from {source['name']}")
                    continue

                cutoff = time.time() - (60 * 24 * 60 * 60)  # 60 days ago - EXTENDED

                # Parse RSS items using BeautifulSoup
                items = rss_soup.find_all('item')
                if not items:
                    # Try channel > item structure
                    channel = rss_soup.find('channel')
                    if channel:
                        items = channel.find_all('item')

                for item in items[:50]:  # Limit entries per feed - INCREASED from 25
                    try:
                        # Extract title
                        title_elem = item.find('title')
                        title = title_elem.get_text().strip() if title_elem else 'No Title'

                        # Extract link
                        link_elem = item.find('link')
                        article_url = link_elem.get_text().strip() if link_elem else None

                        # Extract publication date
                        pub_date_elem = item.find('pubDate') or item.find('published')
                        dt = None
                        if pub_date_elem:
                            date_text = pub_date_elem.get_text().strip()
                            try:
                                # Try different date formats
                                for fmt in ['%a, %d %b %Y %H:%M:%S %Z', '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%d %H:%M:%S']:
                                    try:
                                        dt = datetime.strptime(date_text, fmt)
                                        break
                                    except:
                                        continue
                            except:
                                dt = TODAY

                        if dt and dt.timestamp() > cutoff:
                            clean_date = dt.strftime("%A, %d %B %Y")

                            # Get initial content from RSS
                            description_elem = item.find('description')
                            content = description_elem.get_text().strip() if description_elem else ''

                            # Scrape full article content
                            try:
                                if article_url:
                                    article_response = requests.get(article_url, headers=headers, timeout=10)
                                    article_soup = BeautifulSoup(article_response.content, 'html.parser')

                                    # Try multiple selectors to get full content
                                    content_selectors = [
                                        '.article-content p', '.content p', 'article p',
                                        '.main-content p', '.story-body p', '.post-content p'
                                    ]

                                    for selector in content_selectors:
                                        content_elems = article_soup.select(selector)
                                        if content_elems:
                                            full_content = ' '.join([elem.get_text().strip() for elem in content_elems[:25]])
                                            if len(full_content) > len(content):
                                                content = full_content
                                            break
                            except:
                                pass  # Keep RSS content if scraping fails

                            # Clean content
                            content = re.sub(r'\s+', ' ', content).strip()

                            if len(content) > 200:  # Substantial content
                                articles.append({
                                    "title": title,
                                    "date": clean_date,
                                    "source": source['name'],
                                    "link": article_url or '',
                                    "content": content[:5000]  # Full content for in-depth answers
                                })
                    except Exception as e:
                        print(f"Error parsing RSS item: {e}")
                        continue

            elif source['type'] == 'direct':
                # Direct HTML scraping for sources without RSS
                response = requests.get(source['news_url'], headers=headers, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')
                article_links = soup.select(source['link_selector'])

                for link in article_links[:15]:  # Limit articles
                    article_url = link.get('href')
                    if not article_url.startswith('http'):
                        article_url = source['base_url'] + article_url

                    try:
                        article_response = requests.get(article_url, headers=headers, timeout=10)
                        article_soup = BeautifulSoup(article_response.content, 'html.parser')

                        # Extract title
                        title_elem = article_soup.select_one(source['title_selector'])
                        title = title_elem.get_text().strip() if title_elem else link.get_text().strip()

                        # Extract date
                        date_elem = article_soup.select_one(source['date_selector'])
                        if date_elem:
                            date_text = date_elem.get_text().strip()
                            try:
                                dt = datetime.strptime(date_text, '%d %B %Y')
                            except:
                                dt = TODAY
                        else:
                            dt = TODAY

                        if dt >= START_DATE:
                            # Extract full content
                            content_elems = article_soup.select(source['content_selector'])
                            content = ' '.join([elem.get_text().strip() for elem in content_elems])

                            if len(content) < 500:
                                # Try additional selectors
                                body_selectors = ['article p', '.content-body p', '.main-content p']
                                for selector in body_selectors:
                                    body_elems = article_soup.select(selector)
                                    if body_elems:
                                        content = ' '.join([elem.get_text().strip() for elem in body_elems[:25]])
                                        break

                            content = re.sub(r'\s+', ' ', content).strip()

                            if len(content) > 200:
                                articles.append({
                                    "title": title,
                                    "date": dt.strftime("%A, %d %B %Y"),
                                    "source": source['name'],
                                    "link": article_url,
                                    "content": content[:5000]
                                })

                    except:
                        continue

                    time.sleep(1)  # Be respectful

        except Exception as e:
            print(f"Error scraping {source['name']}: {e}")
            continue

    # Remove duplicates and filter for substantial content
    seen_titles = set()
    unique_articles = []
    for article in articles:
        if (article['title'] not in seen_titles and 
            len(article['content']) > 300 and 
            len(article['title']) > 15):
            seen_titles.add(article['title'])
            unique_articles.append(article)

    print(f"-> Scraped {len(unique_articles)} unique EU news articles with full content.")
    return unique_articles

# --- MAIN ---
def main():
    # 1. Scrape Real EU News from official sources
    full_dataset = scrape_eu_news()

    # 2. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(full_dataset, f, indent=4)

    print(f"\n[SUCCESS] Total Articles: {len(full_dataset)}")
    print(f"Date Range: {START_DATE.strftime('%d %b %Y')} to {TODAY.strftime('%d %b %Y')}")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()