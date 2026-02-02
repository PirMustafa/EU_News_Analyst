import json
import os
import requests
from datetime import datetime, timedelta

# --- CONFIGURATION ---
OUTPUT_FILE = r"D:\NLP_Projects\EU_News_Analyst\eu_news_data.json"  # Updated path
NEWS_API_KEY = "YOUR_NEWSAPI_KEY_HERE"  # Get from https://newsapi.org/
TARGET_TOTAL_ARTICLES = 400

# Dynamic Dates
TODAY = datetime.now()
START_DATE = TODAY - timedelta(days=30)

# --- FETCH REAL NEWS USING NEWSAPI ---
def fetch_real_news():
    """Fetches real news articles from the past 30 days using NewsAPI."""
    articles = []
    print("--- FETCHING REAL NEWS (NewsAPI) ---")
    
    # NewsAPI endpoint
    url = "https://newsapi.org/v2/everything"
    
    # Parameters for EU-related news
    params = {
        'q': 'EU OR European Union OR Europe politics OR Europe economy',
        'from': START_DATE.strftime('%Y-%m-%d'),
        'to': TODAY.strftime('%Y-%m-%d'),
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 100,  # Max per request
        'apiKey': NEWS_API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get('status') == 'ok':
            for article in data['articles']:
                # Format date
                published_at = article.get('publishedAt', '')
                if published_at:
                    dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    clean_date = dt.strftime("%A, %d %B %Y")
                else:
                    clean_date = TODAY.strftime("%A, %d %B %Y")
                
                articles.append({
                    "title": article.get('title', ''),
                    "date": clean_date,
                    "source": article.get('source', {}).get('name', 'NewsAPI'),
                    "link": article.get('url', ''),
                    "content": article.get('description', '') or article.get('content', '')[:1000]
                })
            
            print(f"-> Fetched {len(articles)} real articles from NewsAPI.")
        else:
            print(f"NewsAPI Error: {data.get('message', 'Unknown error')}")
    
    except Exception as e:
        print(f"Error fetching from NewsAPI: {e}")
    
    return articles

# --- MAIN ---
def main():
    # Fetch real news for the past 30 days
    news_data = fetch_real_news()
    
    # If we didn't get enough articles, we could add more sources or adjust query
    if len(news_data) < TARGET_TOTAL_ARTICLES:
        print(f"Warning: Only fetched {len(news_data)} articles. Consider adjusting query or using multiple sources.")
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(news_data, f, indent=4)
        
    print(f"\n[SUCCESS] Total Articles: {len(news_data)}")
    print(f"Date Range: {START_DATE.strftime('%d %b %Y')} to {TODAY.strftime('%d %b %Y')}")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()