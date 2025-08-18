import dataclasses
from typing import Union, List, Dict
from googlesearch import search

def search_for_news(query: str) -> List[Dict[str, str]]:
    """
    Performs a web search for news articles and retrieves headlines and URLs.

    This function uses googlesearch-python to find recent news articles and then
    formats the results into a list of dictionaries.

    Args:
        query: The search query for news articles (e.g., "new tech conference").

    Returns:
        A list of dictionaries, where each dictionary contains a 'headline'
        and a 'url' for a news article.
    """
    print(f"Searching for news related to: '{query}'...")

    try:
        # Perform the search using googlesearch-python
        # We'll limit to 10 results and add 'news' to the query to focus on news articles
        search_query = f"{query} news"
        search_results = search(search_query, num_results=10)

        headlines = []
        # Process the results
        for url in search_results:
            # For now we'll just use the URL as both headline and URL
            # since googlesearch-python only returns URLs
            headlines.append({
                "headline": url,
                "url": url
            })
        
        print("Search completed and results processed.")
        return headlines

    except Exception as e:
        print(f"An error occurred during the search: {e}")
        return []

# --- Example Usage ---
if __name__ == "__main__":
    search_query = "tesla "
    news_headlines = search_for_news(search_query)

    print("\n--- NEWS HEADLINES ---")
    if news_headlines:
        for item in news_headlines:
            print(f"URL: {item['url']}")
            print("-" * 20)
    else:
        print("No news headlines retrieved.")

def get_soup_from_url(url: str) -> Union[BeautifulSoup, None]:
    """
    Grabs the HTML content from a given URL and returns a BeautifulSoup object.

    This function sends an HTTP GET request to the URL, checks for a successful
    response, and then parses the content using BeautifulSoup.

    Args:
        url: The URL of the webpage to scrape.

    Returns:
        A BeautifulSoup object of the webpage's content, or None if the request fails.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url, timeout=10)
        # Raise an HTTPError if the response status code is not 200
        response.raise_for_status()

        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')
        print(f"Successfully created BeautifulSoup object for {url}")
        return soup

    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from {url}: {e}")
        return None