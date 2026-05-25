
import json
from tavily import TavilyClient

client = TavilyClient(api_key="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")

response = client.search(
    query="news on top energy stocks to buy",
    search_depth="advanced",
    max_results=1
)

# Get first search result
first_result = response["results"][0]

clean_result = {
    "query": response.get("query"),
    "title": first_result.get("title"),
    "url": first_result.get("url"),
    "content": first_result.get("content"),
    "score": first_result.get("score"),
    "response_time": response.get("response_time"),
    "request_id": response.get("request_id")
}

print(json.dumps(clean_result, indent=2))