import requests
import logging
"""
    英语文本用 zh-embeddings ......
"""

logger = logging.getLogger(__name__)

def get_embedding(
    text,
    url,
    api_key,
    model="models/bge-large-zh-v1.5",
    timeout=10.0,
    **kwargs) -> dict:

    if api_key or url is None:
        logger.error("API key or URL is not provided.")

    headers = {"Accept": "application/json",
               "Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json",
               }
    data = {"model": model,
            "encoding_format": "float",
            "user": "string",
            "dimensions": 1024,
            "input": text}
    try:
        response = requests.post(url, json=data, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.exception("Embedding request failed: %s", e)
        return None
