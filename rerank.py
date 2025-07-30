import requests
import logging
from typing import List

logger = logging.getLogger(__name__)

def get_rerank(query: str,
    documents: List[str],
    api_key: str,
    url: str,
    top_n: int = 1,
    model: str = "models/bge-reranker-large",
    timeout: float = 10.0,
    **kwargs) -> dict:
    
    if api_key or url is None:
        logger.error("API key or URL is not provided.")

    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json",
               "Accept": "application/json"}
    data = {"query": query, "documents": documents,
            "return_documents": True,
            "raw_scores": True,
            "model": model,
            "top_n": top_n}
    try:
        response = requests.post(url, json=data, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.exception("rerank 请求失败: %s", e)
        return None

