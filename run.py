"""
This script is used to run the LangChain application with the specified configuration.
"""

import yaml
import os
import logging
import rerank 
import dotenv
from langchain_openai import ChatOpenAI
from pre_retrueval import pre_retrieve
from utils import load_yaml_config
# 日志文件路径（可选）
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "RAG.log")

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 最低记录级别
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",  # 日志格式
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),  # 输出到文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

logger = logging.getLogger(__name__)


# Define paths to config files
CONFIG_DIR = "config"
config_files = {
    "model_config": os.path.join(CONFIG_DIR, "model.yaml"),
    "reranker_config": os.path.join(CONFIG_DIR, "reranker.yaml"),
    "embedding_config": os.path.join(CONFIG_DIR, "embedding.yaml"),
    "milvus_config": os.path.join(CONFIG_DIR, "milvus.yaml")
}

# Load configurations
model_config = load_yaml_config(config_files["model_config"])
reranker_config = load_yaml_config(config_files["reranker_config"])
embedding_config = load_yaml_config(config_files["embedding_config"])
milvus_config = load_yaml_config(config_files["milvus_config"])

# Initialize LLM model
# TODO: 只有这一种调用方式吗？
try:
    dotenv.load_dotenv()
except Exception as e:
    logger.error(f"Error loading .env file: {e}")
if not os.getenv("Qwen-API-KEY"):
    raise ValueError("Qwen-API-KEY is not set in the environment variables.")
# config or Env Value？
llm = ChatOpenAI(
    api_key=os.getenv("Qwen-API-KEY", model_config['Qwen-API-key']),
    base_url=os.getenv("Qwen-API-URL", model_config['url']),
    model_name=os.getenv("Qwen-API-MODEL", model_config['model']),
)

# Run pre-retrieval process
# 该方法指定了数据集。
# v1 该方法只处理了查询扩展, expand_query_stream支持流式处理query
pre_retrieve(dataset_path=model_config['dataPath'], llm=llm)

# Run embeddings process
# store embedding results in the milvus database



