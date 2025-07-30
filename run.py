"""
This script is used to run the LangChain application with the specified configuration.
"""

import yaml
import os
import logging
from langchain_openai import ChatOpenAI
from pre_retrueval import pre_retrieve
import rerank 

def load_yaml_config(file_path: str):
    """

    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML file {file_path}: {e}")

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
    "embedding_config": os.path.join(CONFIG_DIR, "embedding.yaml")
}

# Load configurations
model_config = load_yaml_config(config_files["model_config"])
reranker_config = load_yaml_config(config_files["reranker_config"])
embedding_config = load_yaml_config(config_files["embedding_config"])

# Initialize LLM model
# TODO: 只有这一种调用方式吗？
llm = ChatOpenAI(
    api_key=model_config['Qwen-API-key'],
    base_url=model_config['url'],
    model_name=model_config['model'],
)

# Run pre-retrieval process
# 该方法固定了数据集。
pre_retrieve(dataset_path=model_config['dataPath'], llm=llm)

# Run embeddings process
# store embedding results in the milvus database


