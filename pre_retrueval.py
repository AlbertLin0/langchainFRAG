import shutil 
import os 
import re 
import dotenv
import json
import jsonlines
import logging
from langchain_openai import ChatOpenAI
from pathlib import Path

logger = logging.getLogger(__name__)
"""
    该文件用于预处理Query
    TODO: 2025-7-28
    Corpus的处理方式 表格·图片·关键词等等。
"""
def load_prompt(subset, prompt_path=Path("./prompts"), key="queries") -> str:
    """
    加载提示词
    :param subset: 数据集子集名称
    :param prompt_path: 提示词路径
    :param key: 提示词键
    :return: 提示词内容
    """
    if not prompt_path.exists():
        logger.error(f"Prompt file {prompt_path} does not exist.")
        raise FileNotFoundError(f"Prompt file {prompt_path} does not exist.")
    
    with open(prompt_path, 'r', encoding='utf-8') as file:
        prompts = json.load(file)["pre_retrieval"][key]
    if subset not in prompts:
        logger.error(f"Subset {subset} not found in prompts.")
        raise KeyError(f"Subset {subset} not found in prompts.")
    
    return prompts[subset]

def load_jsonl(file_path):
    """
    加载jsonl文件
    :param file_path: jsonl文件路径
    :return: jsonl数据列表
    """
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} does not exist.")
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data

def save_jsonl(data, file_path):
    """
    保存数据为jsonl格式
    :param data: 数据列表
    :param file_path: 保存路径
    """
    # if not data:
    #     raise ValueError("Data is empty, nothing to save.")
    
    with jsonlines.open(file_path, mode='w') as writer:
        for item in data:
            writer.write(item)
    logger.info(f"Data saved to {file_path}")

def expand_queries(subset, dataset_path, llm:ChatOpenAI):
    """
    扩展查询  提取Query中的关键词,作为Query的补充内容
    :param subset: 数据集子集名称
    :param dataset_path: 数据集路径
    :param llm: LLM模型
    """
    # 若文件过大，需要改为流处理方式
    prompt_templete = load_prompt(subset)
    queries_data = load_jsonl(os.path.join(dataset_path, subset, "queries.jsonl"))

    expanded_queries = []
    for query in queries_data:
        # 使用llm模型生成扩展查询
        query_text = query["text"]
        # prompt_templete ... \n\n# Query\n
        prompt = f"{prompt_templete}{query_text}"

        new_query = llm.invoke(prompt).content

        expanded_queries.append({
            "_id": query["id"],
            "title": query["title"],
            "text": f"{new_query}\n\n{query_text}",
        })
    
    # 保存扩展查询
    save_path = Path(f"{dataset_path}/{subset}/expanded_queries.jsonl")
    if not save_path.is_file():
        save_jsonl(expanded_queries, save_path)

def copy_corpus(subset, dataset_path):
    from_path = os.path.join(dataset_path, subset, 'corpus.jsonl')
    to_path = os.path.join(dataset_path, subset, 'corpus_prep.jsonl')
    shutil.copy(from_path, to_path)

def pre_retrieve(dataset_path, llm:ChatOpenAI):
    # load env, 需要使用模型预处理数据集
    # config 文件中与 env文件中有冲突，可以选择将重要信息从config文件中提取出来，放到.env文件中。
    try:
        dotenv.load_dotenv()
    except Exception as e:
        logger.error(f"Error loading .env file: {e}")
    if not os.getenv("Qwen-API-KEY"):
        raise ValueError("Qwen-API-KEY is not set in the environment variables.")

    # 针对不同的数据集有不一样的处理方法
    # 特别是针对Mluti-hop数据集
    # 表格处理
    subsets = [
        # "FinanceBench",
        # "FinDER",
        # "FinQABench",
        # "MultiHiertt",
        # "ConvFinQA",
        # "TATQA",
        "FinQA_corpus",
    ]
    for subset in subsets:
        try: 
            # 处理query扩展
            expand_queries(subset, dataset_path, llm)
            # 处理corpus扩展
            # TODO: 如何处理corpus中的表格数据？ 图片数据？
            copy_corpus(subset, dataset_path)
        except Exception as e:
            logger.error(f"Error processing subset {subset}: {e}")
            continue


    