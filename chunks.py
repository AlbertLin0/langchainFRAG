# 分块策略
# =================
# Operation classes
# =================
# Question: 
# 1. 不同的编码类型（如文本、图像）是否需要不同的分块策略？
# 2. 不同格式的数据（如PDF、HTML）是否需要不同的分块策略？
#    - 专用分割器 - 自适应选择函数？
# 3. 分块的大小和重叠度如何影响模型的性能和准确性？ 
#    - https://zilliz.com.cn/learn/langchain-chunking-milvus
# 4. 如何处理分块后的数据，以确保信息的完整性和一致性？
# 5. 分块策略是否需要根据数据的特性进行动态调整？
# 6. 如何评估分块策略的效果，以便进行优化和改进？
import logging

logger = logging.getLogger(__name__)

__OPERATOR__ = {}

def register_operator(name:str):
    def wrapper(cls):
        if name in __OPERATOR__:
            logger.error(f"Operator {name} is already registered.")
            raise ValueError(f"Operator {name} is already registered.")
        __OPERATOR__[name] = cls
        return cls
    return wrapper

def get_operator(name:str):
    if name not in __OPERATOR__:
        logger.error(f"Operator {name} is not registered.")
        raise ValueError(f"Operator {name} is not registered.")
    return __OPERATOR__[name]

# 抽象类需要实现什么功能？
class BaseOperator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # 可能需要初始化一些参数或配置
    def execute(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")
    # TODO: model选择
    def _tokenize_text(self, text: str, model="Qwen/Qwen3-30B-A3B"):
        """使用指定的模型对文本进行分词"""
        # 使用Hugging Face的transformers库
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model)
        return tokenizer.tokenize(text)

# ================
# LangChain提供的分块策略
# ================
from langchain.text_splitter import CharacterTextSplitter
@register_operator("langchain-length-based")
class langchain_length_based_splitter(BaseOperator):
    def __init__(self, separator="", chunk_size=512, chunk_overlap=51, is_tokenized=False, **kwargs):
        self.splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )
        self.is_tokenized = is_tokenized
    def execute(self, text: str):
        # !!!必须确保text是字符串类型、编码方式是UTF-8
        if self.is_tokenized:
            # 如果已经是分词后的文本，直接返回
            text = super()._tokenize_text(text)
            print(text)
        return self.splitter.split_text(text)    

from langchain.text_splitter import RecursiveCharacterTextSplitter
@register_operator("langchain-structure-based")
class langchain_structure_based_splitter(BaseOperator):
    def __init__(self, chunk_size=512, chunk_overlap=51, is_tokenized=False, **kwargs):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )
        self.is_tokenized = is_tokenized
    def execute(self, text: str):
        if self.is_tokenized:
            text = super()._tokenize_text(text)
        return self.splitter.split_text(text)

@register_operator("resursiveChararterTextSplitter")
class RecursiveChararterTextSplitter(BaseOperator):
    pass

if __name__ == "__main__":
    # 测试 langchain_length_based_splitter
    splitter = get_operator("langchain-length-based")(chunk_size=10, chunk_overlap=2, is_tokenized=True)
    text = "This is a test text for the langchain length based splitter."
    chunks = splitter.execute(text)
    print(chunks)  # 输出分块结果
