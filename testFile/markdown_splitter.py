from typing import Union
from langchain_core.documents import Document
import logging
from ..chunks import BaseOperator 
import re
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

@register_operator("markdown-splitter")
class MarkdownSplitter(BaseOperator):
    """
    🚀🚀🚀 若需要更复杂的逻辑,请使用抽象语法树
    ===== 逻辑 =====
    1. 基本逻辑
        - #符号分块
        - \n 符号分块
        -- 处理无用字符串 如\t
    2. 特殊逻辑 - 需要后处理
        - 行间公式
        - 行间代码块
        - 行内内容公式等等

    ===== 优先级表 =====
    | 优先级  | 规则描述           | 备注                        |
    | ------ | ----------------- | -------------------------- |
    | 1.2    | 井号分块           | 简单粗暴                     |
    | 1      | 行间公式、代码      | 行间公式内容也受到字符串全局匹配 |
    | 2      | 换行符分块、        | 简单                        |
    | 1.1    | 行内内容           | 行内内容                    |
    | 99     | 处理无用字符串      | 其他内容                    |

    ===== 注意事项 =====
    1. #符号分到哪个字标题？若子模块过小，还需要分块吗
        - 解决方法：设定分块大小参考值，除了标题和副标题外，增加合并逻辑
    2. 子模块过小,可以使用recursiveCharacterTextSplitter进行二次分块

    ===== 复杂度 =====
    - 时间复杂度,O(n),n为文本长度
    langchain的实现方式复杂度为O(xn), x为Header数量
    """
    DEFAULT_HEADER_KEYS = {
        "#": "Header 1",
        "##": "Header 2",
        "###": "Header 3",
        "####": "Header 4",
        "#####": "Header 5",
        "######": "Header 6",
    }

    def __init__(self, 
                 headers_to_split_on: Union[dict[str, str], None] = None,
                 strip_headers: bool = True,  # noqa: FBT001,FBT002
                 **kwargs):
        if headers_to_split_on is None:
            self.headers_to_split_on = self.DEFAULT_HEADER_KEYS
        else:
            self.headers_to_split_on = headers_to_split_on
        
        self.strip_headers = strip_headers
        self.kwargs = kwargs
        self.chunks : list[Document] = []  # 存储分块后的文本
        self.current_chunk = Document(page_content="")  # 当前分块内容
        self.current_header_stack: list[tuple[int, str]] = []  # 当前标题栈
        
    def execute(self, text: str) -> list[Document]:
        """
        执行分块操作
        :param text: 输入的文本
        :return: 分块后的文本列表
        """
        # ===== 优先级 1 =====
        # 行间公式、代码
        text = self._split_by_math_and_code(text)
        # ===== 优先级 1.1 =====
        # 行内内容 · preliminary: 认为行间内容已经被解决了、不会被再次匹配
        # 使用该方法，误报概率远大于行间公式、代码
        text = self._split_by_inline_content(text)
        # ===== 优先级 1.2 =====
        # 井号分块
        text = self._split_by_headers(text)

        return text

    def _split_by_headers(self, text: str):
        pass

    def _split_by_inline_content(self, text: str):
        """
        处理行内内容, 也会出现误报
        """
        # 1. 使用正则表达式匹配行内公式
        inline_math_pattern = r"\$.*?\$"
        text = re.sub(inline_math_pattern, self._post_process_inline_math, text)

        # 2. 使用正则表达式匹配行内代码块
        inline_code_pattern = r"`.*?`"
        text = re.sub(inline_code_pattern, self._post_process_inline_code, text)

        return text
    
    def _split_by_math_and_code(self, text: str):
        """
          该方法可能存在误报情况,比如在字符串中包含过滤规则
          可以增加匹配规则，减少误报, 比如\n符号
        """
        # 1. 使用正则表达式匹配行间公式
        math_pattern1 = r"```math \n.*?```"
        math_pattern2 = r"\$\$.*?\$\$"
        # TODO: 测试
        # re.DOTALL, 跨行匹配
        text = re.sub(math_pattern2, self._post_process_math, text, flags=re.DOTALL)
        text = re.sub(math_pattern1, self._post_process_math, text, flags=re.DOTALL)

        # 默认r"```math.*?```"其中的内容已被处理，替换为了xxx
        # 2. 使用正则表达式匹配行间代码块
        langs = ["python", "javascript", "java", "c++", "c#", "go", "ruby", "php", "typescript", "yaml"]
        tmp = "|".join(langs)
        code_pattern1 = rf"```({tmp})\n.*?```"
        text = re.sub(code_pattern1, self._post_process_code, text, flags=re.DOTALL)

        return text

    def _post_process_math(self, match: re.Match):
        """
        处理行间公式
        """
        matched = match.group(0)
        matched = "www.math.com"  # TODO: 处理行间公式
        return matched

    def _post_process_code(self, match: re.Match):
        """
        处理行间代码块
        """
        matched = match.group(0)
        matched = "www.code.com"  # TODO: 处理行间代码块
        return matched

    def _post_process_inline_math(self, match: re.Match):
        """
        处理行内公式
        """
        matched = match.group(0)
        matched = "www.inline-math.com"  # TODO: 处理行内公式
        return matched
    
    def _post_process_inline_code(self, match: re.Match):
        """
        处理行内代码块
        """
        matched = match.group(0)
        matched = "www.inline-code.com"  # TODO: 处理行内代码块
        return matched

if __name__ == "__main__":
    # 测试 langchain_markdown_splitter
    splitter = get_operator("markdown-splitter")()
    with open("README.md", "r", encoding="utf-8") as f:
        text = f.read()
    chunks = splitter.execute(text)
    print(chunks)  # 输出分块结果