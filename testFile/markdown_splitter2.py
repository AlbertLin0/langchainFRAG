from typing import Union, List
from langchain_core.documents import Document
import logging
import re
import os
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


@register_operator("markdown-splitter")
class MarkdownSplitter(BaseOperator):
    """
    🚀🚀🚀 若需要更复杂的逻辑,请使用抽象语法树
    ===== 逻辑 =====
    1. 基本逻辑
        - #符号分块
        - \n 符号分块
        - 分割线分块
        - 处理无用字符串 如\t
    2. 特殊逻辑 - 需要后处理
        - 行间公式
        - 行间代码块
        - 行内内容公式等等

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
        self.chunks = []  # 清空之前的分块结果
        self.current_chunk = Document(page_content="")  # 重置当前分块内容
        self.current_header_stack = []  # 重置当前标题栈

        raw_lines = text.splitlines(keepends=True)
        while raw_lines:
            raw_line = raw_lines.pop(0)
            # ==== 优先级  =====
            # 这些match不添加在chunk中,header可能除外
            header_match = self._split_by_headers(raw_line)
            math_match = self._split_by_math(raw_line)
            code_match = self._split_by_code(raw_line)
            horz_match = self._split_by_horz(raw_line)
  
            if header_match:
                self._complete_current_chunk()
                depth = len(header_match.group(1))
                header_text = header_match.group(2)
                if not self.strip_headers:
                    # 如果不需要去除标题，则将标题添加到当前分块
                    self.current_chunk.page_content += f"{header_text}\n"
                
                self._resolve_header_stack(depth, header_text)
            elif math_match:
                # 行间公式单独分块
                self._complete_current_chunk()
                self.current_chunk.page_content = self._aggregate_math_line(
                    self.current_chunk.page_content, raw_lines)
                self.current_chunk.metadata = {"type": "math"}
                self._complete_current_chunk()
            elif code_match:
                # 行间代码块单独分块
                self._complete_current_chunk()
                self.current_chunk.page_content = self._aggregate_code_line(
                    self.current_chunk.page_content, raw_lines)
                self.current_chunk.metadata = {"type": "code"}
                self._complete_current_chunk()
            elif horz_match:
                # 分割线直接分块吗？
                self._complete_current_chunk()
            else:
                self.current_chunk.page_content += raw_line
        self._complete_current_chunk()
        return self.chunks
    
    def _resolve_header_stack(self, depth: int, header_text: str):
        """
        处理标题栈，更新当前标题栈
        :param depth: 标题深度
        :param header_text: 标题文本
        """
        # 如果当前标题栈不为空且深度小于等于栈顶元素的深度，则弹出栈顶元素
        while self.current_header_stack and self.current_header_stack[-1][0] >= depth:
            self.current_header_stack.pop()
        
        # 添加新的标题到当前标题栈
        self.current_header_stack.append((depth, header_text))
    
    def _complete_current_chunk(self):
        """
        完成当前分块，将其添加到分块列表中
        添加metadata信息
        """
        chunk = self.current_chunk
        if chunk.page_content and not chunk.page_content.isspace():
            for depth, header in self.current_header_stack:
                key = self.headers_to_split_on.get("#" * depth)
                if key:
                    chunk.metadata[key] = header
            self.chunks.append(chunk)
        # reset current chunk
        # current_chunk 只涉及page_content
        self.current_chunk = Document(page_content="")

    def _aggregate_code_line(self, current_chunk: str, raw_lines: List[str]):
        chunk = current_chunk 
        while raw_lines:
            raw_line = raw_lines.pop(0)
            if self._split_by_code(raw_line):
                # 行间代码块
                return chunk
            chunk += raw_line

        logger.warning("No match code found in the remaining lines.")
        return ""

    def _aggregate_math_line(self, current_chunk: str, raw_lines: List[str]):
        chunk = current_chunk 
        while raw_lines:
            raw_line = raw_lines.pop(0)
            if self._split_by_math(raw_line, is_tail=True):
                # 行间公式
                return chunk
            chunk += raw_line

        logger.warning("No match math found in the remaining lines.")
        return ""
    
    def _split_by_headers(self, line: str):
        match = re.match(r"^(#{1,6}) (.*)", line)
        if match and match.group(1) in self.headers_to_split_on:
            return match 
        return None

    def _split_by_code(self, line: str):
        match = [re.match(rule, line) for rule in [r"^```(.*)", r"^~~~(.*)"]]
        return next((match for match in match if match), None)

    def _split_by_horz(self, line: str):
        match = [re.match(rule, line) for rule in [r"^\*\*\*+\n", r"^---+\n", r"^___+\n"]]
        return next((match for match in match if match), None)

    def _split_by_math(self, line: str, is_tail: bool = False):
        if is_tail:
            match = [re.match(rule, line) for rule in [r"^\$\$(.*)", r"^```(.*)"]]
        else:
            match = [re.match(rule, line) for rule in [r"^\$\$(.*)", r"^```math(.*)"]]
        return next((match for match in match if match), None)

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
    print(os.getcwd())
    with open("readme.md", "r", encoding="utf-8") as f:
        text = f.read()
    chunks = splitter.execute(text)
    print(chunks)  # 输出分块结果


    