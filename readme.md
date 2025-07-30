# FinanceRAG
数据集来自Kaggle比赛[ACM-ICAIF '24 FinanceRAG Challenge](https://www.kaggle.com/competitions/icaif-24-finance-rag-challenge/data)

具体内容细节请看比赛描述。

该库主要包含以下内容：
- 语料库、查询的预处理：
  1. 构建查询模版，使用Qwen提取查询的关键词，作为内容补充。
  2. 结构化处理语料库内容，包括对表格的处理。TODO：获取图结构的Summery知识？

- 关联Milvus数据库：
  1. 向量数据库的基本操作
  2. 检索、embedding方法、文本分割算法
  3. TODO：如何适配不同格式的语料？
   
- 过长文本处理方式：TODO：关联更多的语料库知识受到模型上下文长度的限制，如何简化语料库知识、或者如何分批输入到模型，然后汇总信息？

## Dataset
- FinDER: Jargon and abbreviation handling in 10-K reports.
- FinQABench: Detecting hallucinations, ensuring factuality in 10-K reports.
- FinanceBench: Real-world financial queries from 10-K reports.
- TATQA: Numerical reasoning with mixed text and tables.
- FinQA: Multi-step reasoning with earnings reports (text + tables).
- ConvFinQA: Conversational queries on earnings reports.
- MultiHiertt: Complex reasoning across hierarchical tables in annual reports.

## Requierment
xxx
## Chunks
分块策略包含对langchain.text_splite的包装和一部分自定义算子实现。

**Langchain-Length-based**: 该方法很直接、固定分块大小。

优点:
1. Straightforward implementation
2. Consistent chunk sizes
3. Easily adaptable to different model requirements

类型:
1. Token-based
2. Character-based

