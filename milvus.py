from pymilvus import connections, Collection, utility
import logging
# TODO: 代码重构

logger = logging.getLogger(__name__)

# 连接到 Milvus 数据库
connections.connect(
    alias="test",
    host="10.16.41.190",
    port="19530",
    user="root",
    password="Ydkj.nlp@2024",
    secure=False
)
# 检查连接是否成功
if connections.has_connection("test"):
    logger.info("Connected to Milvus successfully.")
else:
    logger.error("Failed to connect to Milvus.")
    raise ConnectionError("Failed to connect to Milvus.")
# 创建知识库
#####
# 因为已经在前端创建了知识库，xxx
#####

# 管理知识库
# 增删查改
