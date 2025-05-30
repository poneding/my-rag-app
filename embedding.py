import os

from dotenv import find_dotenv, load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_zhipuai_dev.embedding import ZhipuAIEmbeddings
from pydantic import SecretStr

_ = load_dotenv(find_dotenv())  # 加载 .env 文件中的环境变量


def get_embeddings():
    """获取 OpenAI 的 Embeddings 实例"""

    # OpenAI
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    # return OpenAIEmbeddings(
    #     api_key=SecretStr(openai_api_key) if openai_api_key else None
    # )

    # Gemini
    # google_api_key = os.getenv("GOOGLE_API_KEY")
    # return GoogleGenerativeAIEmbeddings(
    #     google_api_key=SecretStr(google_api_key) if google_api_key else None,
    #     model="models/gemini-embedding-exp-03-07",  # models/text-embedding-004
    # )

    # 智谱 AI
    zhipuai_api_key = os.getenv("ZHIPUAI_API_KEY")
    return ZhipuAIEmbeddings(
        api_key=SecretStr(zhipuai_api_key) if zhipuai_api_key else None,
        model="embedding-3",
    )


def document_path():
    """获取文档路径"""
    return "documents"


def vector_db_path():
    """获取向量数据库路径"""
    return "vector_db/chroma"


# 加载 documents 文件夹下所有文档作为知识库
file_paths = []
for root, dirs, files in os.walk(document_path()):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
print(file_paths[:3])


loaders = []

for file_path in file_paths:
    file_type = file_path.split(".")[-1]
    # 支持 pdf 和 md 文件
    if file_type == "pdf":
        loaders.append(PyMuPDFLoader(file_path))
    elif file_type == "md":
        loaders.append(UnstructuredMarkdownLoader(file_path))

texts = []
for loader in loaders:
    texts.extend(loader.load())

# 根据字符长度对文档进行切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(texts)


# 将文档转换为向量存储到本地持久化的 Chroma 数据库中
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=get_embeddings(),
    persist_directory=vector_db_path(),
)
