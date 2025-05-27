from typing import List
from langchain_core.embeddings import Embeddings
from zhipuai import ZhipuAI


class ZhipuAIEmbedding(Embeddings):
    def __init__(self, api_key: str):
        self.client = ZhipuAI(api_key=api_key)

    def embed_documents(self, texts: List[str]):
        result = []
        for i in range(0, len(texts), 64):
            embeddings = self.client.embeddings.create(
                model="embedding-3", input=texts[i : i + 64]
            )
            result.extend([embeddings.embedding for embeddings in embeddings.data])
        return result

    def embed_query(self, text: str):
        return self.embed_documents([text])[0]
