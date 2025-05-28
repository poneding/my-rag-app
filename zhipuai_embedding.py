from typing import List, Optional
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr
from zhipuai import ZhipuAI


class ZhipuAIEmbedding(Embeddings):
    def __init__(self, api_key: Optional[SecretStr], model: str = "embedding-3"):
        self.model = model
        self.client = ZhipuAI(api_key=api_key.get_secret_value() if api_key else None)

    def embed_documents(self, texts: List[str]):
        result = []
        for i in range(0, len(texts), 64):
            embeddings = self.client.embeddings.create(
                model=self.model, input=texts[i : i + 64]
            )
            result.extend([embeddings.embedding for embeddings in embeddings.data])
        return result

    def embed_query(self, text: str):
        return self.embed_documents([text])[0]
