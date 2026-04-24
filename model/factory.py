from abc import ABC, abstractmethod
from typing import Optional
from langchain.embeddings import Embeddings
from langchain_community.document_compressors import DashScopeRerank
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_config


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[object]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> ChatTongyi:
        kwargs = {"model": rag_config["chat_model_name"]}
        api_key = rag_config.get("dashscope_api_key")
        if api_key:
            kwargs["api_key"] = api_key
        return ChatTongyi(**kwargs)


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> DashScopeEmbeddings:
        return DashScopeEmbeddings(model=rag_config["embedding_model_name"])


class RerankModelFactory(BaseModelFactory):
    def generator(self) -> DashScopeRerank:
        rerank_cfg = rag_config.get("rerank", {})
        model_name = str(rerank_cfg.get("model_name", ""))
        top_n = int(rerank_cfg.get("top_n", 5))
        api_key = rag_config.get("dashscope_api_key")
        kwargs = {"model": model_name, "top_n": top_n}
        if api_key:
            kwargs["dashscope_api_key"] = api_key
        return DashScopeRerank(**kwargs)


chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingsFactory().generator()
rerank_model = RerankModelFactory().generator()
