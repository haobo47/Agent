from langchain_core.documents import Document

from model.factory import rerank_model
from rag.hybrid_retriever import HybridRetriever
from utils.config_handler import rag_config
from utils.logger_handler import logger


class RerankService:
    def __init__(self):
        rerank_cfg = rag_config.get("rerank", {})
        self.top_n = int(rerank_cfg.get("top_n", 5))
        self.hybrid_retriever = HybridRetriever()
        self.rerank_model = rerank_model

    def rerank(self, query: str) -> list[Document]:
        docs = self.hybrid_retriever.retrieve(query)

        if not docs:
            return []

        if self.rerank_model is None:
            raise RuntimeError("[RerankService]未初始化重排模型")

        try:
            reranked_docs = self.rerank_model.compress_documents(
                documents=docs,
                query=query,
            )
        except Exception as exc:
            logger.error(f"[RerankService]重排模型调用失败: {exc}", exc_info=True)
            raise

        if not reranked_docs:
            msg = "[RerankService]重排返回结果为空"
            logger.error(msg)
            raise RuntimeError(msg)

        reranked_docs = list(reranked_docs)
        for doc in reranked_docs:
            metadata = dict(doc.metadata or {})
            if "relevance_score" in metadata:
                metadata["rank_score"] = float(metadata["relevance_score"])
            doc.metadata = metadata

        logger.info(
            f"[RerankService]候选文档数={len(docs)}, 重排后文档数={len(reranked_docs)}"
        )

        return reranked_docs


if __name__ == "__main__":
    query = "机器人开机没反应怎么回事？"

    service = RerankService()
    try:
        reranked = service.rerank(query)
        print(f"query: {query}")
        print(f"重排后数量: {len(reranked)}")
        for index, doc in enumerate(reranked, start=1):
            metadata = doc.metadata or {}
            print(f"结果{index}:")
            print(f"source: {metadata.get('source', '')}")
            rank_score = metadata.get("rank_score", "")
            print(f"rank_score: {rank_score}")
            print(doc.page_content)
    except Exception as exc:
        print(f"重排测试失败: {exc}")
