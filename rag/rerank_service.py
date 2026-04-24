import time
from http import HTTPStatus

import dashscope
from langchain_core.documents import Document

from utils.config_handler import rag_config
from utils.logger_handler import logger


class RerankService:
    def __init__(self):
        rerank_cfg = rag_config.get("rerank", {})
        self.enabled = bool(rerank_cfg.get("enabled", False))
        self.model_name = str(rerank_cfg.get("model_name", ""))
        self.top_n = int(rerank_cfg.get("top_n", 5))
        self.batch_size = int(rerank_cfg.get("batch_size", 12))
        self.strict_mode = bool(rerank_cfg.get("strict_mode", True))

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return []

        if not self.enabled:
            return docs[: self.top_n]

        candidates = docs[: self.batch_size]
        documents = [doc.page_content for doc in candidates]
        started = time.perf_counter()

        try:
            response = dashscope.TextReRank.call(
                model=self.model_name,
                query=query,
                documents=documents,
                top_n=min(self.top_n, len(documents)),
                return_documents=False,
            )
        except Exception as exc:
            logger.error(f"[RerankService]重排模型调用失败: {exc}", exc_info=True)
            if self.strict_mode:
                raise
            return candidates[: self.top_n]

        status_code = getattr(response, "status_code", None)
        if status_code != HTTPStatus.OK:
            msg = f"[RerankService]重排失败, 状态码={status_code}, 消息={getattr(response, 'message', '')}"
            logger.error(msg)
            if self.strict_mode:
                raise RuntimeError(msg)
            return candidates[: self.top_n]

        output = getattr(response, "output", None)
        results = getattr(output, "results", None)
        if not results:
            msg = "[RerankService]重排返回结果为空"
            logger.error(msg)
            if self.strict_mode:
                raise RuntimeError(msg)
            return candidates[: self.top_n]

        reranked_docs: list[Document] = []
        for item in results:
            index = getattr(item, "index", None)
            score = float(getattr(item, "relevance_score", 0.0))
            if index is None or index >= len(candidates):
                continue
            source_doc = candidates[index]
            metadata = dict(source_doc.metadata or {})
            metadata["rank_score"] = score
            reranked_docs.append(
                Document(page_content=source_doc.page_content, metadata=metadata))

        logger.info(
            f"[RerankService]候选文档数={len(candidates)}, 重排后文档数={len(reranked_docs)}, 耗时={(time.perf_counter() - started) * 1000:.2f}"
        )

        if not reranked_docs:
            msg = "[RerankService]重排未能映射任何结果索引"
            logger.error(msg)
            if self.strict_mode:
                raise RuntimeError(msg)
            return candidates[: self.top_n]

        return reranked_docs
