import hashlib
from threading import Lock
from typing import Any

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer


from rag.vector_store import VectorStoreService
from utils.config_handler import rag_config
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class HybridRetriever:
    def __init__(self, vector_store: VectorStoreService):
        self.vector_store = vector_store
        retrieval_cfg = rag_config["retrieval"]

        self.vector_top_k = int(retrieval_cfg["vector_top_k"])
        self.bm25_top_k = int(retrieval_cfg["bm25_top_k"])
        self.fusion_top_k = int(retrieval_cfg["fusion_top_k"])
        self.bm25_tokenizer = str(retrieval_cfg["bm25_tokenizer"]).lower()
        self.bert_model_name = str(retrieval_cfg["bert_tokenizer_model"])
        self.bm25_min_token_len = int(retrieval_cfg["bm25_min_token_len"])
        self.bm25_stopwords_path = str(retrieval_cfg["bm25_stopwords_path"])

        self.bm25: Any | None = None
        self.bm25_documents: list[Document] = []
        self.bm25_lock = Lock()
        self._bert_tokenizer: Any | None = None
        self._stopwords_cache: set[str] | None = None

    def retrieve(self, query: str) -> list[Document]:
        vector_docs = self.vector_store.get_vector_docs(
            query=query, top_k=self.vector_top_k)
        for doc in vector_docs:
            doc.metadata = dict(doc.metadata or {})
            doc.metadata["retriever"] = "vector"

        bm25_docs = self.retrieve_bm25(query=query, top_k=self.bm25_top_k)
        merged_docs = self.merge_and_dedup(vector_docs, bm25_docs)
        merged_docs = merged_docs[: self.fusion_top_k]

        logger.info(
            f"[HybridRetriever]混合召回成功，向量召回数量={len(vector_docs)}，BM25召回数量={len(bm25_docs)}，合并后数量={len(merged_docs)}"
        )
        return merged_docs

    def retrieve_bm25(self, query: str, top_k: int) -> list[Document]:
        self.ensure_bm25_index()
        if not self.bm25 or not self.bm25_documents:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_idx = sorted(range(len(scores)),
                            key=lambda i: scores[i], reverse=True)

        results: list[Document] = []
        for idx in ranked_idx:
            if len(results) >= top_k:
                break
            score = float(scores[idx])
            if score <= 0:
                continue

            source_doc = self.bm25_documents[idx]
            metadata = dict(source_doc.metadata or {})
            metadata["retriever"] = "bm25"
            metadata["pre_score"] = score
            results.append(
                Document(page_content=source_doc.page_content, metadata=metadata))

        return results

    def ensure_bm25_index(self):
        if self.bm25 is not None:
            return

        with self.bm25_lock:
            if self.bm25 is not None:
                return

            split_docs = self.load_vector_docs()
            if not split_docs:
                logger.warning("[HybridRetriever]BM25语料为空")
                self.bm25 = None
                self.bm25_documents = []
                return

            corpus_tokens = [self.tokenize(doc.page_content) or [
                " "] for doc in split_docs]
            self.bm25 = BM25Okapi(corpus_tokens)
            self.bm25_documents = split_docs
            logger.info(
                f"[HybridRetriever]BM25索引已就绪，分片数量={len(split_docs)}，分词器={self.bm25_tokenizer}"
            )

    def load_vector_docs(self) -> list[Document]:
        try:
            docs = self.vector_store.get_all_vector_docs()
            if not docs:
                logger.warning("[HybridRetriever]向量库中没有可用于BM25的分片")
                return []
            return docs
        except Exception as exc:
            logger.warning(f"[HybridRetriever]读取向量库分片失败，错误信息={exc}")
            return []

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []

        mode = self.bm25_tokenizer
        if mode == "jieba":
            return self.tokenize_with_jieba(text)
        if mode == "bert":
            return self.tokenize_with_bert(text)
        if mode == "char":
            return self.tokenize_with_char(text)

        logger.warning(
            f"[HybridRetriever]未知分词器={mode}，回退到jieba"
        )
        return self.tokenize_with_jieba(text)

    def tokenize_with_char(self, text: str) -> list[str]:
        return [ch for ch in text.lower() if not ch.isspace()]

    def tokenize_with_jieba(self, text: str) -> list[str]:
        try:
            tokens = [token.strip()
                      for token in jieba.cut(text) if token.strip()]
            return self.post_process_tokens(tokens or self.tokenize_with_char(text))
        except Exception as exc:
            logger.warning(
                f"[HybridRetriever]jieba分词失败，回退到字符切分，错误信息={exc}"
            )
            return self.post_process_tokens(self.tokenize_with_char(text))

    def tokenize_with_bert(self, text: str) -> list[str]:
        try:
            if self._bert_tokenizer is None:
                if AutoTokenizer is None:
                    raise ImportError("未安装transformers，无法使用bert分词")
                self._bert_tokenizer = AutoTokenizer.from_pretrained(
                    self.bert_model_name,
                    local_files_only=False,
                )
            tokenizer = self._bert_tokenizer
            if tokenizer is None:
                return self.tokenize_with_jieba(text)
            tokens = tokenizer.tokenize(text)
            return self.post_process_tokens(tokens or self.tokenize_with_char(text))
        except Exception as exc:
            logger.warning(
                f"[HybridRetriever]bert分词失败（模型={self.bert_model_name}），回退到jieba，错误信息={exc}"
            )
            return self.tokenize_with_jieba(text)

    def post_process_tokens(self, tokens: list[str]) -> list[str]:
        if not tokens:
            return []

        min_len = max(self.bm25_min_token_len, 1)
        filtered = [token for token in tokens if len(token) >= min_len]

        stopwords = self.load_stopwords()
        if not stopwords:
            return filtered

        return [token for token in filtered if token not in stopwords]

    def load_stopwords(self) -> set[str]:
        if self._stopwords_cache is not None:
            return self._stopwords_cache

        stopwords_path = get_abs_path(self.bm25_stopwords_path)
        try:
            with open(stopwords_path, "r", encoding="utf-8") as file:
                words = {
                    line.strip().lower()
                    for line in file.readlines()
                    if line.strip() and not line.strip().startswith("#")
                }
                self._stopwords_cache = words
                logger.info(
                    f"[HybridRetriever]停用词已加载，数量={len(words)}，路径={self.bm25_stopwords_path}"
                )
                return words
        except FileNotFoundError:
            logger.warning(
                f"[HybridRetriever]未找到停用词文件，跳过过滤，路径={self.bm25_stopwords_path}"
            )
        except Exception as exc:
            logger.warning(
                f"[HybridRetriever]停用词加载失败，跳过过滤，路径={self.bm25_stopwords_path}，错误信息={exc}"
            )

        self._stopwords_cache = set()
        return self._stopwords_cache

    def doc_key(self, doc: Document) -> str:
        source = str((doc.metadata or {}).get("source", ""))
        content = doc.page_content or ""
        raw = f"{source}|{content}".encode("utf-8", errors="ignore")
        return hashlib.md5(raw).hexdigest()

    def merge_and_dedup(self, vector_docs: list[Document], bm25_docs: list[Document]) -> list[Document]:
        merged: list[Document] = []
        seen = set()

        for doc in vector_docs + bm25_docs:
            key = self.doc_key(doc)
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)

        return merged


if __name__ == "__main__":
    service = HybridRetriever(VectorStoreService())
    query = "机器人开机没反应指示灯不亮怎么回事？"
    docs = service.retrieve(query)
    print(f"query: {query}")
    print(f"docs count: {len(docs)}")
    for index, doc in enumerate(docs, start=1):
        print(f"参考文件{index}：")
        print(f"source: {(doc.metadata or {}).get('source', '')}")
        print(f"retriever: {(doc.metadata or {}).get('retriever', '')}")
        print(f"rank_score: {(doc.metadata or {}).get('rank_score', '')}")
        print(doc.page_content)
