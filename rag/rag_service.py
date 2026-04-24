"""
总结服务类：用户提问，搜索参考资料，将提问与参考资料提交给模型，让模型总结回复
"""
from rag.vector_store import VectorStoreService
from rag.hybrid_retriever import HybridRetriever
from rag.rerank_service import RerankService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


def print_prompt(prompt):
    print("=== prompt start ===")
    print(prompt.to_string())
    print("=== prompt end ===")
    return prompt


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.hybrid_retriever = HybridRetriever(self.vector_store)
        self.rerank_service = RerankService()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        candidate_docs = self.hybrid_retriever.retrieve(query)
        return self.rerank_service.rerank(query=query, docs=candidate_docs)

    def rag_summarize(self, query: str) -> str:
        context_docs = self.retriever_docs(query)
        context = ""
        cnt = 0
        for doc in context_docs:
            cnt += 1
            metadata = doc.metadata or {}
            retriever = metadata.get("retriever", "unknown")
            pre_score = metadata.get("pre_score", "N/A")
            rank_score = metadata.get("rank_score", "N/A")
            context += (
                f"参考资料{cnt}：\n{doc.page_content}\n"
                f"参考来源: {retriever}\n"
                f"检索分值: {pre_score}\n"
                f"重排分值: {rank_score}\n"
                f"参考元数据:\n{metadata}\n\n"
            )
        return self.chain.invoke(
            {
                "input": query,
                "context": context
            }
        )


if __name__ == "__main__":
    rag = RagSummarizeService()
    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
