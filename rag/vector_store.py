from langchain_chroma import Chroma
from utils.config_handler import chroma_config
from model.factory import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from langchain_core.documents import Document


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=embedding_model,
            persist_directory=chroma_config["persist_directory"],
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
            separators=chroma_config["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_config["k"]})

    def load_document(self):
        """
        从数据文件夹中读取数据文件，转为向量存入向量库
        计算文件的MD5值去重，
        return None
        """
        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_config["md5_hex_store"])):
                open(get_abs_path(
                    chroma_config["md5_hex_store"]), "w", encoding="utf-8").close()
                return False  # md5未处理

            with open(get_abs_path(chroma_config["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True  # md5处理过

                return False  # md5未处理

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_config["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith(".txt"):
                return txt_loader(read_path)
            if read_path.endswith(".pdf"):
                return pdf_loader(read_path)
            return []

        allowed_files_path = listdir_with_allowed_type(
            get_abs_path(chroma_config["data_path"]),
            tuple(chroma_config["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            # 获取文件md5
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]文件{path}已处理过，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]文件{path}无有效文本内容，跳过")
                    continue
                split_documnet: list[Document] = self.splitter.split_documents(
                    documents)
                if not split_documnet:
                    logger.warning(f"[加载知识库]文件{path}分割后无有效文档，跳过")
                    continue
                # 内容存入向量库
                self.vector_store.add_documents(split_documnet)
                # 记录已处理好的md5，避免重复加载
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]文件{path}加载成功")
            except Exception as e:
                logger.error(
                    # exc_info为True记录详细报错堆栈，Flase则只记录错误信息
                    f"[加载知识库]文件{path}加载失败，错误信息: {str(e)}", exc_info=True)


if __name__ == "__main__":
    vss = VectorStoreService()
    vss.load_document()
    retriever = vss.get_retriever()
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("===")
