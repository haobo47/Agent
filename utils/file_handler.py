import os
import hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


# 获取文件md5的十六进制字符串
def get_file_md5_hex(file_path: str):
    if not os.path.exists(file_path):
        logger.error(f"[md5计算]文件不存在: {file_path}")
        return
    if not os.path.isfile(file_path):
        logger.error(f"[md5计算]路径不是文件: {file_path}")
        return
    md5_obj = hashlib.md5()
    chunk_size = 4096  # 4KB分片避免文件过大爆内存
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
            """
            等价：
            chunk = f.read(chunk_size)
            while chunk:
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            """
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"计算文件{file_path}md5失败, 错误信息: {e}")
        return None


# 返回文件夹内的文件列表（合法文件后缀)
def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):
    files = []
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]路径不是文件夹: {path}")
        return allowed_types
    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))
    return tuple(files)


def pdf_loader(file_path: str, password=None) -> list[Document]:
    return PyPDFLoader(file_path, password).load()


def txt_loader(file_path: str) -> list[Document]:
    return TextLoader(file_path, encoding="utf-8").load()
