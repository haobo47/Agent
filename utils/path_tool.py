"""
为整个项目提供存储路径
"""
import os


def get_project_root() -> str:
    """
    获取工程所在根目录
    """
    current_file = os.path.abspath(__file__)
    # 获取文件所在文件夹绝对路径
    current_dir = os.path.dirname(current_file)
    # 获取项目根目录绝对路径
    project_root = os.path.dirname(current_dir)

    return project_root


def get_abs_path(relative_path: str) -> str:
    """
    获取相对路径，返回绝对路径
    """
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)


if __name__ == "__main__":
    print(get_abs_path("config/config.txt"))
