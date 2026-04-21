import os
import random

from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


rag = RagSummarizeService()
user_ids = ["1001", "1002", "1003", "1004",
            "1005", "1006", "1007", "1008", "1009", "1010"]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05",
             "2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]
external_data = {}


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description="获取指定城市天气信息，以消息字符串形式返回")
def get_weather(city: str) -> str:
    return f"{city}的天气晴朗，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"


@tool(description="获取用户所在城市名称，以纯字符串形式返回")
def get_user_location() -> str:
    return random.choice(["北京", "上海", "广州", "深圳", "杭州", "成都"])


@tool(description="获取用户ID，以字符串形式返回")
def get_user_id() -> str:
    return random.choice(user_ids)


@tool(description="获取当前月份，以字符串形式返回")
def get_current_month() -> str:
    return random.choice(month_arr)


def generator_external_data():
    """
    返回一个字典
    {
        "user_id":{
            "month":{"特征":xxx,"效率":xxx"}
            "month":{"特征":xxx,"效率":xxx"}
            "month":{"特征":xxx,"效率":xxx"}
            ...
        },
        "user_id":{
            "month":{"特征":xxx,"效率":xxx"}
            "month":{"特征":xxx,"效率":xxx"}
            "month":{"特征":xxx,"效率":xxx"}
            ...
        }
    }
    """
    if not external_data:
        external_data_path = get_abs_path(agent_config["external_data_path"])
        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件不存在: {external_data_path}")
        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:  # Skip header
                arr: list(str) = line.strip().split(",")

                user_id = arr[0].replace('"', "")
                feature = arr[1].replace('"', "")
                efficiency = arr[2].replace('"', "")
                consumables = arr[3].replace('"', "")
                comparison = arr[4].replace('"', "")
                time = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "feature": feature,
                    "efficiency": efficiency,
                    "consumables": consumables,
                    "comparison": comparison
                }


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以字符串形式返回，若未检索到，返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generator_external_data()

    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(
            f"[fetch_external_data]未找到用户 {user_id} 在月份 {month} 的使用记录")
        return ""


@tool(description="无入参无返回值，调用后触发中间件为报告生成的场景动态注入上下文信息")
def fill_context_for_report():
    return "fill_context_for_report工具被调用"


if __name__ == "__main__":
    print(fetch_external_data("1005", "2025-05"))
