import os
import random
from ipaddress import ip_address, IPv4Address
from typing import Any

import requests

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
    target_city = (city or "").strip()
    if not target_city:
        ip = discover_public_ipv4()
        if ip:
            ip_data = call_gaode_api("/v3/ip", {"ip": ip})
            city_name = str(ip_data.get("city", "")).strip()
            province = str(ip_data.get("province", "")).strip()
            target_city = city_name or province

    if not target_city:
        logger.warning("[get_weather]城市参数为空且无法自动定位城市")
        return ""

    adcode = resolve_city_name_to_adcode(target_city)
    if not adcode:
        logger.warning(f"[get_weather]未找到城市编码 city={target_city}")
        return ""

    weather_data = call_gaode_api(
        "/v3/weather/weatherInfo", {"city": adcode, "extensions": "base"}
    )
    lives = weather_data.get("lives", []) if isinstance(
        weather_data, dict) else []

    if not lives:
        logger.warning(
            f"[get_weather]天气数据为空 city={target_city}, adcode={adcode}")
        return ""

    live = lives[0]
    report_city = str(live.get("city", "")).strip() or target_city
    weather = str(live.get("weather", "")).strip()
    temperature = str(live.get("temperature", "")).strip()
    humidity = str(live.get("humidity", "")).strip()
    wind_direction = str(live.get("winddirection", "")).strip()
    wind_power = str(live.get("windpower", "")).strip()
    report_time = str(live.get("reporttime", "")).strip()

    return (
        f"{report_city}当前天气{weather}，气温{temperature}摄氏度，"
        f"空气湿度{humidity}%，{wind_direction}风{wind_power}级，"
        f"数据更新时间{report_time}"
    )


@tool(description="获取用户所在城市名称，以纯字符串形式返回")
def get_user_location() -> str:
    ip = discover_public_ipv4()
    if not ip:
        return ""

    ip_data = call_gaode_api("/v3/ip", {"ip": ip})
    city_name = str(ip_data.get("city", "")).strip()
    province = str(ip_data.get("province", "")).strip()

    if city_name:
        return city_name
    if province:
        return province

    logger.warning(f"[get_user_location]IP定位结果为空 ip={ip}")
    return ""


@tool(description="获取用户ID，以字符串形式返回")
def get_user_id() -> str:
    return random.choice(user_ids)


@tool(description="获取当前月份，以字符串形式返回")
def get_current_month() -> str:
    return random.choice(month_arr)


def is_valid_ipv4(ip: str) -> bool:
    if not ip:
        return False

    try:
        return isinstance(ip_address(ip), IPv4Address)
    except ValueError:
        return False


def call_gaode_api(endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    base_url = str(agent_config["gaode_base_url"]).rstrip("/")
    gaode_key = str(agent_config["gaode_key"])
    timeout = int(agent_config["gaode_timeout"])

    if not base_url or not gaode_key:
        logger.warning("[gaode_api]缺少 gaode_base_url 或 gaode_key 配置")
        return {}

    request_params: dict[str, Any] = dict(params or {})
    request_params["key"] = gaode_key

    try:
        response = requests.get(
            f"{base_url}{endpoint}", params=request_params, timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.warning(f"[gaode_api]请求失败 endpoint={endpoint}, error={e}")
        return {}
    except ValueError as e:
        logger.warning(f"[gaode_api]解析JSON失败 endpoint={endpoint}, error={e}")
        return {}

    if str(data.get("status", "")) != "1":
        info = data.get("info", "")
        infocode = data.get("infocode", "")
        logger.warning(
            f"[gaode_api]接口返回失败 endpoint={endpoint}, info={info}, infocode={infocode}"
        )
        return {}

    return data


def discover_public_ipv4() -> str:
    providers = [
        "https://ipv4.icanhazip.com",
        "https://api.ipify.org",
    ]

    for url in providers:
        try:
            response = requests.get(url, timeout=3)
            response.raise_for_status()
            ip = response.text.strip()
            if is_valid_ipv4(ip):
                return ip
            logger.warning(
                f"[discover_public_ipv4]无效IP响应 provider={url}, ip={ip}")
        except requests.RequestException as e:
            logger.warning(
                f"[discover_public_ipv4]请求失败 provider={url}, error={e}")

    logger.warning("[discover_public_ipv4]未能发现有效公网IPv4")
    return ""


def resolve_city_name_to_adcode(city_name: str) -> str:
    if not city_name:
        return ""

    data = call_gaode_api("/v3/geocode/geo", {"address": city_name})
    geocodes = data.get("geocodes", []) if isinstance(data, dict) else []

    if not geocodes:
        logger.warning(
            f"[resolve_city_name_to_adcode]未找到城市编码 city={city_name}")
        return ""

    adcode = str(geocodes[0].get("adcode", "")).strip()
    if not adcode:
        logger.warning(f"[resolve_city_name_to_adcode]城市编码为空 city={city_name}")
        return ""

    return adcode


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
                arr: list[str] = line.strip().split(",")

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
    print(get_weather.invoke({"city": ""}))
