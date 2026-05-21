"""
AiQuant 内置数据源注册文件。

import 本模块即自动注册默认数据源:
    import data.service_defaults

注册后可用 key:
    - "funding_rate"  -> fetch_funding_rate
    - "open_interest" -> fetch_open_interest
"""

from data.market_data import fetch_funding_rate, fetch_open_interest
from data.service import register

register("funding_rate", fetch_funding_rate)
register("open_interest", fetch_open_interest)
