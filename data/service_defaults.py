"""
Default data source registrations for AiQuant.

Import this module to register built-in data fetchers:
    import data.service_defaults

Available keys after import:
    - "funding_rate"  -> fetch_funding_rate
    - "open_interest" -> fetch_open_interest
"""

from data.market_data import fetch_funding_rate, fetch_open_interest
from data.service import register

register("funding_rate", fetch_funding_rate)
register("open_interest", fetch_open_interest)
