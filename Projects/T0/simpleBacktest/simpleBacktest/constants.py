# -*- coding; utf-8 -*-
"""
Project: main.py
File: constants.PY
Time: 9:54
Date: 2022/6/15
AUTHOR: ZhihanWU<cv007wzh@yeah.net>
"""
from enum import Enum

class ActionType(Enum):
    """
    Buy: 多单
    Sell: 空单
    BuyClose: 多单平仓
    SellClose: 空单平仓
    Cancel: 撤单
    ForceExit: 超时平仓
    """
    Buy = "Buy"
    Sell = "Sell"
    BuyClose = "BuyClose"
    SellClose = "SellClose"

    Cancel = "Cancel"
    ForceExit = "ForceExit"


class OrderType(Enum):
    """
    Market: 市价单
    Limit：限价单
    Stop：止损
    Trailing_stop：追踪止损
    """
    Market = "Market"
    Limit = "Limit"
    Stop = "Stop"
    Trailing_stop = "Trailing_stop"

    Limit_pct = "Limit_pct"
    Stop_pct = "Stop_pct"
    Trailing_stop_pct = "Trailing_stop_pct"


class OrderStatus(Enum):

    Created = "Created"
    Submitted = "Submitted"
    Partial = "Partial"
    Completed = "Completed"
    Canceled = "Canceled"
    Expired = "Expired"
    Margin = "Margin"
    Rejected = "Rejected"
    Triggered = "Triggered"


class EVENT(Enum):
    Market_updated = "Market_updated"
    Data_cleaned = "Data_cleaned"
    Signal_generated = "Signal_generated"
    Submit_order = "Submit_order"
    Record_result = "Record_result"