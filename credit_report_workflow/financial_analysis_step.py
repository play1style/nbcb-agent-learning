"""
步骤3：基于原始数据计算关键财务指标。
"""


def analyze_financial(state):
    """计算销售净利率、资产负债率、应收账款周转次数。"""
    net_margin = state["net_profit"] / state["annual_revenue"] * 100
    debt_ratio = state["total_liabilities"] / state["total_assets"] * 100
    turnover = 365 / state["receivable_days"] if state["receivable_days"] > 0 else 0

    state["financial"] = {
        "销售净利率": round(net_margin, 2),
        "资产负债率": round(debt_ratio, 2),
        "应收账款周转次数": round(turnover, 2),
    }
    return state
