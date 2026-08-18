"""
步骤2：将原始数据整理为可读的基本信息。
"""


def build_basic_info(state):
    """整理企业基本信息。"""
    info = {
        "企业名称": state["company_name"],
        "所属行业": state["industry"],
        "年营业收入": f"{state['annual_revenue']}万元",
        "净利润": f"{state['net_profit']}万元",
    }
    state["basic_info"] = info
    return state
