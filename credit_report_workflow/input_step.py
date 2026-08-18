"""
步骤1：输入企业原始数据。

学员可以修改以下数据，观察报告和风险判断的变化。
"""


def get_input():
    """返回企业原始数据字典。"""
    return {
        "company_name": "宁波纺织贸易有限公司",
        "industry": "纺织服装",
        "annual_revenue": 8000,      # 年营业收入，单位：万元
        "net_profit": 480,           # 净利润，单位：万元
        "receivable_days": 110,       # 应收账款周转天数
        "total_assets": 5000,        # 总资产，单位：万元
        "total_liabilities": 2800,   # 总负债，单位：万元
    }
