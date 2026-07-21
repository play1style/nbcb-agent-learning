"""贷款计算工具.

计算等额本息贷款的月供和总利息。
"""

from __future__ import annotations

from app.tools.registry import tool_registry


@tool_registry.register(
    name="calculate_loan",
    description="计算等额本息贷款的月供和总利息。"
    "当用户询问贷款金额计算时使用。"
    "参数：principal-贷款本金（万元），annual_rate-年利率（百分比），"
    "years-贷款期限（年）",
)
def calculate_loan(
    principal: float, annual_rate: float, years: int
) -> dict:
    """计算等额本息月供和总利息.

    使用等额本息还款公式计算每月还款额和总利息支出。

    Args:
        principal: 贷款本金（万元）
        annual_rate: 年利率（百分比，如3.5表示3.5%）
        years: 贷款期限（年）

    Returns:
        计算结果字典，包含月供、总利息、总还款额等

    Examples:
        >>> result = calculate_loan(100, 3.5, 30)
        >>> "monthly_payment" in result
        True
        >>> result["principal_wan"]
        100
    """
    print("调用calculate_loan工具开始==================")
    # 校验参数
    if principal <= 0:
        raise ValueError("贷款本金必须大于0")
    if annual_rate <= 0:
        raise ValueError("年利率必须大于0")
    if years <= 0:
        raise ValueError("贷款期限必须大于0")

    # 转换为元
    principal_yuan = principal * 10000
    # 月利率
    monthly_rate = (annual_rate / 100) / 12
    # 还款月数
    total_months = years * 12

    if monthly_rate == 0:
        monthly_payment = principal_yuan / total_months
    else:
        # 等额本息公式：M = P * [r(1+r)^n] / [(1+r)^n - 1]
        factor = (1 + monthly_rate) ** total_months
        monthly_payment = (
            principal_yuan * monthly_rate * factor / (factor - 1)
        )

    total_payment = monthly_payment * total_months
    total_interest = total_payment - principal_yuan

    return {
        "principal_wan": principal,
        "annual_rate": annual_rate,
        "years": years,
        "monthly_payment_yuan": round(monthly_payment, 2),
        "total_interest_yuan": round(total_interest, 2),
        "total_payment_yuan": round(total_payment, 2),
    }
