"""
主题2：对公信贷尽调报告生成工作流（Python 版）

本文件为工作流入口，串联 5 个步骤完成尽调报告生成。
"""
from input_step import get_input
from basic_info_step import build_basic_info
from financial_analysis_step import analyze_financial
from risk_assessment_step import assess_risk
from report_generation_step import generate_report
from report_exporter import export_report


def run_workflow():
    """按顺序执行工作流各节点。"""
    state = get_input()              # 步骤1：输入数据
    state = build_basic_info(state)  # 步骤2：整理基本信息
    state = analyze_financial(state) # 步骤3：财务指标分析
    state = assess_risk(state)       # 步骤4：风险规则判断
    report = generate_report(state)  # 步骤5：生成结构化报告
    output_path = export_report(report)
    print(f"报告已输出到: {output_path}")


if __name__ == "__main__":
    run_workflow()
