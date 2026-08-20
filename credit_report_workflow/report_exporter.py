"""将尽调报告导出为 txt 文件。"""
import random
from datetime import datetime
from pathlib import Path


EXPORT_DIR = Path(__file__).resolve().parent / "export"


def export_report(report):
    """将报告写入 export 目录，文件名格式为 yyyyMMddHHMMSS + 两位随机数。"""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_suffix = f"{random.randint(0, 99):02d}"
    output_path = EXPORT_DIR / f"{timestamp}{random_suffix}.txt"
    output_path.write_text(report, encoding="utf-8")
    return output_path
