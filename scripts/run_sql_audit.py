# 读取冻结实验结果，执行 SQL 与 Python 的指标核对并输出审计报告。

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from criteo_uplift.runtime import configure_runtime

configure_runtime()

from criteo_uplift.sql_audit import run_sql_audit


#-----------------------------------------------------------------------------------------------------------------------
# 函数：main —— 根据命令行配置启动任务。
# 输入：
#     无。
# 输出：
#     返回值：None；执行请求的流程并保存或显示结果。
# 说明：根据命令行配置启动任务。
#-----------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='读取冻结实验结果，执行 SQL 与 Python 的指标核对并输出审计报告。')
    parser.add_argument(
        "--source", type=Path, default=ROOT / "artifacts/conversion_1000000_seed42"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/sql_audit_v1",
        help="new output directory; existing directories and data/ are rejected",
    )
    args = parser.parse_args()
    result = run_sql_audit(args.source, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "n_eligible": result["n_eligible"],
                "primary_budget": result["primary_budget"],
                "frozen_selected_policy": result["frozen_selected_policy"],
                "maximum_absolute_sql_python_difference": result[
                    "maximum_absolute_sql_python_difference"
                ],
                "source_unchanged": result["source_unchanged"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
