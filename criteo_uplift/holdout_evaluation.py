# 使用测试集评价已冻结的候选模型，并保存指标、曲线和预测。
# Notebook 04 的核心入口：evaluate_test_set；不重新训练或重新选择策略。
# 结果文件包括 test_policies.csv、test_ranking.csv、curves.csv、ranking_intervals.csv、predictions.npz。

from pathlib import Path

import numpy as np
import pandas as pd

from .data import FEATURES
from .pipeline_io import write_json
from .evaluation import (
    ate_summary,
    calibration_table,
    evaluate_policies,
    ranking_confidence_intervals,
    ranking_metrics,
    uplift_curve,
)


#-----------------------------------------------------------------------------------------------------------------------
# 函数：evaluate_test_set —— 评价已冻结模型并保存测试集结果。
# 输入：
#     test：测试集数据表，用于冻结选择后的最终评价。
#     models：模型名称到已拟合模型对象的映射。
#     outcome：当前结果列名，通常为 conversion 或 visit。
#     assignment_probability：固定使用的处理分配概率估计值。
#     seed：控制抽样、同分排序或模型拟合的随机种子。
#     bootstrap：重抽样次数；支持时可用 0 仅计算点估计。
#     primary_budget：主策略允许触达的人群比例。
#     out：当前实验的结果保存目录。
#     progress：接收进度文字的函数。
# 输出：
#     policies：测试集策略评价表。
# 说明：计算排序、预算、校准和区间结果，保存逐行预测；不根据测试结果重新选择策略。
#-----------------------------------------------------------------------------------------------------------------------
def evaluate_test_set(
    test,
    models,
    *,
    outcome,
    assignment_probability,
    seed,
    bootstrap,
    primary_budget,
    out,
    progress=print,
):
    out = Path(out)
    test_scores = {
        name: model.predict(test[FEATURES])
        for name, model in models.items()
    }
    test_scores["propensity"] = models["t_xgboost"].predict_outcomes(
        test[FEATURES]
    )[1]

    rankings = []
    curves = []
    for name, score in test_scores.items():
        rankings.append(
            {
                "policy": name,
                **ranking_metrics(
                    test[outcome],
                    test.treatment,
                    score,
                    assignment_probability,
                    seed,
                ),
                "mean_score": float(np.mean(score)),
                "negative_fraction": float(np.mean(score < 0)),
            }
        )
        curve = uplift_curve(
            test[outcome],
            test.treatment,
            score,
            assignment_probability,
            seed,
        )
        # 指标使用完整曲线；仅压缩用于绘图的保存表。
        curve = curve.iloc[
            np.unique(np.linspace(0, len(curve) - 1, 501).astype(int))
        ].copy()
        curve.insert(0, "policy", name)
        curves.append(curve)
        if name != "propensity":
            calibration_table(test[outcome], test.treatment, score).to_csv(
                out / f"calibration_{name}.csv", index=False
            )

    pd.DataFrame(rankings).to_csv(out / "test_ranking.csv", index=False)
    pd.concat(curves).to_csv(out / "curves.csv", index=False)

    budgets = sorted(set([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, primary_budget]))
    policies = evaluate_policies(
        test[outcome],
        test.treatment,
        test_scores,
        assignment_probability,
        budgets=budgets,
        bootstrap=bootstrap,
        seed=seed,
    )
    policies.to_csv(out / "test_policies.csv", index=False)
    progress("计算冻结模型的配对 AUUC / Qini 区间")
    ranking_confidence_intervals(
        test[outcome],
        test.treatment,
        test_scores,
        assignment_probability,
        bootstrap=bootstrap,
        seed=seed,
    ).to_csv(out / "ranking_intervals.csv", index=False)

    write_json(
        ate_summary(test[outcome], test.treatment),
        out / "ate_test.json",
    )
    np.savez_compressed(
        out / "predictions.npz",
        row_id=test.row_id.to_numpy(),
        y=test[outcome].to_numpy(),
        t=test.treatment.to_numpy(),
        **test_scores,
    )
    return policies
