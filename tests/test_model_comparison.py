# -*- coding: utf-8 -*-
"""已训练模型汇总必须只按验证结果选通道和模型。"""

import unittest

import pandas as pd

from compare_models import select_model, summarize


class ModelComparisonTests(unittest.TestCase):
    def test_best_channel_and_model_are_selected_by_validation(self):
        rows = []
        scores = {
            ("cnn1d", "raw"): .70,
            ("cnn1d", "raw_snv"): .82,
            ("resnet1d", "raw"): .86,
            ("resnet1d", "raw_snv"): .80,
        }
        for (architecture, channel_set), validation in scores.items():
            for seed, offset in ((66, 0), (67, -.01)):
                rows.append(
                    {
                        "architecture": architecture,
                        "channel_set": channel_set,
                        "training_seed": seed,
                        "trainable_parameters": 100,
                        "best_validation_loss": 1 - validation,
                        "best_validation_accuracy": validation + offset,
                        "best_validation_macro_f1": validation + offset,
                        "test_accuracy": .99 if architecture == "cnn1d" else .60,
                        "test_macro_f1": .99 if architecture == "cnn1d" else .60,
                        "training_seconds": 1.0,
                        "checkpoint": f"{architecture}-{channel_set}-{seed}.pt",
                        "metrics_file": f"{architecture}-{channel_set}-{seed}.json",
                    }
                )
        results = pd.DataFrame(rows)
        summary, model_summary = summarize(results)
        self.assertEqual(
            dict(zip(model_summary["architecture"], model_summary["channel_set"])),
            {"resnet1d": "raw", "cnn1d": "raw_snv"},
        )
        self.assertEqual(model_summary.iloc[0]["architecture"], "resnet1d")
        selection = select_model(results, model_summary, "comparison")
        self.assertEqual(selection["architecture"], "resnet1d")
        self.assertEqual(selection["channel_set"], "raw")
        self.assertEqual(selection["training_seed"], 66)
        self.assertNotEqual(
            selection["architecture"], "cnn1d",
            "测试正确率更高不能改变验证集选模结果",
        )

    def test_summary_without_legacy_robustness_columns(self):
        rows = []
        for architecture, score in (("cnn1d", .8), ("resnet1d", .85)):
            rows.append(
                {
                    "architecture": architecture,
                    "channel_set": "raw",
                    "training_seed": 66,
                    "trainable_parameters": 100,
                    "best_validation_loss": 1 - score,
                    "best_validation_accuracy": score,
                    "best_validation_macro_f1": score,
                    "test_accuracy": score,
                    "test_macro_f1": score,
                    "training_seconds": 1.0,
                    "checkpoint": f"{architecture}.pt",
                    "metrics_file": f"{architecture}.json",
                }
            )
        summary, model_summary = summarize(pd.DataFrame(rows))
        self.assertNotIn("robustness_accuracy_mean", summary)
        self.assertEqual(model_summary.iloc[0]["architecture"], "resnet1d")


if __name__ == "__main__":
    unittest.main()
