# -*- coding: utf-8 -*-
"""回归验证：最低损失、有效改善、权重恢复和结果记录必须一致。"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from config import DEFAULT_DATA
from predict import load_checkpoint
import train
from training_control import TRAINING_POLICY_VERSION, ValidationMonitor


class ValidationMonitorTests(unittest.TestCase):
    def test_small_losses_keep_updating_checkpoint_after_reference_freezes(self):
        monitor = ValidationMonitor(min_delta=1e-4, patience=3)
        losses = [0.3, 4.5e-5, 4e-5, 2e-5, 1e-5]
        decisions = [monitor.update(epoch, loss) for epoch, loss in enumerate(losses, 1)]
        self.assertTrue(all(item.checkpoint_updated for item in decisions))
        self.assertEqual([item.early_stopping_improved for item in decisions],
                         [True, True, False, False, False])
        self.assertFalse(any(item.should_stop for item in decisions[:-1]))
        self.assertTrue(decisions[-1].should_stop)
        self.assertEqual(monitor.best_epoch, 5)
        self.assertEqual(monitor.best_loss, 1e-5)
        self.assertEqual(monitor.reference_epoch, 2)
        self.assertEqual(monitor.stale_epochs, 3)

    def test_small_improvements_accumulate_against_last_significant_reference(self):
        monitor = ValidationMonitor(min_delta=0.1, patience=5)
        decisions = [monitor.update(epoch, loss)
                     for epoch, loss in enumerate([1.0, .96, .92, .89], 1)]
        self.assertTrue(decisions[-1].early_stopping_improved)
        self.assertEqual(monitor.reference_epoch, 4)
        self.assertEqual(monitor.stale_epochs, 0)

    def test_ties_and_regressions_keep_earliest_best_and_exhaust_patience(self):
        monitor = ValidationMonitor(min_delta=0, patience=2)
        monitor.update(1, .5)
        self.assertFalse(monitor.update(2, .5).checkpoint_updated)
        self.assertTrue(monitor.update(3, .6).should_stop)
        self.assertEqual(monitor.best_epoch, 1)
        self.assertEqual(monitor.reference_epoch, 1)

    def test_zero_delta_resets_patience_on_every_strict_improvement(self):
        monitor = ValidationMonitor(min_delta=0, patience=1)
        for epoch, loss in enumerate([.5, .49, .48], 1):
            decision = monitor.update(epoch, loss)
            self.assertTrue(decision.checkpoint_updated)
            self.assertTrue(decision.early_stopping_improved)
            self.assertFalse(decision.should_stop)

    def test_exact_delta_updates_checkpoint_but_does_not_reset_patience(self):
        monitor = ValidationMonitor(min_delta=.125, patience=1)
        monitor.update(1, .5)
        decision = monitor.update(2, .375)
        self.assertTrue(decision.checkpoint_updated)
        self.assertFalse(decision.early_stopping_improved)
        self.assertTrue(decision.should_stop)

    def test_invalid_loss_does_not_modify_monitor(self):
        for loss in (float('nan'), float('inf'), float('-inf'), -1):
            with self.subTest(loss=loss):
                monitor = ValidationMonitor(min_delta=1e-4, patience=2)
                monitor.update(1, .5)
                before = monitor.summary()
                with self.assertRaises(ValueError):
                    monitor.update(2, loss)
                self.assertEqual(monitor.summary(), before)

    def test_invalid_settings_and_nonconsecutive_epochs(self):
        for delta, patience in ((-1, 2), (float('nan'), 2), (0, 0), (0, 1.5)):
            with self.subTest(delta=delta, patience=patience):
                with self.assertRaises(ValueError):
                    ValidationMonitor(delta, patience)
        monitor = ValidationMonitor(0, 2)
        with self.assertRaises(ValueError):
            monitor.update(2, .5)


class TrainingSelectionIntegrationTests(unittest.TestCase):
    def test_saved_weights_test_evaluation_and_all_reports_use_absolute_minimum(self):
        # 控制验证损失和模型参数，避免依靠随机训练碰巧触发原始缺陷。
        # 第2轮后早停参考冻结，第3轮仍必须更新权重；第5轮停止后
        # 测试必须恢复第3轮参数，不能使用最后一轮参数。
        losses = [.3, 4.5e-5, 1e-5, 2e-5, 3e-5]
        epoch = 0
        validation_calls = 0
        tested_epoch = None

        def fake_train_epoch(model, loader, optimizer, *unused):
            nonlocal epoch
            epoch += 1
            with torch.no_grad():
                model.classifier.bias.fill_(epoch)
            optimizer.step()
            return .1

        def fake_predict(model, loader, device):
            nonlocal validation_calls, tested_epoch
            truth = loader.dataset.y.numpy()
            probabilities = np.full((len(truth), 3), .01, dtype=np.float32)
            probabilities[np.arange(len(truth)), truth] = .98
            if validation_calls < len(losses):
                loss = losses[validation_calls]
                validation_calls += 1
            else:
                tested_epoch = int(model.classifier.bias[0].item())
                loss = .123
            return probabilities, truth, loss

        with tempfile.TemporaryDirectory(prefix='spectra_selection_') as temporary:
            root = Path(temporary)
            arguments = ['--epochs', '8', '--patience', '3', '--min-delta', '.0001',
                         '--group-size', '1', '--device', 'cpu',
                         '--output-dir', str(root/'output'), '--pt-dir', str(root/'pt')]
            for role, name in DEFAULT_DATA:
                frame = pd.DataFrame(np.ones((3, 16)), columns=np.linspace(600, 1800, 16))
                frame.insert(0, '浓度', ['10^-3']*3)
                frame.insert(0, '编号', [1, 2, 3])
                path = root/name
                frame.to_csv(path, index=False, encoding='utf-8-sig')
                arguments.extend(['--data', f'{role}={path}'])
            args = train.build_parser().parse_args(arguments)
            with patch.object(train, 'train_epoch', side_effect=fake_train_epoch), \
                    patch.object(train, 'predict_loader', side_effect=fake_predict), \
                    contextlib.redirect_stdout(io.StringIO()):
                train.command_train(args)

            directory = root/'output/model3'
            checkpoint = load_checkpoint(root/'pt/model3/best.pt', torch.device('cpu'))
            metrics = json.loads((directory/'test_metrics.json').read_text(encoding='utf-8'))
            config = json.loads((directory/'training_config.json').read_text(encoding='utf-8'))
            history = pd.read_csv(directory/'tables/training_history.csv')
            self.assertEqual(epoch, 5)
            self.assertEqual(tested_epoch, 3)
            self.assertTrue(torch.equal(checkpoint['model_state']['classifier.bias'],
                                        torch.full((3,), 3.0)))
            self.assertEqual(metrics['best_epoch'], 3)
            self.assertEqual(metrics['best_validation_loss'], min(losses))
            self.assertEqual(metrics['early_stopping_reference_epoch'], 2)
            self.assertEqual(metrics['epochs_completed'], 5)
            self.assertTrue(metrics['stopped_early'])
            self.assertEqual(metrics['training_policy_version'], TRAINING_POLICY_VERSION)
            self.assertEqual(checkpoint['training_control'], config['training_control'])
            self.assertEqual(json.loads(json.dumps(checkpoint['metrics'], ensure_ascii=False)),
                             metrics)
            self.assertEqual(checkpoint['training_policy_version'], TRAINING_POLICY_VERSION)
            self.assertEqual(config['training_policy_version'], TRAINING_POLICY_VERSION)
            self.assertEqual(history.loc[history.val_loss.idxmin(), 'epoch'], 3)
            self.assertEqual(history.checkpoint_updated.tolist(),
                             [True, True, True, False, False])
            self.assertEqual(history.epochs_without_significant_improvement.tolist(),
                             [0, 0, 1, 2, 3])
            chinese = pd.read_csv(directory/'tables/训练记录.csv')
            self.assertIn('是否更新最低损失权重', chinese)
            summary = (directory/'结果摘要.txt').read_text(encoding='utf-8')
            self.assertIn('最佳训练轮次：3', summary)
            self.assertIn('早停参考轮次：2', summary)


if __name__ == '__main__':
    unittest.main()
