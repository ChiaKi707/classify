# -*- coding: utf-8 -*-
"""三个模型结构和四种累积光谱表示接口测试。"""

import unittest

import torch

from models import available_architectures, build_model
from representations import available_representations, get_representation


class ExperimentInterfaceTests(unittest.TestCase):
    def test_registered_models_accept_all_four_representations(self):
        self.assertEqual(available_architectures(), ("msnet", "cnn1d", "resnet1d"))
        self.assertEqual(len(available_representations()), 4)
        for architecture in available_architectures():
            for channels, name in enumerate(available_representations(), 1):
                with self.subTest(architecture=architecture, channel_set=name):
                    self.assertEqual(len(get_representation(name).channels), channels)
                    model = build_model(
                        architecture, in_channels=channels, dropout=.15,
                        num_classes=3, model_options={},
                    ).eval()
                    with torch.no_grad():
                        output = model(torch.zeros(2, channels, 64))
                    self.assertEqual(tuple(output.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
