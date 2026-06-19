import unittest 
import numpy as np 
import torch
from src.cifar_models.preresnet import PreActResNet20

class TestNoise(unittest.TestCase):
    def test_noise(self):
        SEED = 2025071410
        def get_gen():
            gen = torch.Generator(device="cuda").manual_seed(SEED)
            return gen 
        def get_np_gen():
            gen = np.random.default_rng(SEED)
            return gen
        x = torch.randn((1, 3, 32, 32), generator=get_gen(), device="cuda")
        targets = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unsqueeze(0).cuda()
        resnet = PreActResNet20().cuda()
        out1, targets_a1, targets_b1, lam1 = resnet(x, targets, mixup_alpha=1.0, add_noise_level=0.5, 
                                                    mult_noise_level=0.5, sparse_level=0.65, torch_gen=get_gen(), numpy_gen=get_np_gen())
        out2, targets_a2, targets_b2, lam2 = resnet(x, targets, mixup_alpha=1.0, add_noise_level=0.5, 
                                                    mult_noise_level=0.5, sparse_level=0.65, torch_gen=get_gen(), numpy_gen=get_np_gen())
        # out1, targets_a1, targets_b1, lam1 = resnet(x, targets, mixup_alpha=1.0, add_noise_level=0.5, mult_noise_level=0.5, sparse_level=0.65, torch_gen=get_gen())
        # out2, targets_a2, targets_b2, lam2 = resnet(x, targets, mixup_alpha=1.0, add_noise_level=0.5, mult_noise_level=0.5, sparse_level=0.65, torch_gen=get_gen())
        # mask = torch.isclose(out1, out2)
        # print(mask)
        # print(out1[~mask])
        # print(out2[~mask])
        self.assertTrue(torch.isclose(out1, out2).all())
        self.assertTrue(torch.isclose(targets_b1, targets_b2).all())
        self.assertTrue(torch.isclose(targets_a1, targets_a2).all())
        self.assertEqual(lam1, lam2)
