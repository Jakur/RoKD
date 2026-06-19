## Robust Knowledge Distillation (RoKD): Boosting Robustness of Compressed Models

This code implements our paper: Small Yet Robust: Knowledge Distillation Improves Out-of-Distribution Robustness in Compact Vision Models (ECML PKDD 2026). It is based on the codebase of [NoisyMix](https://github.com/erichson/NoisyMix) https://proceedings.mlr.press/v238/erichson24a.html, which is in turn based on the code from [AugMix](https://github.com/google-research/augmix) https://arxiv.org/abs/1912.02781. 

Our work focuses on improving the state-of-the-art in Out-of-Distribution robustness using knowledge distillation. Our RoKD framework can produce models which are compact, yet robust, and allows for ensembling models for fine-grained control of model overhead. 

## Environment
```
pip install lightly robustness gdown
mkdir cifar10_models
mkdir cifar100_models
```
## Model Training 
```
# Train a teacher model 
export CUDA_VISIBLE_DEVICES=0; python3 cifar.py --arch wideresnet28 --augmix 1 --jsd 1 --alpha 1.0 --manifold_mixup 1 --add_noise_level 0.5 --mult_noise_level 0.5 --sparse_level 0.65 --seed 1

# Train reduced size Wide ResNet-28 with distillation and CRD auxiliary KD
export CUDA_VISIBLE_DEVICES=0; python3 cifar.py --arch halfresnet28 --augmix 1 --jsd 1 --alpha 1.0 --manifold_mixup 1 --add_noise_level 0.5 --mult_noise_level 0.5 --sparse_level 0.65 --seed 1 --distill --teacher-path ./teachers/wrn28_teacher.pt -a 0.5 --kd_schedule log --extra_kd crd
```
## Robust Evaluation 
```
for f in models/*; do python evaluate_robustness.py --dir $f/; done
```

