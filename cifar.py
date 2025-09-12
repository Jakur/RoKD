"""
This code is based on the following two repositories
* https://github.com/google-research/augmix
* https://github.com/erichson/NoisyMixup
"""

import argparse
import os
from copy import deepcopy

import augmentations
import numpy as np
import math
import time
import json
import tqdm

from src.cifar_models import preactwideresnet18, preactresnet18, wideresnet28, preactresnet20, preactresnet32, halfwideresnet28, VIT, Ensemble
from kd_utils import dkd_loss, freeze, RKDLoss, CRDLoss, CIFAR10InstanceSample, CIFAR100InstanceSample, strip_dataparallel, Attention, NSTLoss


import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

from src.noisy_mixup import mixup_criterion
from src.tools import get_lr
from aug_utils import *

parser = argparse.ArgumentParser(
    description='Trains a CIFAR Classifier', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'cifar100'], help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--arch', '-m', type=str, default='preactresnet18',
                    choices=['preactresnet18', 'preactwideresnet18', 'wideresnet28', 'preactresnet20', 'preactresnet32', 'vit', 'halfresnet28'], help='Choose architecture.')
parser.add_argument('--seed', type=int, default=1,
                    metavar='S', help='random seed (default: 0)')
parser.add_argument('--workers', type=int, default=4,
                    help="Number of workers for PyTorch DataLoaders")
parser.add_argument('--tqdm', action='store_true', help="Enable Progress Bar")

# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--learning-rate', '-lr', type=float,
                    default=0.1, help='Initial learning rate.')
parser.add_argument('--train-batch-size', type=int,
                    default=128, help='Batch size.')
parser.add_argument('--test-batch-size', type=int, default=1000)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-wd', type=float,
                    default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--grad-steps', type=int, default=1, help="Number of steps for gradient accumulation. 1 for normal training")

# AugMix options
parser.add_argument('--augmix', type=int, default=1,
                    metavar='S', help='aug mixup (default: 1)')
parser.add_argument('--mixture-width', default=3, type=int,
                    help='Number of augmentation chains to mix per augmented example')
parser.add_argument('--mixture-depth', default=-1, type=int,
                    help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument('--aug-severity', default=3, type=int,
                    help='Severity of base augmentation operators')
parser.add_argument('--jsd', type=int, default=1, metavar='S',
                    help='JSD consistency loss (default: 1)')

parser.add_argument('--all-ops', '-all', action='store_true',
                    help='Turn on all operations (+brightness,contrast,color,sharpness).')

# Noisy Feature Mixup options
parser.add_argument('--alpha', type=float, default=0.0,
                    metavar='S', help='for mixup')
parser.add_argument('--manifold_mixup', type=int, default=0,
                    metavar='S', help='manifold mixup (default: 0)')
parser.add_argument('--add_noise_level', type=float,
                    default=0.0, metavar='S', help='level of additive noise')
parser.add_argument('--mult_noise_level', type=float, default=0.0,
                    metavar='S', help='level of multiplicative noise')
parser.add_argument('--sparse_level', type=float,
                    default=0.0, metavar='S', help='sparse noise')

# Distillation
parser.add_argument('--distill', action='store_true',
                    help="Enable distillation")
parser.add_argument('--extra_kd', type=str, default="none", choices=["none", "rkd", "crd", "dkd", "at", "nst"])
parser.add_argument('--rkd_wd', type=float, default=25)
parser.add_argument('--rkd_wa', type=float, default=50)
parser.add_argument('--crd_beta', type=float, default=0.4) # CRD it is 0.8 but I think this is huge 
parser.add_argument('--crd_randomize', action='store_true', help="Randomize weights of mixing representations")
parser.add_argument('--dkd_alpha', type=float, default=1.0,
                    help="DKD alpha parameter")
parser.add_argument('--dkd_beta', type=float, default=8.0,
                    help="DKD beta parameter")
parser.add_argument('--dkd_all', action='store_true', help="Use normal and augment samples for DKD")
parser.add_argument('--at_beta', type=float, default=1000)
parser.add_argument('--nst_beta', type=float, default=50)
parser.add_argument('--teacher-path', type=str,
                    help="Path to PyTorch saved model")
parser.add_argument('--kd_alpha', '-a', type=float, default=0.0,
                    help="Final weight for Standard KD (0-1)")
parser.add_argument('--kd_schedule', type=str,
                    default="linear", choices=["linear", "log", "cos", "const"])
parser.add_argument('--kd_temp', type=float, default=4.0,
                    help="Temperature parameter for knowledge distillation")
parser.add_argument('--save_freq', type=int, default=-1, help="Save every k epochs, -1 to disable")

start_time = int(time.time())
args = parser.parse_args()
args.crd = False
args.dkd = False
args.rkd = False
args.at = False
args.nst = False 
out_name = f'arch_{args.arch}_jsd_{args.jsd}_seed_{args.seed}_kd_{args.kd_alpha}_{args.kd_schedule}_{args.extra_kd}'
if args.extra_kd == "crd":
    args.crd = True 
    out_name += f"_{args.crd_beta}{args.crd_randomize}"
if args.extra_kd == "dkd":
    args.dkd = True
    out_name += f"_{args.dkd_alpha}a{args.dkd_beta}b{args.dkd_all}c"
if args.extra_kd == "rkd":
    args.rkd = True
    out_name += f"_{args.rkd_wd}a{args.rkd_wa}"
if args.extra_kd == "at":
    args.at = True
    extra_kd_loss_fn = Attention()
    extra_kd_weight = args.at_beta 
    out_name += f"_{args.at_beta}"
if args.extra_kd == "nst":
    args.nst = True
    extra_kd_loss_fn = NSTLoss()
    extra_kd_weight = args.nst_beta
    out_name += f"_{args.nst_beta}"

print(vars(args))

if args.seed != 0:
    numpy_rng = np.random.default_rng(args.seed)
    torch_rng = torch.Generator(device="cuda").manual_seed(args.seed)
else:
    numpy_rng = np.random.default_rng()
    torch_rng = torch.Generator(device="cuda")

def logarithmic_param_schedule(epoch, total_epochs, start, end):
    if epoch <= 0:
        return start

    # Use log(epoch + 1) to avoid log(0)
    log_epoch = math.log(epoch + 1)
    log_max = math.log(total_epochs)

    log_ratio = log_epoch / log_max  # normalized to [0, 1]
    return start + log_ratio * (end - start)


def linear_param_schedule(epoch, total_epochs, start, end):
    ratio = epoch / (total_epochs - 1)
    return start + ratio * (end - start)


def cosine_param_schedule(epoch, total_epochs, start, end):
    cosine_ratio = (1 - math.cos(math.pi * epoch / (total_epochs - 1))) / 2
    return start + cosine_ratio * (end - start)


if args.distill:
    teacher_net = torch.load(args.teacher_path)
    teacher_net = strip_dataparallel(teacher_net)
    if isinstance(teacher_net, Ensemble):
         # Todo fix this properly later
        teacher_net.models = torch.nn.ModuleList([strip_dataparallel(m) for m in teacher_net.models])
        teacher_net.linear = torch.nn.Identity()
    teacher_net.eval()
    freeze(teacher_net)
    if args.kd_schedule == "log":
        print("Log KD Scaling")
        kd_schedule = logarithmic_param_schedule
    elif args.kd_schedule == "cos":
        print("Cosine KD Scaling")
        kd_schedule = cosine_param_schedule
    elif args.kd_schedule == "const":
        print("Constant KD Scaling")
        kd_schedule = lambda _a, _b, _c, d: d 
    else:
        print("Linear KD Scaling")
        kd_schedule = linear_param_schedule


def get_mix(logits_all, num_images):
    t_logits_clean, t_logits_aug1, t_logits_aug2 = torch.split(
        logits_all, num_images)
    t_p_clean, t_p_aug1, t_p_aug2 = F.softmax(
        t_logits_clean, dim=1), F.softmax(
        t_logits_aug1, dim=1), F.softmax(
            t_logits_aug2, dim=1)
    t_p_mixture = torch.clamp(
        (t_p_clean + t_p_aug1 + t_p_aug2) / 3., 1e-7, 1).log()
    return t_p_mixture, t_logits_clean


def train(net, train_loader, optimizer, scheduler, epoch=0, crd_loss=None):
    """Train for one epoch."""
    net.train()
    loss_ema = 0.

    criterion = torch.nn.CrossEntropyLoss().cuda()
    pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}", disable=not args.tqdm)
    optimizer.zero_grad()
    features = {}
    student_handles = []
    teacher_handles = []
    flatten_repr = lambda x: torch.mean(torch.stack(torch.split(features[x], bs, dim=0), dim=0), dim=0)
    def get_pre_features(name):
        def hook(_model, input):
            features[name] = input
        return hook
    def get_features(name):
        def hook(_model, _input, output):
            features[name] = output
        return hook
    if args.crd or args.rkd:
        handle_student = net.linear.register_forward_pre_hook(get_pre_features('student'))
        handle_teacher = teacher_net.linear.register_forward_pre_hook(get_pre_features('teacher'))
        student_handles.append(handle_student)
        teacher_handles.append(handle_teacher)
    if args.at or args.nst:
        student_handles.append(net.layer1.register_forward_hook(get_features("student1")))
        student_handles.append(net.layer2.register_forward_hook(get_features("student2")))
        student_handles.append(net.layer3.register_forward_hook(get_features("student3")))
        teacher_handles.append(teacher_net.layer1.register_forward_hook(get_features("teacher1")))
        teacher_handles.append(teacher_net.layer2.register_forward_hook(get_features("teacher2")))
        teacher_handles.append(teacher_net.layer3.register_forward_hook(get_features("teacher3")))


    for i, data in pbar:
        if args.crd: 
            (images, targets, pos_idx, neg_idx) = data
            (pos_idx, neg_idx) = (pos_idx.cuda(), neg_idx.cuda())
        else: 
            (images, targets) = data

        bs = targets.size(0)

        if args.jsd == 0:
            images = images.cuda()
            targets = targets.cuda()
            loss = 0.0
            if args.alpha == 0.0:
                outputs = net(images)
            else:
                torch_rng_backup = torch.Generator(
                    device="cuda").set_state(torch_rng.get_state())
                numpy_rng_backup = deepcopy(numpy_rng)
                outputs, targets_a, targets_b, lam = net(images, targets=targets, jsd=args.jsd,
                                                         numpy_gen=numpy_rng,
                                                         torch_gen=torch_rng,
                                                         mixup_alpha=args.alpha,
                                                         manifold_mixup=args.manifold_mixup,
                                                         add_noise_level=args.add_noise_level,
                                                         mult_noise_level=args.mult_noise_level,
                                                         sparse_level=args.sparse_level)
                # if args.distill: 
                #     kd_loss = DistillKL(args.kd_temp)
                #     teacher_outputs, _, _, _ = net(images, targets=targets, jsd=args.jsd,
                #                                          numpy_gen=numpy_rng_backup,
                #                                          torch_gen=torch_rng_backup,
                #                                          mixup_alpha=args.alpha,
                #                                          manifold_mixup=args.manifold_mixup,
                #                                          add_noise_level=args.add_noise_level,
                #                                          mult_noise_level=args.mult_noise_level,
                #                                          sparse_level=args.sparse_level)
                #     loss += args.kd_alpha * kd_loss(outputs, teacher_outputs)
                    

            if args.alpha > 0:
                loss += mixup_criterion(
                    criterion, outputs, targets_a, targets_b, lam)
            else:
                loss += criterion(outputs, targets)

        elif args.jsd == 1:
            images_all = torch.cat(images, 0).cuda()
            targets = targets.cuda()

            torch_rng_backup = torch.Generator(
                device="cuda").set_state(torch_rng.get_state())
            numpy_rng_backup = deepcopy(numpy_rng)

            if args.alpha == 0.0:
                logits_all = net(images_all)
            else:
                logits_all, targets_a, targets_b, lam, _ = net(images_all, targets=targets, jsd=args.jsd,
                                                            numpy_gen=numpy_rng,
                                                            torch_gen=torch_rng,
                                                            mixup_alpha=args.alpha,
                                                            manifold_mixup=args.manifold_mixup,
                                                            add_noise_level=args.add_noise_level,
                                                            mult_noise_level=args.mult_noise_level,
                                                            sparse_level=args.sparse_level)

            num_images = images[0].size(0)
            logits_clean, logits_aug1, logits_aug2 = torch.split(
                logits_all, num_images)
            if args.alpha > 0:
                loss = mixup_criterion(
                    criterion, logits_clean, targets_a, targets_b, lam)
            else:
                loss = criterion(logits_clean, targets)

            if args.distill:
                # loss *= args.reward
                # Todo does this need grad info? 
                with torch.no_grad():
                    if args.alpha == 0:
                        t_logits_all = teacher_net(images_all)
                    else:
                        t_logits_all, t_targets_a, t_targets_b, _, mixup_index = teacher_net(images_all, targets=targets, jsd=args.jsd,
                                                                                    numpy_gen=numpy_rng_backup,
                                                                                    torch_gen=torch_rng_backup,
                                                                                    mixup_alpha=args.alpha,
                                                                                    manifold_mixup=args.manifold_mixup,
                                                                                    add_noise_level=args.add_noise_level,
                                                                                    mult_noise_level=args.mult_noise_level,
                                                                                    sparse_level=args.sparse_level)

                    t_p_mixture, t_logits_clean = get_mix(t_logits_all, num_images)

                if args.crd:
                    # loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
                    if args.crd_randomize:
                        rand = torch.randn((3), generator=torch_rng, device="cuda")
                        rand2 = torch.softmax(rand, dim=0)
                        rand = torch.softmax(-rand, dim=0)
                        
                        s_split = torch.split(rand.repeat_interleave(bs).unsqueeze(1) * features["student"][0], bs, dim=0)
                        t_split = torch.split(rand2.repeat_interleave(bs).unsqueeze(1) * features["teacher"][0], bs, dim=0)
                    else:
                        s_split = torch.split(features["student"][0], bs, dim=0)
                        t_split = torch.split(features["teacher"][0], bs, dim=0)
                    proportion = lam
                    left = neg_idx
                    right = neg_idx[mixup_index]
                    if lam < 0.5:
                        left, right = right, left
                        proportion = 1.0 - lam
                    assert(proportion >= 0.5)
                    stop = int(neg_idx.size(1) * proportion) 
                    use_neg = torch.cat((left[:, :stop], right[:, stop:]), dim=1)
                    if args.crd_randomize:
                        s_repr = torch.sum(torch.stack(s_split, dim=0), dim=0)
                        t_repr = torch.sum(torch.stack(t_split, dim=0), dim=0)
                    else:
                        s_repr = torch.mean(torch.stack(s_split, dim=0), dim=0)
                        t_repr = torch.mean(torch.stack(t_split, dim=0), dim=0)
                    loss_from_crd = args.crd_beta * crd_loss(s_repr, t_repr, pos_idx, use_neg).item()
                    loss += loss_from_crd

                if args.rkd:
                    s_split = torch.split(features["student"][0], bs, dim=0)
                    t_split = torch.split(features["teacher"][0], bs, dim=0)
                    s_repr = torch.mean(torch.stack(s_split, dim=0), dim=0)
                    t_repr = torch.mean(torch.stack(t_split, dim=0), dim=0)
                    rkd_loss_fn = RKDLoss(args.rkd_wd, args.rkd_wa)
                    loss += rkd_loss_fn(s_repr, t_repr)


            # JSD Loss

            p_clean, p_aug1, p_aug2 = F.softmax(
                logits_clean, dim=1), F.softmax(
                    logits_aug1, dim=1), F.softmax(
                        logits_aug2, dim=1)
            # JSD Teacher Mixing
            if args.distill:
                alpha = kd_schedule(epoch, args.epochs, 0.0, args.kd_alpha)
                assert (0.0 <= alpha and alpha <= 1.0)
                reward = 1.0 - alpha
                p_mixture = alpha * t_p_mixture + reward * \
                    torch.clamp((p_clean + p_aug1 + p_aug2) /
                                3., 1e-7, 1).log()
            else:
                p_mixture = torch.clamp(
                    (p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            if args.dkd:
                if args.dkd_all:
                    t_logits_clean, t_logits_aug1, t_logits_aug2 = torch.split(
                        t_logits_all, num_images)
                    check = [(logits_clean, t_logits_clean), (logits_aug1, t_logits_aug1), (logits_aug2, t_logits_aug2)]
                else:
                    check = [(logits_clean, t_logits_clean)]
                for log, t_log in check:
                    dkd = 0.125 * lam * dkd_loss(log, t_log, t_targets_a, args.dkd_alpha, 
                        args.dkd_beta, args.kd_temp)
                    dkd += 0.125 * (1.0 - lam) * dkd_loss(log, t_log, t_targets_b, args.dkd_alpha, 
                        args.dkd_beta, args.kd_temp)
                dkd /= len(check)

                loss += dkd
            if args.at or args.nst:
                s = [flatten_repr("student1"), flatten_repr("student2"), flatten_repr("student3")]
                t = [flatten_repr("teacher1"), flatten_repr("teacher2"), flatten_repr("teacher3")]
                loss += extra_kd_weight * sum(extra_kd_loss_fn(s, t))

                
            loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                        F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                        F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
            # loss = args.kd_alpha * kd_loss + args.reward * loss
        loss /= args.grad_steps
        loss.backward()
        if i % args.grad_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        loss_ema = loss_ema * 0.9 + float(loss) * 0.1
        pbar.set_postfix_str(f"Train Loss: {loss_ema:.3f}")

    for handle in student_handles:
        handle.remove()
    for handle in teacher_handles:
        handle.remove()
    
    return loss_ema


def test(net, test_loader):
    """Evaluate network on given dataset."""
    net.eval()
    total_loss = 0.
    total_correct = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()

    return total_loss / len(test_loader.dataset), total_correct / len(test_loader.dataset)


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load datasets
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4)])
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5] * 3, [0.5] * 3)])
    test_transform = preprocess

    if args.augmix == 0:
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(32, padding=4),
             transforms.ToTensor(),
             transforms.Normalize([0.5] * 3, [0.5] * 3),
             ])

    if args.dataset == 'cifar10':
        if args.crd :
            train_data = CIFAR10InstanceSample('./data/cifar', train=True, transform=train_transform, download=True)
        else:
            train_data = datasets.CIFAR10(
                './data/cifar', train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(
            './data/cifar', train=False, transform=test_transform, download=True)
        num_classes = 10
    else:
        if args.crd:
            train_data = CIFAR100InstanceSample('./data/cifar', train=True, transform=train_transform, download=True)
        else:
            train_data = datasets.CIFAR100(
                './data/cifar', train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(
            './data/cifar', train=False, transform=test_transform, download=True)
        num_classes = 100

    if args.augmix == 1:
        train_data = AugMixDataset(train_data, preprocess, args.jsd, args, contrastive=args.crd)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.train_batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.test_batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True)

    # Create model
    if args.arch == 'preactresnet18':
        net = preactresnet18(num_classes=num_classes)
    elif args.arch == 'preactwideresnet18':
        net = preactwideresnet18(num_classes=num_classes)
    elif args.arch == 'wideresnet28':
        net = wideresnet28(num_classes=num_classes)
    elif args.arch == 'halfresnet28':
        net = halfwideresnet28(num_classes=num_classes)
    elif args.arch == "preactresnet20":
        net = preactresnet20(num_classes=num_classes)
    elif args.arch == "preactresnet32":
        net = preactresnet32(num_classes=num_classes)
    elif args.arch == "vit":
        from transformers import ViTImageProcessor, ViTModel
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        net = VIT(ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').cuda(), processor, num_classes=num_classes)
        # inputs = processor(images=data, return_tensors="pt", do_normalize=False, do_rescale=False)

    if args.crd:
        t_features = teacher_net.num_features()
        s_features = net.num_features()
        crd_loss = CRDLoss(s_dim=s_features, t_dim=t_features).cuda()
        #module_list.append(crd_loss.embed_s)
        #module_list.append(crd_loss.embed_t)
        model_params = list(net.parameters())
        for name, param in crd_loss.embed_s.named_parameters():
            model_params.append(param)
        for name, param in crd_loss.embed_t.named_parameters():
            model_params.append(param)
    else:
        crd_loss = None
        model_params = net.parameters()

    optimizer = torch.optim.SGD(model_params,
                                args.learning_rate, momentum=args.momentum,
                                weight_decay=args.decay, nesterov=True)

    net = net.cuda()
    # net = torch.nn.DataParallel(net).cuda() # Distribute across all GPUs
    # cudnn.benchmark = True

    start_epoch = 0

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step, args.epochs * len(train_loader),
            1,  # lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))

    best_acc = 0
    DESTINATION_PATH = args.dataset + f"_models/{start_time}/"
    loss_data = []
    for epoch in range(start_epoch, args.epochs):
        train_loss_ema = train(
            net, train_loader, optimizer, scheduler, epoch=epoch, crd_loss=crd_loss)
        test_loss, test_acc = test(net, test_loader)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        if is_best:
            OUT_DIR = os.path.join(DESTINATION_PATH, "best_" + out_name)
            if not os.path.isdir(DESTINATION_PATH):
                os.mkdir(DESTINATION_PATH)
            torch.save(net, OUT_DIR+'.pt')
        if args.save_freq > 0 and epoch % args.save_freq == 0:
            out = os.path.join(DESTINATION_PATH, f"ep{epoch}_" + out_name)
            torch.save(net, out + '.pt')
            
        info = 'Epoch {0:3d} | Train Loss {1:.4f} |'\
            ' Test Accuracy {2:.2f}'.format(
                (epoch + 1), train_loss_ema, 100. * test_acc)
        print(info)
        loss_data.append(info + "\n")

    OUT_DIR = os.path.join(DESTINATION_PATH, "final_" + out_name)
    if not os.path.isdir(DESTINATION_PATH):
        os.mkdir(DESTINATION_PATH)
    with open(f"{DESTINATION_PATH}settings.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    with open(f"{DESTINATION_PATH}loss.txt", "w") as f:
        f.writelines(loss_data)
    torch.save(net, OUT_DIR+'.pt')


if __name__ == '__main__':
    main()
