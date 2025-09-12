import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from PIL import Image

eps = 1e-7

def strip_dataparallel(model):
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    else:
        return model

def freeze(model):
    for name, param in model.named_parameters():
        param.requires_grad_(False)

def unfreeze(model):
    for name, param in model.named_parameters():
        param.requires_grad_(True)

class NSTLoss(nn.Module):
    """like what you like: knowledge distill via neuron selectivity transfer"""
    def __init__(self):
        super(NSTLoss, self).__init__()
        pass

    def forward(self, g_s, g_t):
        return [self.nst_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def nst_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass

        f_s = f_s.view(f_s.shape[0], f_s.shape[1], -1)
        f_s = F.normalize(f_s, dim=2)
        f_t = f_t.view(f_t.shape[0], f_t.shape[1], -1)
        f_t = F.normalize(f_t, dim=2)

        # set full_loss as False to avoid unnecessary computation
        full_loss = True
        if full_loss:
            return (self.poly_kernel(f_t, f_t).mean().detach() + self.poly_kernel(f_s, f_s).mean()
                    - 2 * self.poly_kernel(f_s, f_t).mean())
        else:
            return self.poly_kernel(f_s, f_s).mean() - 2 * self.poly_kernel(f_s, f_t).mean()

    def poly_kernel(self, a, b):
        a = a.unsqueeze(1)
        b = b.unsqueeze(2)
        res = (a * b).sum(-1).pow(2)
        return res

class Attention(nn.Module):
    """Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer
    code: https://github.com/szagoruyko/attention-transfer"""
    def __init__(self, p=2):
        super(Attention, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        return [self.at_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def at_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))

class RKDLoss(nn.Module):
    """Relational Knowledge Disitllation, CVPR2019"""
    def __init__(self, w_d=25, w_a=50):
        super(RKDLoss, self).__init__()
        self.w_d = w_d
        self.w_a = w_a

    def forward(self, f_s, f_t):
        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)

        # RKD distance loss
        with torch.no_grad():
            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss_d = F.smooth_l1_loss(d, t_d)

        # RKD Angle loss
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss_a = F.smooth_l1_loss(s_angle, t_angle)

        loss = self.w_d * loss_d + self.w_a * loss_a

        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss
    
def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student+1e-5)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction='sum')
        * (temperature**2)
        / target.shape[0]
    )
    # print(tckd_loss)
    # if len(target.size()) > 1:
    #     label = torch.max(target, dim=1, keepdim=True)[1]
    # else:
    #     label = target.view(len(target), 1)

    # # N*class
    # N, c = logits_student.shape
    
    # mask = torch.ones_like(logits_student).scatter_(1, label, 0).bool()
    # logits_student = logits_student[mask].reshape(N, -1)
    # logits_teacher = logits_teacher[mask].reshape(N, -1)
    
    # pred_teacher_part2 = F.softmax(
    #     logits_teacher / temperature, dim=1
    # )
    
    # log_pred_student_part2 = F.log_softmax(
    #     logits_student / temperature, dim=1
    # )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    # print(pred_teacher_part2)
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    # print(log_pred_student_part2)
    # print('\n\n')
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
        * (temperature**2)
        / target.shape[0]
    )
    # print(nckd_loss)
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

# parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
# parser.add_argument('--nce_t', default=0.1, type=float, help='temperature parameter for softmax') 
# parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
# parser.add_argument('--head', default='linear', type=str, choices=['linear', 'mlp', 'pad'])
# CRD

class CRDOpt():
    def __init__(self):
        self.s_dim = 128 
        self.t_dim = 128
        self.feat_dim = 128
        self.nce_k = 16384
        self.nce_t = 0.1
        self.nce_m = 0.5
        self.n_data = 50000
        self.head = "linear"

class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side

    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """
    def __init__(self, opt=CRDOpt(), s_dim=None, t_dim=None):
        super(CRDLoss, self).__init__()
        if s_dim is None:
            s_dim = opt.s_dim
        if t_dim is None:
            t_dim = opt.t_dim
        if opt.head == "linear":
            self.embed_s = Embed(s_dim, opt.feat_dim)
            self.embed_t = Embed(t_dim, opt.feat_dim)
        elif opt.head == "mlp":
            self.embed_s = Embed_mlp(s_dim, opt.feat_dim)
            self.embed_t = Embed_mlp(t_dim, opt.feat_dim)
        elif opt.head == "pad":
            self.embed_s = Embed_pad(s_dim, opt.feat_dim)
            self.embed_t = Embed_pad(t_dim, opt.feat_dim)
        else:
            raise NotImplementedError(f'head not supported: {opt.head}') 
        self.contrast = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.criterion_t = ContrastLoss(opt.n_data)
        self.criterion_s = ContrastLoss(opt.n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The contrastive loss
        """
        f_s = self.embed_s(f_s) # [bs, 128]
        f_t = self.embed_t(f_t) # [bs, 128]
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx) # [bs, 16385 (1 + nce_k), 1]
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss = s_loss + t_loss
        return loss


class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1 # 16384

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        # log_D1 = log[P_pos / (P_pos + nce_k/n_data)]
        P_pos = x.select(1, 0) # [4, 1], select positive
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()  # [4, 1]
        

        # loss for K negative pair
        # log_D0 = log[ (nce_k/n_data) / (P_neg + nce_k/n_data) ]
        P_neg = x.narrow(1, 1, m) # [4, 16384, 1]
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_() # [4, 16384, 1]
        
        
        # log_D1.sum(0): tensor([-39.1143], device='cuda:0', grad_fn=<SumBackward1>)
        # log_D0.view(-1, 1).shape: [65536, 1]
        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz
        return loss

class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Embed_pad(nn.Module):
    """Embed_padding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed_pad, self).__init__()
        self.l2norm = Normalize(2)
        self.dim_in = dim_in
        self.dim_out = dim_out

    def zero_pad(self, inputs, dim_in, dim_out):
        paddings = ((dim_out - dim_in) // 2, (dim_out - dim_in) // 2)
        outputs = torch.nn.functional.pad(inputs, paddings)
        return outputs

    def forward(self, x):
        x = x.view(x.shape[0], -1) # [bs, dim_in]; linear: [dim_in, dim_out]
        x = self.zero_pad(x, self.dim_in, self.dim_out) # [bs, dim_out]
        x = self.l2norm(x)
        return x


class Embed_mlp(nn.Module):
    """Embed_mlp module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed_mlp, self).__init__()
        self.linear1 = nn.Linear(dim_in, dim_in)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class ContrastMemory(nn.Module):
    """
    memory buffer that supplies large amount of negative samples.
    """
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5):
        super(ContrastMemory, self).__init__()
        """
        inputSize: opt.feat_dim, 128
        outputSize: n_data, number of training data, cifar100: 50000
        """
        self.K = K

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_v1', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, v1, v2, y, idx=None):
        """
        Refer to CRDLoss forward()
        Args:
            v1---f_s: the feature of student network, size [batch_size, s_dim]
            v2---f_t: the feature of teacher network, size [batch_size, t_dim]
            y ---idx: the indices of these positive samples in the dataset, size [batch_size]
            idx---contrast_idx: the indices of negative samples, size [batch_size, nce_k]
        """
        assert(idx is not None)
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()
        momentum = self.params[4].item()

        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0) 
        inputSize = self.memory_v1.size(1) 

        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach() # shape: [65540, 128]
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize) # shape: [4, 16385, 128]
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1)) # shape: [4, 16385, 1]
        out_v2 = torch.exp(torch.div(out_v2, T)) # shape: [4, 16385, 1]
        # sample
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        out_v1 = torch.exp(torch.div(out_v1, T)) # shape: [4, 16385, 1]
        
        # set Z if haven't been set yet
        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1)) # 72995.1
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2)) # 73917.7

        # compute out_v1, out_v2
        # contiguous(): returns itself if input tensor is already contiguous, otherwise it returns a new contiguous tensor by copying data.
        out_v1 = torch.div(out_v1, Z_v1).contiguous() 
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        # update memory
        with torch.no_grad():
            # memory_v1: [5000, 128]
            # l_pos = select positive index from memory_v1
            # updated_v1 = Norm[l_pos * momentum + v1 * (1-momentum)]
            # put updated_v1 back to memory_v1
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1)) # [4, 128], y.view(-1): tensor([ 2120, 11102, 33744,  1827], device='cuda:0')
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum)) # v1: [4,128]
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm) # [4,128]
            self.memory_v1.index_copy_(0, y, updated_v1) # y: tensor([ 2120, 11102, 33744,  1827], device='cuda:0')
            
            # ab_pos = select positive index from memory_v2
            # updated_v2 = Norm[ab_pos * momentum + v2 * (1-momentum)]
            # put updated_v2 back to memory_v2
            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        return out_v1, out_v2
    
class CIFAR10InstanceSample(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=16384, mode='exact', is_sample=True, percent=1.0): 
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
                
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 10
        if self.train:
            num_samples = len(self.data)
            label = self.targets
        else:
            num_samples = len(self.test_data)
            label = self.test_labels

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])
        

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
        
        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive) 
        self.cls_negative = np.asarray(self.cls_negative)
      
    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False 
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace) 
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))

            return img, target, index, sample_idx

class CIFAR100InstanceSample(torchvision.datasets.CIFAR100):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=16384, mode='exact', is_sample=True, percent=1.0): 
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
                
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        if self.train:
            num_samples = len(self.data)
            label = self.targets
        else:
            num_samples = len(self.test_data)
            label = self.test_labels

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])
        

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
        
        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive) 
        self.cls_negative = np.asarray(self.cls_negative)
      
    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False 
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace) 
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))

            return img, target, index, sample_idx

# class CIFAR100InstanceSample(torchvision.datasets.CIFAR100):
#     """
#     CIFAR100Instance+Sample Dataset
#     100 classes
#     Training data: 50000, 500 images per class
#     Testing data: 10000,  100 images per class
#     """
#     def __init__(self, root, train=True,
#                  transform=None, target_transform=None,
#                  download=False, k=4096, mode='exact', is_sample=True, percent=1.0, no_labels=False): 
#         super().__init__(root=root, train=train, download=download,
#                          transform=transform, target_transform=target_transform)
                
#         self.k = k
#         self.mode = mode
#         self.is_sample = is_sample

#         num_classes = 100
#         if self.train:
#             num_samples = len(self.data)
#             label = self.targets
#         else:
#             num_samples = len(self.test_data)
#             label = self.test_labels

#         self.cls_positive = [[] for i in range(num_classes)]
#         for i in range(num_samples):
#             self.cls_positive[label[i]].append(i)

#         self.cls_negative = [[] for i in range(num_classes + 1)]
#         for i in range(num_classes):
#             for j in range(num_classes):
#                 if j == i:
#                     continue
#                 self.cls_negative[i].extend(self.cls_positive[j])
#         self.cls_negative[-1] = np.arange(num_samples)
    

#         self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
#         self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
        
#         if 0 < percent < 1:
#             n = int(len(self.cls_negative[0]) * percent)
#             self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
#                                  for i in range(num_classes)]

#         self.cls_positive = np.asarray(self.cls_positive) # shape: (100, 500)
#         self.cls_negative = np.asarray(self.cls_negative) # shape: (100, 49500)
#         self.no_labels = no_labels
     

#     def __getitem__(self, index):
#         if self.train:
#             img, target = self.data[index], self.targets[index]
#         else:
#             img, target = self.test_data[index], self.test_labels[index]

#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = Image.fromarray(img)

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         if not self.is_sample:
#             return img, target, index
#         else:
#             # sample contrastive examples
#             if self.mode == 'exact':
#                 pos_idx = index
#             elif self.mode == 'relax':
#                 pos_idx = np.random.choice(self.cls_positive[target], 1)
#                 pos_idx = pos_idx[0]
#             else:
#                 raise NotImplementedError(self.mode)
#             replace = True if self.k > len(self.cls_negative[target]) else False
#             if self.no_labels:
#                 neg_idx = np.random.choice(self.cls_negative[-1], self.k, replace=replace) 
#             else:
#                 neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace) 
#             sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx)) 
#             return img, target, index, sample_idx

