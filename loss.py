import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable
import utils_ent

class CenterTripletLoss(nn.Module):
    
    def __init__(self, batch_size, margin=0.3):
        super(CenterTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, labels):
        label_uni = labels.unique()
        targets = torch.cat([label_uni,label_uni])
        label_num = len(label_uni)
        feat = feats.chunk(label_num*2, 0)
        center = []
        for i in range(label_num*2):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))
        inputs = torch.cat(center)

        n = inputs.size(0)
        
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  
        
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct



class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


        
def pdist_torch(emb1, emb2):
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    return dist_mtx


class CMCD(nn.Module):
    def __init__(self,args, batch_size, margin=0.3):
        super(EntLoss, self).__init__()
        self.args = args

    def forward(self, inputs, targets):
        feat1, feat2 = torch.chunk(inputs, 2, dim=0)
        probs1 = torch.nn.functional.softmax(feat1, dim=-1)
        probs2 = torch.nn.functional.softmax(feat2, dim=-1)
        loss = dict()
        loss_KL = 0.5 * (KL(probs1, probs2, self.args) + KL(probs2, probs1, self.args))

        sharpened_probs1 = torch.nn.functional.softmax(feat1 / self.args.tau, dim=-1)
        sharpened_probs2 = torch.nn.functional.softmax(feat2 / self.args.tau, dim=-1)
        loss_EH = 0.5 * (EH(sharpened_probs1, self.args) + EH(sharpened_probs2, self.args))

        loss_HE = 0.5 * (HE(sharpened_probs1, self.args) + HE(sharpened_probs2, self.args))

        loss = loss_KL + ((1 + 0) * loss_EH - 1 * loss_HE)
        return loss

def KL(probs1, probs2, args):
    kl = (probs1 * (probs1 + args.EPS).log() - probs1 * (probs2 + args.EPS).log()).sum(dim=1)
    kl = kl.mean()
    return kl

def CE(probs1, probs2, args):
    ce = - (probs1 * (probs2 + args.EPS).log()).sum(dim=1)
    ce = ce.mean()
    return ce

def HE(probs, args):
    mean = probs.mean(dim=0)
    ent  = - (mean * (mean + utils_ent.get_world_size() * args.EPS).log()).sum()
    return ent

def EH(probs, args):
    ent = - (probs * (probs + args.EPS).log()).sum(dim=1)
    mean = ent.mean()
    return mean