import torch
import torch.nn as nn
from torch.nn import functional as F

class NTXent(nn.Module):
    def __init__(self, tau, normalize):
        super(InfoNCE, self).__init__()
        self.tau = tau
        self.normalize = normalize

    def forward(self, x0, x1, **kwargs):
        bsize = x0.shape[0]
        target = torch.arange(bsize).cuda()
        eye_mask = torch.eye(bsize).cuda() * 1e9
        if self.normalize:
            x0 = F.normalize(x0, p=2, dim=1)
            x1 = F.normalize(x1, p=2, dim=1)
        logits00 = x0 @ x0.t() / self.tau - eye_mask
        logits11 = x1 @ x1.t() / self.tau - eye_mask
        logits01 = x0 @ x1.t() / self.tau
        logits10 = x1 @ x0.t() / self.tau
        xent0 = F.cross_entropy(torch.cat([logits01, logits00], dim=1), target)
        xent1 = F.cross_entropy(torch.cat([logits10, logits11], dim=1), target)
        return (xent0 + xent1) / 2



class InfoNCE(nn.Module):

    def __init__(self, tau, normalize, nviews):
        super(InfoNCE, self).__init__()
        self.tau = tau
        self.normalize = normalize
        self.nviews = nviews

    def forward(self, features, **kwargs):
        bsize = features.shape[0] / 2
        labels = torch.cat([torch.arange(bsize) for i in range(self.nviews)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        if self.normalize:
            features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool)#.to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long)

        logits = logits / self.tau
        return logits, labels
