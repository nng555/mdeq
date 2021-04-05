import torch
import torch.nn as nn
import torch.optim as optim

def get_data(model, loader, output_size, device):
    """ encodes the whole dataset into embeddings """
    xs = torch.empty(
        len(loader), loader.batch_size, output_size, dtype=torch.float32, device=device
    )
    ys = torch.empty(len(loader), loader.batch_size, dtype=torch.long, device=device)
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.cuda()
            xs[i] = model(x).to(device)
            ys[i] = y.to(device)
    xs = xs.view(-1, output_size)
    ys = ys.view(-1)
    return xs, ys

def eval_knn(x_train, y_train, x_test, y_test, k=5):
    """ k-nearest neighbors classifier accuracy """
    d = torch.cdist(x_test, x_train)
    topk = torch.topk(d, k=k, dim=1, largest=False)
    labels = y_train[topk.indices]
    pred = torch.empty_like(y_test)
    for i in range(len(labels)):
        x = labels[i].unique(return_counts=True)
        pred[i] = x[0][x[1].argmax()]

    acc = (pred == y_test).float().mean().cpu().item()
    del d, topk, labels, pred
    return acc

def eval_sgd(x_train, y_train, x_test, y_test, topk=[1, 5], epoch=500):
    """ linear classifier accuracy (sgd) """
    lr_start, lr_end = 1e-2, 1e-6
    gamma = (lr_end / lr_start) ** (1 / epoch)
    output_size = x_train.shape[1]
    num_class = y_train.max().item() + 1
    clf = nn.Linear(output_size, num_class)
    clf.cuda()
    clf.train()
    optimizer = optim.Adam(clf.parameters(), lr=lr_start, weight_decay=5e-6)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epoch):
        perm = torch.randperm(len(x_train)).view(-1, 1000)
        for idx in perm:
            optimizer.zero_grad()
            criterion(clf(x_train[idx]), y_train[idx]).backward()
            optimizer.step()
        scheduler.step()

    clf.eval()
    with torch.no_grad():
        y_pred = clf(x_test)
    pred_top = y_pred.topk(max(topk), 1, largest=True, sorted=True).indices
    acc = {
        t: (pred_top[:, :t] == y_test[..., None]).float().sum(1).mean().cpu().item()
        for t in topk
    }
    del clf
    return acc




"""
class ContrastiveBaseMethod(nn.Module):

    def __init__(self, encoder, cfg):
        super().__init__()
        self.model = encoder
        self.emb_size = cfg['EMB_SIZE']
        self.out_size = cfg['REPR_SIZE']
        self.head = build_head(cfg)
        self.knn = cfg.knn
        self.num_pairs = cfg['NUM_SAMPLES'] * (cfg['NUM_SAMPLES'] - 1) // 2
        self.eval_head = cfg['EVAL_HEAD']


    def forward(self, samples):
        raise NotImplementedError

    def get_acc(self, ds_clf, ds_test):
        self.eval()
        if self.eval_head:
            model = lambda x: self.head(self.model(x))
            out_size = self.emb_size
        else:
            model, out_size = self.model, self.out_size

        x_train, y_train = get_data(model, ds_clf, out_size, "cuda")
        x_test, y_test = get_data(model, ds_test, out_size, "cuda")

        acc_knn = eval_knn(x_train, y_train, x_test, y_test, self.knn)
        acc_linear = eval_sgd(x_train, y_train, x_test, y_test)
        del x_train, y_train, x_test, y_test
        self.train()
        return acc_knn, acc_linear

    def step(self, progress):
        pass
"""

