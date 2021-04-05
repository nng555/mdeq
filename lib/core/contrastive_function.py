# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import torch

from core.cls_evaluate import accuracy
from core.contrastive_criterion import InfoNCE


logger = logging.getLogger(__name__)

def make_contrastive_batch(x):
    """
    turn list of tuples n x (x0, x1) into a batch of n x (n+1)/2 size
    """
    x0s = [ex[0] for ex in x]
    x1s = [ex[1] for ex in x]
    x0t = []
    x1t = []
    for x0i in range(len(x0s)):
        for x1i in range(x0i, len(x1s)):
            x0t.append(x0s[x0i])
            x1t.append(x1s[x1i])

    x0t = torch.stack(x0t)
    x1t = torch.stack(x1t)

    return x0t, x1t


def train(config, train_loader, model, info_nce, criterion, optimizer, lr_scheduler, epoch,
          output_dir, tb_log_dir):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()



    # switch to train mode
    model.train()

    end = time.time()
    total_batch_num = len(train_loader)
    effec_batch_num = int(config.PERCENT * total_batch_num)

    for i, (imgs, t) in enumerate(train_loader):
        # train on partial training data
        if i >= effec_batch_num:
            break

        # measure data loading time
        data_time.update(time.time() - end)
        #target = target - 1 # Specific for imagenet

        # stack to form 2N size batch
        imgs = torch.cat(imgs, dim=0)

        features = model(imgs, train_step=(lr_scheduler._step_count-1))

        logits, labels = info_nce(features)
        loss = criterion(logits, labels)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        if config['TRAIN']['CLIP'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['TRAIN']['CLIP'])
        optimizer.step()
        if config.TRAIN.LR_SCHEDULER != 'step':
            lr_scheduler.step()

        # measure accuracy and record loss
        losses.update(loss.item(), features.size(0))

        #prec1, prec5 = accuracy(output, target, topk=topk)

        #top1.update(prec1[0], input.size(0))
        #top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                      epoch, i, effec_batch_num, batch_time=batch_time,
                      speed=features.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)


def validate(config, val_loader, model, criterion, output_dir, tb_log_dir,
             writer_dict=None, topk=(1,5)):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # compute output
            output = model(input,
                           train_step=-1,       # Evaluate using MDEQ (even when pre-training)
                           writer=None if writer_dict is None else writer_dict['writer'])
            target = target.cuda(non_blocking=True)

            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            prec1, prec5 = accuracy(output, target, topk=topk)
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            break

        msg = 'Test: Time {batch_time.avg:.3f}\t' \
              'Loss {loss.avg:.4f}\t' \
              'Error@1 {error1:.3f}\t' \
              'Error@5 {error5:.3f}\t' \
              'Accuracy@1 {top1.avg:.3f}\t' \
              'Accuracy@5 {top5.avg:.3f}\t'.format(
                  batch_time=batch_time, loss=losses, top1=top1, top5=top5,
                  error1=100-top1.avg, error5=100-top5.avg)
        logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_top1', top1.avg, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return top1.avg

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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

"""
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
"""
