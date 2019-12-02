from config import params
from torch import nn, optim
import os
from model import c3d
from model import sscn
from dataset.data import ReverseDataSet
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import random
import numpy as np
from tensorboardX import SummaryWriter

save_path = params['save_path_base'] + 'ft_classify_' + params['data']

os.environ["CUDA_VISIBLE_DEVISES"] = "6,7"

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


def accuracy(output, target, topk=(1, )):

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k =correct[:k].view(-1).float().sum(0)
        res.append((correct_k.mul_(100.0 / batch_size)))
    return res


def train(train_loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()

    for step, (input, label) in enumerate(train_loader):
        data_time.update(time.time() - end)

        label = label.cuda()
        input = input.cuda()

        output = model(input)

        loss = criterion(output, label)
        prec1, prec5 = accuracy(output.data, label, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if (step + 1) % params['display'] == 0:
            print('-------------------------------------------------')
            for param in optimizer.param_groups:
                print("lr:", param['lr'])

            p_str = "Epoch:[{0}]/[{1}/{2}]".format(epoch, step+1, len(train_loader))
            print(p_str)

            p_str = "data_time:{data_time:.3f}, batch_time:{batch_time:.3f}".format(data_time=data_time.val,
                                                                                    batch_time=batch_time.val)
            print(p_str)

            p_str = "loss:{loss:.5f}".format(loss=losses.val)
            print(p_str)

            total_step = (epoch-1) * len(train_loader) + step + 1
            writer.add_scalar('train/loss', losses.avg, total_step)
            writer.add_scalar('train/acc', top1.avg, total_step)

            p_str = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                top1_acc=top1.avg,
                top5_acc=top5.avg
            )
            print(p_str)


def validation(val_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    end = time.time()
    total_loss = 0.0

    with torch.no_grad():
        for step, (inputs, labels) in enumerate(val_loader):
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.update(loss.item(), inputs.size(0))

            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))

            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            total_loss += loss.item()

            if (step + 1) % params['display'] == 0:
                print('-----------------------------validation-------------------')
                p_str = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(val_loader))
                print(p_str)

                p_str = 'data_time:{data_time:.3f},batch time:{batch_time:.3f}'.format(data_time=data_time.val,
                                                                                       batch_time=batch_time.val);
                print(p_str)

                p_str = 'loss:{loss:.5f}'.format(loss=losses.avg)
                print(p_str)

                p_str = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                    top1_acc=top1.avg,
                    top5_acc=top5.avg)
                print(p_str)

    avg_loss = total_loss / len(val_loader)
    return avg_loss, top1.avg


def main():
    base = c3d.C3D(with_classifier=False)
    model = sscn.sscn(base, with_classifier=True, num_classes=15)

    start_epoch = 1
    # pretrain_weight = loadcontinur_weights(pretrain_path)

    # model.load_state_dict(pretrain_weight, strict=False)
    # train
    train_dataset = ReverseDataSet(params['dataset'], mode="train")
    if params['data'] == 'UCF-101':
        val_size = 800
    elif params['data'] == 'hmdb':
        val_size = 400
    train_dataset, val_dataset = random_split(train_dataset, (len(train_dataset) - val_size, val_size))

    print("num_works:{:d}".format(params['num_workers']))
    print("batch_size:{:d}".format(params['batch_size']))
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,
                              num_workers=params['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=True,
                            num_workers=params['num_workers'])

    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=params['momentum'],
                          weight_decay=params['weight_decay'])
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-5, patience=20, factor=0.1)

    #pretrain_model = pretrain_path.split('/')[-1].split('.')[0] + 'pth'

    model_save_dir = os.path.join(save_path, '_' + time.strftime('%m-%d-%H-%M'))
    writer = SummaryWriter(model_save_dir)

    for data in train_loader:
        clip, label = data
        writer.add_video('train/clips', clip, 0, fps=8)
        writer.add_text('train/idx', str(label.tolist()), 0)
        clip = clip.cuda()
        #writer.add_graph(model, (clip, clip));
        break
    for name, param in model.named_parameters():
        writer.add_histogram('params/{}'.format(name), param, 0)

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    prev_best_val_loss = float('inf')
    prev_best_loss_model_path = None
    prev_best_acc_model_path = None
    best_acc = 0
    best_epoch = 0
    for epoch in tqdm(range(start_epoch, start_epoch + params['epoch_num'])):
        train(train_loader, model, criterion, optimizer, epoch, writer)
        val_loss, top1_avg = validation(val_loader, model, criterion, optimizer, epoch)
        if top1_avg >= best_acc:
            best_acc = top1_avg
            best_epoch = epoch
            model_path = os.path.join(model_save_dir, 'best_acc_model_{}.pth.tar'.format(epoch))
            torch.save(base.state_dict(), model_path)

            prev_best_acc_model_path = model_path
        if val_loss < prev_best_val_loss:
            model_path = os.path.join(model_save_dir, 'best_loss_model_{}.pth.tar'.format(epoch))
            torch.save(base.state_dict(), model_path)
            prev_best_val_loss = val_loss

            prev_best_loss_model_path = model_path
        scheduler.step(val_loss)
        if epoch % 20 == 0:
            checkpoints = os.path.join(model_save_dir, str(epoch) + ".pth.tar")
            torch.save(base.state_dict(), checkpoints)
            print("save_to:", checkpoints)
    print("best is :", best_acc, best_epoch)


if __name__ == '__main__':
    seed = 632
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    main()