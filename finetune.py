import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from torch.autograd import Variable
import time
from config import params
from dataset.data import ReverseDataSet
from model import ren
from model import c3d
from tensorboardX import SummaryWriter
from tqdm import tqdm
from collections import OrderedDict


save_path = params['save_path_base'] + 'finetune_model_' + params['data']

os.environ["CUDA_VISIBLE_DEVISES"] = "4,5,6,7"
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

params['epoch'] = 100

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



def initialize_model(model, pretrain_path, num_classes):

    state_dict = torch.load(pretrain_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc6.parameters():
        param.requires_grad = True

    channel_in = model.fc6.in_features
    model.fc6 = nn.Linear(channel_in, num_classes)
    print(model)

    return model



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

        label = Variable(label).to(device)
        input = Variable(input).to(device)

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

            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)

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
    pretrain_path = "/home/fb/project/AB_reverse" \
                    "/ft_classify_UCF-101/_12-02-16-36" \
                    "/best_acc_model_180.pth.tar"
    base = c3d.C3D(with_classifier=False)
    model = ren.REN(base, with_classifier=True)
    model_ft = initialize_model(model, pretrain_path, num_classes=101)

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

    model_ft = nn.DataParallel(model_ft, device_ids=[4,5,6,7])  #multi-gpu
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer_ft = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001     # ?????????
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, 'min', min_lr=1e-5, patience=20, factor=0.5)


    model_save_dir = os.path.join(save_path, '_' + time.strftime('%m-%d-%H-%M'))
    writer = SummaryWriter(model_save_dir)

    for data in train_loader:
        clip, label = data
        writer.add_video('train/clips', clip, 0, fps=8)
        writer.add_text('train/idx', str(label.tolist()), 0)
        clip = clip.to(device)
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
        train(train_loader, model_ft, criterion, optimizer_ft, epoch, writer)
        val_loss, top1_avg = validation(val_loader, model, criterion, optimizer_ft, epoch)
        if top1_avg >= best_acc:
            best_acc = top1_avg
            best_epoch = epoch
            model_path = os.path.join(model_save_dir, 'best_acc_model_{}.pth.tar'.format(epoch))
            torch.save(model.state_dict(), model_path)

            prev_best_acc_model_path = model_path
        if val_loss < prev_best_val_loss:
            model_path = os.path.join(model_save_dir, 'best_loss_model_{}.pth.tar'.format(epoch))
            torch.save(model.state_dict(), model_path)
            prev_best_val_loss = val_loss

            prev_best_loss_model_path = model_path
        scheduler.step(val_loss)
        if epoch % 20 == 0:
            checkpoints = os.path.join(model_save_dir, str(epoch) + ".pth.tar")
            torch.save(model.state_dict(), checkpoints)
            print("save_to:", checkpoints)
    print("best is :", best_acc, best_epoch)


if __name__ == '__main__':
    main()