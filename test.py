from config import params
from torch.utils.data import DataLoader
from torch import nn, optim
import os
from model import c3d
from model import ren
from dataset.data import ReverseDataSet
import random
import numpy as np
import torch
from tqdm import tqdm
from torch.autograd import Variable

save_path = "train_classify"

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
device_ids=[4]

params['batch_size'] = 8
params['num_workers'] = 4

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.pwd = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test(test_loader, model, criterion):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0
    top1 = AverageMeter()

    for step, (inputs, labels) in enumerate(test_loader):
        labels = labels.cuda()
        inputs = inputs.cuda()
        outputs = []
        for clip in inputs:
            clip = clip.cuda()
            out = model(clip)
            out = torch.mean(out, dim=0)

            outputs.append(out)
        outputs = torch.stack(outputs)

        loss = criterion(outputs, labels)

        total_loss += loss.item()

        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(labels == pts).item()
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
        print(str(step), len(test_loader))
        print(correct)

    avg_loss = total_loss / len(test_loader)
    avg_acc = correct / len(test_loader.dataset)
    print('[TEST] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss

def load_pretrained_weights(ckpt_path):

    adjusted_weights = {}
    pretrained_weights = torch.load(ckpt_path, map_location='cpu')
    for name, param in pretrained_weights.items():
        print(name)
        if "module" in name:
            name = name[name.find('.')+1:]
        adjusted_weights[name] = param
    return adjusted_weights


def test_model(model, pretrain_path):
    #print(pretrain_path)
    pretrain_weight = load_pretrained_weights(pretrain_path)
    model.load_state_dict(pretrain_weight, strict=True)

    test_dataset = ReverseDataSet(params['dataset'], mode="test")
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False,
                             num_workers=params['num_workers'])
    if len(device_ids)>1:
        print(torch.cuda.device_count())
        model = nn.DataParallel(model)
    # model = nn.DataParallel(model, device_ids=[4,5,6,7])

    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    test(test_loader, model, criterion)


if __name__ == '__main__':
    seed = 632
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = c3d.C3D(with_classifier=True, num_classes=101)

    pretrain_path ="/home/fb/project/AB_reverse" \
                   "/finetune_model_UCF-101/_12-06-09-36" \
                   "/best_loss_model_147.pth.tar"

    test_model(model, pretrain_path)


