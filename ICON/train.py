import os
import sys
import math
import time
import pickle
import random
import logging
import argparse
import warnings

import numpy as np
from sklearn import metrics

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import ICON


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def time_transform(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f'{int(h):02d}:{int(m):02d}:{int(s):02d}'


class IEMOCAP(Dataset):
    def __init__(self, path, mini=False):
        self.query, self.own_history, self.other_history, self.own_history_mask, self.other_history_mask, self.label = pickle.load(open(path, 'rb'))
        if mini:
            self.query = self.query[:int(self.query.shape[0] * 0.1)]
            self.own_history = self.own_history[:int(self.own_history.shape[0] * 0.1)]
            self.other_history = self.other_history[:int(self.other_history.shape[0] * 0.1)]
            self.own_history_mask = self.own_history_mask[:int(self.own_history_mask.shape[0] * 0.1)]
            self.other_history_mask = self.other_history_mask[:int(self.other_history_mask.shape[0] * 0.1)]
            self.label = self.label[:int(self.label.shape[0] * 0.1)]
    
    def __len__(self):
        return self.query.shape[0]
    
    def __getitem__(self, idx):
        return self.query[idx], self.own_history[idx], self.other_history[idx], self.own_history_mask[idx], self.other_history_mask[idx], self.label[idx]


def display_info(args, model):
    n_trainable_params, n_nontrainable_params = 0, 0
    for p in model.parameters():
        n_params = torch.prod(torch.tensor(p.shape))
        if p.requires_grad:
            n_trainable_params += n_params
        else:
            n_nontrainable_params += n_params
    logger.info(f'n_trainable_params: {n_trainable_params}, n_nontrainable_params: {n_nontrainable_params}')
    if args.device.type == 'cuda':
        logger.info(f'cuda memory allocated: {torch.cuda.memory_allocated(device=args.device.index)}')
    logger.info('training arguments:')
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')


def plot_training_curve(args, record, title, xlabel, ylabel):
    epoch = len(record['train']) if 'train' in record else len(record['dev'])
    x_1 = range(1, epoch + 1)
    plt.cla()
    plt.figure(figsize=(6,4))
    if 'train' in record:
        plt.plot(x_1, record['train'], c='tab:red', label='train')
    if 'dev' in record:
        plt.plot(x_1, record['dev'], c='tab:cyan', label='dev')
    if 'test' in record:
        plt.plot(x_1, record['test'], c='tab:orange', label='test')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Training curve of {title}')
    plt.legend()
    plt.savefig(f'{args.store_path}/{title}.png', dpi=300)
    plt.show()


def train_epoch(args, model, dataloader, optimizer, scheduler, criterion, mode='train'):
    loss_epoch, preds, labels = [], [], []
    if mode == 'train':
        model.train()
    else:
        model.eval()
    for batch in dataloader:
        query, own_history, other_history, own_history_mask, other_history_mask, label = [_.to(args.device) for _ in batch]
        if mode == 'train':
            optimizer.zero_grad()
        output = model(query, own_history, other_history, own_history_mask, other_history_mask)
        loss = criterion(output, label)
        if mode == 'train':
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
        loss_epoch.append(loss.item())
        preds += torch.argmax(output, -1).detach().cpu().numpy().tolist()
        labels += label.detach().cpu().numpy().tolist()
    loss_epoch = sum(loss_epoch) / len(loss_epoch)
    acc_epoch = metrics.accuracy_score(labels, preds)
    return loss_epoch, acc_epoch


def test_from_file(args, dataloader):
    model = torch.load(f'{args.store_path}/model.pth')
    preds, labels = [], []
    model.eval()
    for batch in dataloader:
        query, own_history, other_history, own_history_mask, other_history_mask, label = [_.to(args.device) for _ in batch]
        with torch.no_grad():
            output = model(query, own_history, other_history, own_history_mask, other_history_mask)
        preds += torch.argmax(output, -1).detach().cpu().numpy().tolist()
        labels += label.detach().cpu().numpy().tolist()
    
    acc = metrics.accuracy_score(labels, preds)
    cmat = metrics.confusion_matrix(labels, preds, labels=range(args.num_class))
    fscore = metrics.classification_report(labels, preds, digits=5)
    f1 = metrics.f1_score(labels, preds, labels=range(args.num_class), average='macro')
    logger.info(f'Best model metrics:')
    logger.info(f'accuracy: {acc:.5f}')
    logger.info(f'confusion_matrix: \n{cmat}')
    logger.info(f'classification report: \n{fscore}')
    if args.record_path:
        with open(args.record_path, 'a') as record_file:
            record_file.write(f'> {args.modal} batch_size={args.batch_size}  acc: {acc:.5f} f1: {f1:.5f}\n')


def main():
    # Command Parameters and Config
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--feature_path', default='./feature', type=str)
    parser.add_argument('--store_path', default='static_dict', type=str)
    parser.add_argument('--record_path', default=None, type=str)
    parser.add_argument('--modal', default='all', type=str, help='text, video, audio, text_video, text_audio, audio_video, all')
    parser.add_argument('--device', default=7, type=str)
    parser.add_argument('--seed', default=1227, type=int)
    parser.add_argument('--mini', default=False, type=bool)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=400, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--l2reg', default=1e-3, type=float)
    parser.add_argument('--annealing_max', default=100, type=int)
    parser.add_argument('--clip', default=40., type=float)
    parser.add_argument('--embedding_size', default=100, type=int)
    parser.add_argument('--hop_size', default=3, type=int)
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--num_class', default=6, type=int)
    args = parser.parse_args()
    if not os.path.exists(args.store_path):
        os.mkdir(args.store_path)
    args.device = torch.device('cpu') if args.device is None else torch.device(f'cuda:{args.device}')
    log_file = f'{args.store_path}/{time.strftime("%y%m%d-%H%M", time.localtime())}.log'
    logger.addHandler(logging.FileHandler(log_file))
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    args.feature_size = 712 if args.modal == 'all' else \
                        100 * (('text' in args.modal) + ('audio' in args.modal)) + 512 * ('video' in args.modal)
    
    # Model, Optimizer, Scheduler, Criterion
    model = ICON(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2reg)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.annealing_max)
    criterion = nn.CrossEntropyLoss()

    # Display Hyper-Parameters and Model Info, Init Model Parameters
    display_info(args, model)
    
    # Dataset
    train_dataset = IEMOCAP(f'{args.feature_path}/{args.modal}/train.data', args.mini)
    val_dataset = IEMOCAP(f'{args.feature_path}/{args.modal}/val.data', args.mini)
    test_dataset = IEMOCAP(f'{args.feature_path}/{args.modal}/test.data', args.mini)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Training
    record = {
        'loss': { 'train': [], 'dev': [], 'test': [] },
        'acc': { 'train': [], 'dev': [], 'test': [] }
    }
    def add_record(x, train_var, dev_var, test_var=None):
        record[x]['train'].append(train_var)
        record[x]['dev'].append(dev_var)
        if test_var is not None:
            record[x]['test'].append(test_var)
    
    epoch = 0
    train_stime = time.time()
    max_val_acc = float('-inf')
    while epoch < args.epoch:
        epoch_stime = time.time()
        train_loss, train_acc = train_epoch(args, model, train_dataloader, optimizer, scheduler, criterion, mode='train')
        val_loss, val_acc = train_epoch(args, model, val_dataloader, optimizer, scheduler, criterion, mode='val')
        test_loss, test_acc = train_epoch(args, model, test_dataloader, optimizer, scheduler, criterion, mode='test')
        add_record('loss', train_loss, val_loss, test_loss)
        add_record('acc', train_acc, val_acc, test_acc)
        if epoch == 0 or val_acc > max_val_acc:
            torch.save(model, f'{args.store_path}/model.pth')
        epoch += 1
        if val_acc > max_val_acc:
            max_val_acc = val_acc
        logger.info(f'[ Epoch {epoch:03d}/{args.epoch:03d} ]( {time_transform(time.time() - epoch_stime)} ) train loss: {train_loss:.5f} train acc: {train_acc:.5f} val loss: {val_loss:.5f} val acc: {val_acc:.5f} test loss: {test_loss:.5f} test acc: {test_acc:.5f}')
        if args.mini or math.isnan(train_loss):
            break
    logger.info(f'Finish train. ( {time_transform(time.time() - train_stime)} )')

    # Testing
    plot_training_curve(args, record['loss'], title='loss', xlabel='epoch', ylabel='loss')
    plot_training_curve(args, record['acc'], title='accuracy', xlabel='epoch', ylabel='accuracy')
    test_from_file(args, test_dataloader)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()

