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
import pandas as pd
from sklearn import metrics

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from model import Model, BiModel


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def time_transform(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f'{int(h):02d}:{int(m):02d}:{int(s):02d}'


class IEMOCAP(Dataset):
    def __init__(self, path, mini=False):
        self.text, self.audio, self.visual, self.speaker, self.mask, self.label = pickle.load(open(path, 'rb'))
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.text[idx]), \
            torch.FloatTensor(self.audio[idx]), \
            torch.FloatTensor(self.visual[idx]), \
            torch.FloatTensor(self.speaker[idx]), \
            torch.FloatTensor(self.mask[idx]), \
            torch.LongTensor(self.label[idx])
    
    def collate_fn(self, data):
        data = pd.DataFrame(data)
        ret = []
        for i in data:
            if i < 4:
                ret.append(torch.FloatTensor(pad_sequence(data[i].tolist())))
            elif i == 4:
                ret.append(torch.FloatTensor(pad_sequence(data[i].tolist(), True)))
            else:
                ret.append(torch.LongTensor(pad_sequence(data[i].tolist(), True)))
        return ret


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
    epoch = len(record['train']) if 'train' in record else len(record['val'])
    x_1 = range(1, epoch + 1)
    plt.cla()
    plt.figure(figsize=(6,4))
    if 'train' in record:
        plt.plot(x_1, record['train'], c='tab:red', label='train')
    if 'val' in record:
        plt.plot(x_1, record['val'], c='tab:cyan', label='val')
    if 'test' in record:
        plt.plot(x_1, record['test'], c='tab:orange', label='test')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Training curve of {title}')
    plt.legend()
    plt.savefig(f'{args.store_path}/{title}.png', dpi=300)
    plt.show()


def fusion_features(args, text, audio, visual):
    if args.modal == 'text':
        return text
    elif args.modal == 'audio':
        return audio
    elif args.modal == 'visual':
        return visual
    elif args.modal == 'text_audio':
        return torch.cat((text, audio), dim=-1)
    elif args.modal == 'text_visual':
        return torch.cat((text, visual), dim=-1)
    elif args.modal == 'audio_visual':
        return torch.cat((audio, visual), dim=-1)
    else:
        return torch.cat((text, audio, visual), dim=-1)


def train_epoch(args, model, dataloader, optimizer, scheduler, criterion, mode='train'):
    loss_epoch, preds, labels, masks = [], [], [], []
    if mode == 'train':
        model.train()
    else:
        model.eval()
    for batch in dataloader:
        text, audio, visual, speaker, mask, label = [_.to(args.device) for _ in batch]
        if mode == 'train':
            optimizer.zero_grad()
        output = model(fusion_features(args, text, audio, visual), speaker, mask)
        label = label.view(-1)
        loss = criterion(output * mask.view(-1, 1), label)
        if mode == 'train':
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
        loss_epoch.append(loss.item())
        preds += torch.argmax(output, -1).detach().cpu().numpy().tolist()
        labels += label.detach().cpu().numpy().tolist()
        masks += mask.view(-1).cpu().numpy().tolist()
    loss_epoch = sum(loss_epoch) / len(loss_epoch)
    acc_epoch = metrics.accuracy_score(labels, preds, sample_weight=masks)
    f1_epoch = metrics.f1_score(labels, preds, sample_weight=masks, labels=range(args.num_class), average='weighted')
    return loss_epoch, acc_epoch, f1_epoch


def test_from_file(args, dataloader):
    model = torch.load(f'{args.store_path}/model.pth')
    preds, labels, masks = [], [], []
    model.eval()
    for batch in dataloader:
        text, audio, visual, speaker, mask, label = [_.to(args.device) for _ in batch]
        with torch.no_grad():
            output = model(fusion_features(args, text, audio, visual), speaker, mask)
        preds += torch.argmax(output, -1).detach().cpu().numpy().tolist()
        labels += label.view(-1).detach().cpu().numpy().tolist()
        masks += mask.view(-1).cpu().numpy().tolist()
    
    acc = metrics.accuracy_score(labels, preds, sample_weight=masks)
    cmat = metrics.confusion_matrix(labels, preds, sample_weight=masks, labels=range(args.num_class))
    fscore = metrics.classification_report(labels, preds, sample_weight=masks, digits=5)
    f1 = metrics.f1_score(labels, preds, sample_weight=masks, labels=range(args.num_class), average='weighted')
    logger.info(f'Best model metrics:')
    logger.info(f'accuracy: {acc:.5f}')
    logger.info(f'confusion_matrix: \n{cmat}')
    logger.info(f'classification report: \n{fscore}')
    if args.record_path:
        with open(args.record_path, 'a') as record_file:
            record_file.write(f'> modal: {args.modal.replace("_", ", "):<15}  acc: {acc:.5f} f1: {f1:.5f}\n')


def main():
    # Command Parameters and Config
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--feature_path', default='./feature', type=str)
    parser.add_argument('--store_path', default='static_dict', type=str)
    parser.add_argument('--record_path', default=None, type=str)
    parser.add_argument('--modal', default='all', type=str, help='text, audio, visual, text_audio, text_visual, audio_visual, all')
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--val_ratio', default=0.0, type=float)
    parser.add_argument('--epoch', default=60, type=int)
    parser.add_argument('--batch_size', default=30, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--l2reg', default=1e-5, type=float)
    parser.add_argument('--annealing_max', default=100, type=int)
    parser.add_argument('--clip', default=40., type=float)
    parser.add_argument('--global_size', default=500, type=int)
    parser.add_argument('--party_size', default=500, type=int)
    parser.add_argument('--emotion_size', default=300, type=int)
    parser.add_argument('--attn', action='store_true')
    parser.add_argument('--bilateral', action='store_true')
    parser.add_argument('--listener_state', action='store_true')
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--num_class', default=6, type=int)
    args = parser.parse_args()
    if not os.path.exists(args.store_path):
        os.mkdir(args.store_path)
    if args.device is not None:
        torch.cuda.set_device(int(args.device))
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
                        100 * (('text' in args.modal) + ('audio' in args.modal)) + 512 * ('visual' in args.modal)
    
    # Model, Optimizer, Scheduler, Criterion
    if args.bilateral:
        model = BiModel(args).to(args.device)
    else:
        model = Model(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2reg)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.annealing_max)
    criterion = nn.CrossEntropyLoss()

    # Display Hyper-Parameters and Model Info, Init Model Parameters
    display_info(args, model)
    
    # Dataset
    train_dataset = IEMOCAP(f'{args.feature_path}/train.data')
    test_dataset = IEMOCAP(f'{args.feature_path}/test.data')
    logger.info(f'train dataset size: {len(train_dataset)}')
    logger.info(f'test dataset size: {len(test_dataset)}')

    # split = int(len(train_dataset) * args.val_ratio)
    # indices = list(range(len(train_dataset)))
    # random.shuffle(indices)
    # train_sampler = SubsetRandomSampler(indices[split:])
    # val_sampler = SubsetRandomSampler(indices[:split])

    train_dataloader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                # sampler=train_sampler,
                                collate_fn=train_dataset.collate_fn)
    # val_dataloader = DataLoader(train_dataset,
    #                             batch_size=args.batch_size,
    #                             sampler=val_sampler,
    #                             collate_fn=train_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset,
                                batch_size=args.batch_size,
                                collate_fn=test_dataset.collate_fn)

    # Training
    record = {
        'loss': { 'train': [], 'test': [] },
        'acc': { 'train': [], 'test': [] },
        'f1': { 'train': [], 'test': [] }
    }
    def add_record(x, train_var, val_var=None, test_var=None):
        record[x]['train'].append(train_var)
        if val_var is not None:
            record[x]['val'].append(val_var)
        if test_var is not None:
            record[x]['test'].append(test_var)
    
    epoch = 0
    train_stime = time.time()
    max_test_acc = float('-inf')
    while epoch < args.epoch:
        epoch_stime = time.time()
        train_loss, train_acc, train_f1 = train_epoch(args, model, train_dataloader, optimizer, scheduler, criterion, mode='train')
        # val_loss, val_acc, val_f1 = train_epoch(args, model, val_dataloader, optimizer, scheduler, criterion, mode='val')
        test_loss, test_acc, test_f1 = train_epoch(args, model, test_dataloader, optimizer, scheduler, criterion, mode='test')
        # add_record('loss', train_loss, val_loss, test_loss)
        # add_record('acc', train_acc, val_acc, test_acc)
        # add_record('f1', train_f1, val_f1, test_f1)
        add_record('loss', train_loss, test_var=test_loss)
        add_record('acc', train_acc, test_var=test_acc)
        add_record('f1', train_f1, test_var=test_f1)
        if epoch == 0 or test_acc > max_test_acc:
            torch.save(model, f'{args.store_path}/model.pth')
        epoch += 1
        if test_acc > max_test_acc:
            max_test_acc = test_acc
        # logger.info(f'[ Epoch {epoch:03d}/{args.epoch:03d} ]( {time_transform(time.time() - epoch_stime)} ) train loss: {train_loss:.5f} train acc: {train_acc:.5f} val loss: {val_loss:.5f} val acc: {val_acc:.5f} test loss: {test_loss:.5f} test acc: {test_acc:.5f}')
        logger.info(f'[ Epoch {epoch:03d}/{args.epoch:03d} ]( {time_transform(time.time() - epoch_stime)} ) train loss: {train_loss:.5f} train acc: {train_acc:.5f} train f1: {train_f1:.5f} test loss: {test_loss:.5f} test acc: {test_acc:.5f} test f1: {test_f1:.5f}')
        if math.isnan(train_loss):
            break
    logger.info(f'Finish train. ( {time_transform(time.time() - train_stime)} )')

    # Testing
    plot_training_curve(args, record['loss'], title='loss', xlabel='epoch', ylabel='loss')
    plot_training_curve(args, record['acc'], title='accuracy', xlabel='epoch', ylabel='accuracy')
    plot_training_curve(args, record['f1'], title='f1', xlabel='epoch', ylabel='f1')
    test_from_file(args, test_dataloader)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()

