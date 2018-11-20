from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import models
from models import Visualizer, SCIFAR  # ANet, BNet, CNet, Visualizer
from config import opt
from torchnet import meter


# NOTE
# 1. model: ACCNet, no other choise


def trainer(**kwargs):
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    # init
    best_acc = -1
    best_epoch = -1

    # data
    train_loader, test_loader = data_process()
    print('Data processed.')

    # models
    model_name = 'ACCNet'
    model = getattr(models, model_name)(opt.num_classes)

    ##################### TODO #####################
    model.cuda()
    if not opt.from_scratch:
        model.load(opt.save_model_path + model_name + '_best.pth')
        nn.init.xavier_normal_(model.conv5.weight)
    else:
        model.init_weights()


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momemtum, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.T_max)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    print('Model done.')

    # train
    for epoch_i in range(opt.epoch):
        scheduler.step()
        loss, accuracy = train(model, train_loader, criterion, optimizer, epoch_i)
        vis.plot('train_loss', loss)
        vis.plot('train_acc', accuracy)

        test_loss, test_acc, best_acc, best_epoch = val(model, test_loader, criterion, best_acc, best_epoch, epoch_i)
        vis.plot('val_loss', loss)
        vis.plot('val_acc', test_acc)

        print('epoch {}, loss {}, train_acc {}, test_acc {}.'.format(epoch_i, loss, accuracy, test_acc))

    # output info
    print('Best epoch: Best accuracy', best_epoch, best_acc)


def tester(**kwargs):
    # data
    _, test_loader = data_process()

    # model
    model = getattr(models, opt.model)(opt.num_classes)
    pretrained_path = opt.save_model_path + opt.model + '_best.pth'
    model.load(pretrained_path)
    model.cuda()

    # forward
    accuracy = test(model, test_loader)
    print('#########################')
    print('TEST ACCURACY: ', accuracy)
    print('#########################')


def data_process():

    if opt.data == 'class1':
        train_set = SCIFAR(root=opt.class1_dir, train=True)
        test_set = SCIFAR(root=opt.class1_dir, train=False)
    else:
        train_set = SCIFAR(root=opt.class2_dir, train=True)
        test_set = SCIFAR(root=opt.class2_dir, train=False)

    mean = train_set.data.mean(axis=(0, 1, 2)) / 255
    std = train_set.data.std(axis=(0, 1, 2)) / 255
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if opt.data == 'cifar10':
        train_set = SCIFAR(root=opt.class1_dir, train=True, transform=transform)
        test_set = SCIFAR(root=opt.class1_dir, train=False, transform=transform_test)
    else:
        train_set = SCIFAR(root=opt.class2_dir, train=True, transform=transform)
        test_set = SCIFAR(root=opt.class2_dir, train=False, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=opt.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=opt.test_batch_size, shuffle=False)

    return train_loader, test_loader


# train and test
def train(model, train_loader, criterion, optimizer, epoch_i):
    model.train()
    # vis loss, accuracy
    loss_meter = meter.AverageValueMeter()
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.cuda()), Variable(target.cuda())

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        # meter and visualize
        loss_meter.add(loss.item())
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        # if batch_idx % 300 ==0:
        #     print("Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}".format(
        #         epoch, batch_idx*len(data), len(train_loader.dataset),
        #         100.*batch_idx/len(train_loader), loss.item()
        #     ))

    accuracy = correct.item() * 1.0 / len(train_loader.dataset)
    # print('train correct: ', correct)
    return loss_meter.value()[0], accuracy


def val(model, test_loader, criterion, best_acc, best_epoch, epoch_i):
    model.eval()
    test_loss = meter.AverageValueMeter()
    correct = 0

    for data, target in test_loader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = model(data)
        loss = criterion(output, target)
        test_loss.add(loss.item())

        pred = output.data.max(1, keepdim=True)[1]  # the second 1 is to retrieve the index
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    acc = correct.item() * 1.0 / len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset), 100.*correct/len(test_loader.dataset)
    # ))

    if acc > best_acc:
        best_epoch = epoch_i
        best_acc = acc
        model.save(name=opt.save_model_path + opt.model + '_best_' + opt.data +'.pth')

    # print('test correct: ', correct)
    return test_loss.value()[0], acc, best_acc, best_epoch


def test(model, test_loader):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = model(data)

        pred = output.data.max(1, keepdim=True)[1]  # the second 1 is to retrieve the index
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    acc = correct[0] * 1.0 / len(test_loader.dataset)
    return acc


if __name__ == '__main__':
    trainer()
