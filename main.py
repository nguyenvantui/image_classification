from ipdb import set_trace as bp

import time
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from model.vgg import VGG
from smodel import qmodel
from etldata import etl
import numpy as np
from matplotlib.pyplot import imshow
from config import *
import torch.nn.functional as F
import torch
from tqdm import tqdm

# look at my code my code is amazing
# slow and steady
# criterion = nn.CrossEntropyLoss()
# bp()
# kill or be kill

batch_size = 80
model_name = "resnet18"
net = qmodel(model_name, "cuda")

print(">>>>>>>> Start <<<<<<<<<<")
train_transforms = transforms.Compose([transforms.Resize((64, 64)),
                                       transforms.ToTensor()
                                       ])

test_transforms = transforms.Compose([transforms.ToTensor()
                                      ])

train_data = etl("training", train_transforms)
test_data = etl("testing", test_transforms)

datadir_train = datadir + "train"
datadir_test = datadir + "test"

# print()
# xx = train_data[1]

best_acc = 0
real_best_acc = 0
# nhan chi so tinh bon thien

link_temp = "./checkpoint/*.pt"
real_link_temp = "./checkpoint/real_*.pt"
link_temp = link_temp.replace("*", model_name)
real_link_temp = real_link_temp.replace("*", model_name)
print(link_temp)
print(real_link_temp)

if os.path.isfile(link_temp) == False:
    state = {
        'net': net.state_dict(),
        'acc': 0,
        'epoch': 0,
    }
    start_epoch = state['epoch']
    best_acc = state['acc']

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, link_temp)
else:
    assert os.path.isdir('checkpoint'), 'Error!'
    checkpoint = torch.load(link_temp)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

if os.path.isfile(real_link_temp) == True:
    real_checkpoint = torch.load(real_link_temp)
    real_best_acc = checkpoint['acc']

# print(datadir_train)
# print(datadir_test)

# train_data = datasets.ImageFolder(datadir_train, transform=train_transforms)

real_test_data = datasets.ImageFolder("data/test", transform=test_transforms)

train_load = torch.utils.data.DataLoader(dataset=train_data,
                                         batch_size=batch_size,
                                         shuffle=True)

test_load = torch.utils.data.DataLoader(dataset=test_data,
                                        batch_size=batch_size,
                                        shuffle=False)

real_test_load = torch.utils.data.DataLoader(dataset=real_test_data,
                                        batch_size=batch_size,
                                        shuffle=False)


def lr_decay(epoch):

    # 0.1 for epoch  from 0 to 150
    # 0.01 for epoch from 150,250
    # 0.001 for epoch [250,350)

    if epoch < 1:
        return 0.3

    if epoch < 3:
        return 0.2

    if epoch < 150:
        return 0.1

    if epoch < 250:
        return 0.001

    if epoch < 350:
        return 0.0001

    return 0.00001

# def generator(epoch):
#     for batch_idx, (inputs, targets) in enumerate(train_load):
#         pass


def train(epoch):

    device = "cuda"
    _lr = lr_decay(epoch)
    net.train()

    print("Learning_rate: ", _lr)
    # if optim_method == "SGD":
    optimizer = optim.SGD(net.parameters(), lr=_lr,
                          momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(), lr=_lr, weight_decay=5e-4)

    train_loss = 0
    correct = 0
    total = 0
    now = time.time()
    # print(len())

    for batch_idx, (inputs, targets) in enumerate(train_load):
		
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if (batch_idx + 1) % 20 == 0:

            loss_now = (train_loss / (batch_idx + 1))
            time_now = time.time()-now
            total_data = len(train_data)
            done_data = batch_idx * batch_size
            print("Data: {} / {} Loss:{:.7f}  Time:{:.2f} s".format(done_data,
                                                                    total_data, loss_now, time_now))
            now = time.time()


def test(epoch):

    global best_acc, link_temp
    device = "cuda"
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_load)):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # print("Batch: {}/{} Loss:{0.4f}".format(batch_idx, len(testloader), (test_loss / (batch_idx + 1))))
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    # print(correct)
    # print(total)
    acc = (100 * correct) / total
    print("Accuracy: {:.3f}%".format(acc))
    if acc > best_acc:
        print('Saving..')
        print('Best accuracy: {:.3f}%'.format(acc))
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        torch.save(state, link_temp)
        best_acc = acc

def test1(epoch):
    global real_best_acc, real_link_temp
    device = "cuda"
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(real_test_load)):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # print("Batch: {}/{} Loss:{0.4f}".format(batch_idx, len(testloader), (test_loss / (batch_idx + 1))))
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    # print(correct)
    # print(total)
    acc = (100 * correct) / total
    print("Accuracy: {:.3f}%".format(acc))
    if acc > real_best_acc:
        print('Saving..')
        print('Best accuracy real test: {:.3f}%'.format(acc))
        state = {
            'net': net.state_dict(),
            'acc': acc,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        torch.save(state, real_link_temp)
        real_best_acc = acc

def main():

    print("Start epoch: ", start_epoch)
    print("Total_epoch: ", total_epoch)
    print("Best now: ", best_acc)
    print("Batch size: ", batch_size)
    # ultra-test
    for idx in range(start_epoch, total_epoch):
        train(idx)
        test(idx)
        test1(idx)

if __name__ == "__main__":
    # pass
    # print(os.path.isfile('./checkpoint/resnet18.pt'))
    # print(">>> done <<<")
    print(">>> hey <<<")
    main()
    print(">>> done <<<")
