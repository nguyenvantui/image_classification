import time
import torch.nn as nn
import  torch.optim as optim
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

batch_size = 128

# look at my code my code is amazing
# slow and steady

criterion = nn.CrossEntropyLoss()

model_name = "vgg19"
net = qmodel(model_name ,"cuda")

train_transforms = transforms.Compose([transforms.ToTensor()
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

# print(datadir_train)
# print(datadir_test)

# train_data = datasets.ImageFolder(datadir_train, transform=train_transforms)

train_load = torch.utils.data.DataLoader(dataset=train_data,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=10)

test_load = torch.utils.data.DataLoader(dataset=test_data,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=10)
def lr_decay(epoch):

    # 0.1 for epoch [0,150)
    # 0.01 for epoch [150,250)
    # 0.001 for epoch [250,350)
    if epoch<3:
        return 0.2

    if epoch<50:
        return 0.1
      
    if epoch<150:
        return 0.01

    if epoch<250:
        return 0.001

    if epoch<350:
        return 0.0001

    return 0.00001
# def generator(epoch):
#     for batch_idx, (inputs, targets) in enumerate(train_load):
#         pass

def train(epoch):

    device = "cuda"
    _lr = lr_decay(epoch)
    net.train()
    print("Learning_rate: ",_lr)
    # if optim_method == "SGD":
    optimizer = optim.SGD(net.parameters(), lr=_lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(), lr=_lr, weight_decay=5e-4)

    train_loss = 0
    correct = 0
    total = 0
    now = time.time()
    # print(len())
    total_time = 0
    for batch_idx, (inputs, targets) in enumerate(train_load):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if (batch_idx + 1)%20 == 0:

            loss_now = (train_loss / (batch_idx + 1))
            time_now = time.time()-now
            total_data = len(train_data)
            done_data = batch_idx * batch_size
            total_time += time_now
            print("Data: {} / {} Loss:{:.7f}  Time:{:.2f} s".format(done_data, total_data, loss_now, time_now))
            now = time.time()
     
    print("Total_time:{} mins".format(total_time/60))    
def test(epoch):

    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_load):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

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
        link_temp = "./checkpoint/*.pt"
        link_temp = link_temp.replace("*",model_name)
        torch.save(state, link_temp)
        best_acc = acc

def main():

    for idx in range(epoch):
        train(idx)
        test(idx)

if __name__ == "__main__":
    print(">>> hey <<<")
    main()
    print(">>> done <<<")
