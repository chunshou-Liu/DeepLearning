# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:58:42 2019

@author: Susan
"""
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F  # get avtivation functions
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#%%
LR = 0.01
EPOCH = 300
BATCH_SIZE = 64

#%%
'''Q1-1 | Data Process'''
classes = ('Dog', 'Horse', 'Elephant', 'Butterﬂy', 'Chicken', 'Cat', 'Cow', 'Sheep', 'Spider', 'Squirrel')    

transform = transforms.Compose(
         [transforms.Resize([32,32]),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train = torchvision.datasets.ImageFolder(root='./animal-10/train/', transform=transform)
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

val = torchvision.datasets.ImageFolder(root='./animal-10/val/', transform=transform)
val_loader = torch.utils.data.DataLoader(val, batch_size=len(val), shuffle=True, num_workers=2)

#%%
'''Build CNN(4 Layer) Model'''
class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.conv1 = nn.Sequential(
                # if stride = 1 , padding = (kernel_size - 1)/2 = (5-1)/2 
                nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 5, \
                          stride = 1, padding = 2),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )
        self.conv3 = nn.Sequential(
                # if stride = 1 , padding = (kernel_size - 1)/2 = (5-1)/2 
                nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, \
                          stride = 1, padding = 1),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )
        self.conv4 = nn.Sequential(
                nn.Conv2d(16, 32, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )
        self.out = nn.Linear(32*2*2,10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1,x.size(1)*x.size(2)*x.size(3))
        output = self.out(x)
        return output

#%%
'''Train CNN(4 Layer) Model'''
c = classifier()
c.cuda()
print(c)
optimizer = torch.optim.Adam(c.parameters(), lr = LR)
loss_func = nn.CrossEntropyLoss()
loss_func.cuda()

test_acc = np.zeros((EPOCH,))
train_acc = np.zeros((EPOCH,))
train_loss = np.zeros((EPOCH,))
# training and testing
for epoch in range(EPOCH):
    mini_loss, mini_acc = 0, 0
    start = time.time()
    for i, data in enumerate(train_loader, 1):
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        output = c(inputs)               # cnn output
        loss = loss_func(output, labels)   # cross entropy loss
        mini_loss = mini_loss + loss.data.cpu().numpy()
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        
        pred_y = torch.max(output, 1)[1].data.cpu().numpy()
        accuracy = float((pred_y == labels.data.cpu().numpy()).astype(int).sum()) / float(labels.size(0))
        mini_acc = mini_acc + accuracy

    for j, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        test_output = c(inputs)
    pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
    accuracy = float((pred_y == labels.data.cpu().numpy()).astype(int).sum()) / float(labels.size(0))
    end = time.time()
    print('Epoch: ', epoch + 1, '|times: %.fs' % (end-start), '| train loss: %.4f' % (mini_loss/i), '| train accuracy: %.4f' % (mini_acc/i), '| test accuracy: %.2f' % accuracy)
    test_acc[epoch] = accuracy
    train_acc[epoch] = (mini_acc/i)
    train_loss[epoch] = (mini_loss/i)

#%%
'''Q1-2.2 CNN(4 Layers) | compare between models'''
plt.plot(train_loss)
plt.title('Learning curve')
plt.xlabel('# of epochs')
plt.ylabel('Cross entropy')
plt.show()
plt.close()

plt.plot(train_acc, label='train')
plt.plot(test_acc, label='test')
plt.title('Accuracy')
plt.xlabel('# of epochs')
plt.ylabel('Accuracy rate')
plt.legend()
plt.show()
plt.close()
#%%
'''Build CNN(2 Layer) Model'''
class classifier_2(nn.Module):
    def __init__(self):
        super(classifier_2, self).__init__()
        self.conv1 = nn.Sequential(
                # if stride = 1 , padding = (kernel_size - 1)/2 = (5-1)/2 
                nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 5, \
                          stride = 1, padding = 2),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )
        
        self.out = nn.Linear(32*8*8,10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1,x.size(1)*x.size(2)*x.size(3))
        output = self.out(x)
        return output
#%%
'''Train CNN(2 Layer) Model'''
c2 = classifier_2()
c2.cuda()
print(c2)
optimizer2 = torch.optim.Adam(c2.parameters(), lr = LR)
loss_func2 = nn.CrossEntropyLoss()
loss_func2.cuda()

test_acc2 = np.zeros((EPOCH,))
train_acc2 = np.zeros((EPOCH,))
train_loss2 = np.zeros((EPOCH,))
# training and testing
for epoch in range(EPOCH):
    mini_loss2, mini_acc2 = 0, 0
    start = time.time()
    for i, data in enumerate(train_loader, 1):
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        output = c2(inputs)               # cnn output
        loss = loss_func2(output, labels)   # cross entropy loss
        mini_loss2 = mini_loss2 + loss.data.cpu().numpy()
        optimizer2.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer2.step()                # apply gradients
        
        pred_y = torch.max(output, 1)[1].data.cpu().numpy()
        accuracy = float((pred_y == labels.data.cpu().numpy()).astype(int).sum()) / float(labels.size(0))
        mini_acc2 = mini_acc2 + accuracy

    for j, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        test_output = c2(inputs)
    pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
    accuracy = float((pred_y == labels.data.cpu().numpy()).astype(int).sum()) / float(labels.size(0))
    end = time.time()
    print('Epoch: ', epoch + 1, '|times: %.fs' % (end-start), '| train loss: %.4f' % (mini_loss2/i), '| train accuracy: %.4f' % (mini_acc2/i), '| test accuracy: %.2f' % accuracy)
    test_acc2[epoch] = accuracy
    train_acc2[epoch] = (mini_acc2/i)
    train_loss2[epoch] = (mini_loss2/i)
#%%
'''Q1-2.1 CNN(2 Layers) | compare between models'''
plt.plot(train_loss2)
plt.title('Learning curve')
plt.xlabel('# of epochs')
plt.ylabel('Cross entropy')
plt.show()
plt.close()

plt.plot(train_acc2, label='train')
plt.plot(test_acc2, label='test')
plt.title('Accuracy')
plt.xlabel('# of epochs')
plt.ylabel('Accuracy rate')
plt.legend()
plt.show()
plt.close()

#%%
'''Q1-3.1 | show the acc for every class'''
# Make dictionary and set dataloader
class_ = torchvision.datasets.ImageFolder(root='./animal-10/val/', transform=transform, target_transform=None)
loader = torch.utils.data.DataLoader(class_, batch_size=400, shuffle=False, num_workers=2)
dictionary = class_.class_to_idx
dict2 = dict((n, x) for n, x in enumerate(dictionary))
print(dict2)

print('Accuracy of classes')
for j, d in enumerate(loader, 0):
    inputs, labels = d
    inputs = Variable(inputs.cuda())
    labels = Variable(labels.cuda())
    d_output = c(inputs)
    pred_y = torch.max(d_output, 1)[1].data.cpu().numpy()
    accuracy = float((pred_y == labels.data.cpu().numpy()).astype(int).sum()) / float(labels.size(0))
    if j<=9:
        print("%s : %d " % (dict2[j], accuracy*100) + '%')
    else:
        break
#%%
'''Q1-3.2 | show the wrong case'''
# Make Dictionary
class_ = torchvision.datasets.ImageFolder(root='./animal-10/val/', transform=transform, target_transform=None)
loader = torch.utils.data.DataLoader(class_, batch_size=400, shuffle=False, num_workers=2)
print(class_.class_to_idx)
dictionary = class_.class_to_idx
dict2 = dict((n, x) for n, x in enumerate(dictionary))
print(dict2)

# Print photos
unloader = transforms.ToPILImage()
from PIL import Image
# training and testing
start = 7
end = start + 2
for j, d in enumerate(loader, 0):
    if (j<end and j>start):
        inputs, labels = d
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        d_output = c(inputs)

        pred_y = torch.max(d_output, 1)[1].data.cpu().numpy()
    
        for t in range(3):
            if pred_y[t] != 9:
                pre = pred_y[t]
                #print([dict2[x] for x in pred_y])
                #plt.imshow(  inputs[pre].permute(1, 2, 0) )
                image = inputs[pre].cpu().clone()
                image = image / 2 + 0.5
                image = image.squeeze(0)
                image = unloader(image)
                plt.imshow(image)
                print("True label is [%s] ， predict as [%s] " % (dict2[j], dict2[pre]))
            break
    else:
        pass

#%%
'''Q2 | RNN'''
'''Data process'''    
# Read file
accept = pd.read_excel('./ICLR_accepted.xlsx', header=None)[1:]
reject = pd.read_excel('./ICLR_rejected.xlsx', header=None)[1:]
accept.iloc[:,0] = 1
reject.iloc[:,0] = 0

# Split train & test
test = pd.concat([accept[:50], reject[:50]], ignore_index=True)
test_x = test.iloc[:,1]
test_y = test.iloc[:,0]

train = pd.concat([accept[50:], reject[50:]], ignore_index=True)
train_x = train.iloc[:,1]
train_y = train.iloc[:,0]

#  Making dictionary from training data
word_list = []
for i, _ in enumerate(train_x):
    for x in (_.split()):
        word_list.append(x.lower()) 
word_list = set(word_list)
dictionary = {e:i+1 for i, e in enumerate(word_list)}
dictionary[' '] = 0
# Tokenize(train and test)
for i, _ in enumerate(train_x):   
    a = [] 
    for x in (_.split()):
        a.append(dictionary[x.lower()])
    train_x.iloc[i] = a
    num = len(train_x.iloc[i])
    if num <= 10:
        a = train_x.iloc[i]
        b = [dictionary[' ']] * (10 - num)
        train_x.iloc[i] = a + b
    else:
        train_x.iloc[i] = train_x.iloc[i][:-(num-10)]

for i, _ in enumerate(test_x):   
    a = [] 
    for x in (_.split()):
        if x.lower() in dictionary: 
            a.append(dictionary[x.lower()])
        else:
            a.append(dictionary[' '])
    test_x.iloc[i] = a
    num = len(test_x.iloc[i])
    if num <= 10:
        a = test_x.iloc[i]
        b = [dictionary[' ']] * (10 - num)
        test_x.iloc[i] = a + b
    else:
        test_x.iloc[i] = test_x.iloc[i][:-(num-10)]
#%%
'''Experiment config'''
EPOCH = 300
BATCH_SIZE = 64
TIME_STEP = 10    
INPUT_SIZE = 10
EMBED_SIZE = 100 
LR = 0.001

'''Build RNN model'''
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.embed = nn.Embedding(num_embeddings = len(dictionary), embedding_dim = EMBED_SIZE)
        self.rnn = nn.RNN(input_size = EMBED_SIZE, hidden_size = 8, num_layers = 1)#, dropout = 0.3)
        self.out = nn.Linear(8,2)

    def forward(self, inputs):
        # Embedding
        inputs = inputs.t()
        x = self.embed(inputs) # [batch_sz, seq_len, embed_dim]
        # RNN
        r_out, _ = self.rnn(x)
        # Output
        output = self.out(r_out[-1])
        return output


'''Build LSTM model'''
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding(num_embeddings = len(dictionary), embedding_dim = EMBED_SIZE)
        self.lstm = nn.LSTM(input_size = EMBED_SIZE, hidden_size = 8, num_layers = 1)
        self.out = nn.Linear(8,2)

    def forward(self, inputs):
        inputs = inputs.t()
        x = self.embed(inputs)
        r_out, _ = self.lstm(x)
        output = self.out(r_out[-1])
        return output
        

rnn = RNN()
rnn.cuda()
print(rnn)
optimizer = torch.optim.Adam(rnn.parameters(), lr = LR)
loss_func = nn.CrossEntropyLoss()
loss_func = loss_func.cuda()
#%%
'''Train RNN model'''
test_acc, train_acc, train_loss = [], [], []

idx = np.arange(len(train_x))
val_x, val_y = np.stack(test_x.values), np.stack(test_y.values)

for epoch in range(EPOCH):
    np.random.shuffle(idx)
    x, y = train_x[idx], train_y[idx]
    x = np.stack(x.values)
    y = np.stack(y.values)
    mini_loss, mini_acc = 0, 0
    start = time.time()
    rnn.train()
    for i in range(0, len(x), BATCH_SIZE):
        data = x[i:i+BATCH_SIZE]
        label = y[i:i+BATCH_SIZE]
        data = torch.from_numpy(data).type('torch.LongTensor').cuda() # transform to torch tensors
        label = torch.from_numpy(label).type('torch.LongTensor').cuda()
        output = rnn(data)               # cnn output
        loss = loss_func(output, label)   # cross entropy loss
        mini_loss = mini_loss + loss.data.cpu().numpy()
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        pred_y = torch.max(output, 1)[1].data.cpu().numpy()
        accuracy = float((pred_y == label.data.cpu().numpy()).astype(int).sum()) / float(len(label))
        mini_acc = mini_acc + accuracy
    rnn.eval()
    data = torch.from_numpy(val_x).type('torch.LongTensor').cuda()
    label = torch.from_numpy(val_y).type('torch.LongTensor').cuda()
    output = rnn(data)
    pred_y = torch.max(output, 1)[1].data.cpu().numpy()
    accuracy = float((pred_y == label.data.cpu().numpy()).astype(int).sum()) / float(label.size(0))
    end = time.time()
    print('Epoch: ', epoch + 1, '| times: %.fs' % (end-start), '| train loss: %.4f' % (mini_loss/(i/64+1)), '| train accuracy: %.4f' % (mini_acc/(i/64+1)), '| test accuracy: %.2f' % accuracy)
    test_acc.append(accuracy)
    train_acc.append(mini_acc/(i/64+1))
    train_loss.append(mini_loss/(i/64+1))
#%%
'''Q2-1 RNN Performance'''
plt.plot(train_loss)
plt.title('Learning curve')
plt.xlabel('# of epochs')
plt.ylabel('Cross entropy')
plt.show()
plt.close()

plt.plot(train_acc, label='train')
plt.plot(test_acc, label='test')
plt.title('Accuracy')
plt.xlabel('# of epochs')
plt.ylabel('Accuracy rate')
plt.legend()
plt.show()
plt.close()
#%%
'''Train LSTM model'''
lstm = LSTM()
lstm.cuda()
print(lstm)
optimizer2 = torch.optim.Adam(lstm.parameters(), lr = LR)
loss_func2 = nn.CrossEntropyLoss()
loss_func2 = loss_func2.cuda()

test_acc2, train_acc2, train_loss2 = [], [], []

idx = np.arange(len(train_x))
val_x, val_y = np.stack(test_x.values), np.stack(test_y.values)

for epoch in range(EPOCH):
    np.random.shuffle(idx)
    x, y = train_x[idx], train_y[idx]
    x = np.stack(x.values)
    y = np.stack(y.values)
    mini_loss2, mini_acc2 = 0, 0
    start = time.time()
    lstm.train()
    for i in range(0, len(x), BATCH_SIZE):
        data = x[i:i+BATCH_SIZE]
        label = y[i:i+BATCH_SIZE]
        data = torch.from_numpy(data).type('torch.LongTensor').cuda() # transform to torch tensors
        label = torch.from_numpy(label).type('torch.LongTensor').cuda()
        output = lstm(data)               # cnn output
        loss = loss_func2(output, label)   # cross entropy loss
        mini_loss2 = mini_loss2 + loss.data.cpu().numpy()
        optimizer2.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer2.step()                # apply gradients
        pred_y = torch.max(output, 1)[1].data.cpu().numpy()
        accuracy = float((pred_y == label.data.cpu().numpy()).astype(int).sum()) / float(len(label))
        mini_acc2 = mini_acc2 + accuracy
    lstm.eval()
    data = torch.from_numpy(val_x).type('torch.LongTensor').cuda()
    label = torch.from_numpy(val_y).type('torch.LongTensor').cuda()
    output = lstm(data)
    pred_y = torch.max(output, 1)[1].data.cpu().numpy()
    accuracy = float((pred_y == label.data.cpu().numpy()).astype(int).sum()) / float(label.size(0))
    end = time.time()
    print('Epoch: ', epoch + 1, '| times: %.fs' % (end-start), '| train loss: %.4f' % (mini_loss2/(i/64+1)), '| train accuracy: %.4f' % (mini_acc2/(i/64+1)), '| test accuracy: %.2f' % accuracy)
    test_acc2.append(accuracy)
    train_acc2.append(mini_acc2/(i/64+1))
    train_loss2.append(mini_loss2/(i/64+1))

#%%
'''Q2-2 LSTM Performance'''
plt.plot(train_loss2)
plt.title('Learning curve')
plt.xlabel('# of epochs')
plt.ylabel('Cross entropy')
plt.show()
plt.close()

plt.plot(train_acc2, label='train')
plt.plot(test_acc2, label='test')
plt.title('Accuracy')
plt.xlabel('# of epochs')
plt.ylabel('Accuracy rate')
plt.legend()
plt.show()
plt.close()

#%%
'''Q 2-3 Performance between RNN & LSTM'''
plt.plot(train_acc, label='train', color='r')
plt.plot(test_acc, label='test', color='b')
plt.title('Accuracy')
plt.xlabel('# of epochs')
plt.ylabel('Accuracy rate')
plt.legend()
plt.show()
plt.close()


plt.plot(train_acc2, label='train', color='r')
plt.plot(test_acc2, label='test', color='b')
plt.title('Accuracy')
plt.xlabel('# of epochs')
plt.ylabel('Accuracy rate')
plt.legend()
plt.show()
plt.close()

#%%
#'''MAKING DATASETS'''
#class HW_datasets():
#    def __init__(self, bach_size, grid):
#        self.shuffle = True
#        self.grid = grid
#        self.batch_size = bach_size
#        self.path = {'train':'./animal-10/train', \
#                     'val':'./animal-10/val'
#                }
#        
#        classes = ('Dog', 'Horse', 'Elephant', 'Butterﬂy', 'Chicken', 'Cat', 'Cow', 'Sheep', 'Spider', 'Squirrel')    
#        transform = transforms.Compose(
#                 [transforms.Resize([self.grid,self.grid]),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#        
#        self.train = torchvision.datasets.ImageFolder(root=self.path['train'], transform=transform)
#        self.train_loader = torch.utils.data.DataLoader(train, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=2)
#        
#        self.val = torchvision.datasets.ImageFolder(root=self.path['val'], transform=transform)
#        self.val_loader = torch.utils.data.DataLoader(val, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=2)
#
#    def __len__(self, data):
#        return len(data)
#    