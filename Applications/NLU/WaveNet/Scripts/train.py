
# coding: utf-8

# # Setup

# ## Imports

# In[1]:


import IPython.display as display
import os.path
import itertools
import numpy as np

from glob import glob
from scipy.io import wavfile
from tqdm import tqdm
from time import time


# PyTorch Modules

# In[2]:


import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable


# vai Modules

# In[3]:


from vai.utils.utils import path_consts, randpick
from vai.io import pickle_load, pickle_dump
for k, v in path_consts(): exec(k + ' = v')


# Local Modules

# In[4]:


from wavenet_utils import *


# ## Checks

# In[5]:


assert 'LJSpeech' in DIR_DATA.keys(), 'Dataset not downloaded. Please run Preprocess notebook first.'


# ## Hyperparameters

# In[182]:


batch_size = 32
num_layers = 1
num_stacks = 1
loss_size = 32


# In[160]:


receptive_field = num_stacks * (2 ** num_layers - 1) + 2
frame_size = receptive_field + loss_size - 1
skip_size = loss_size


# In[161]:


dilation_channels = 32
residual_channels = 32
skip_channels = 256
end_channels = 256


# In[111]:


bias = False


# Fixed

# In[162]:


use_gpu = True
num_classes = 256
soft = False
shuffle = False
training_mins = 1


# ## Define Useful Features

# In[163]:


cuda = lambda x: x.cuda() if torch.cuda.is_available() and use_gpu else x


# In[164]:


DIR_DATA = DIR_DATA['LJSpeech']


# In[165]:


def categorize(x):
    l = -1
    h = 1
    d = (h - l) / (2 * num_classes)
    
    buckets = np.linspace(-1 + d, 1 - d, num_classes, dtype=x.dtype)
    
    x = mu_law(x)
    if soft:
        return gaussian_filter(x, buckets, std=d)
    else:
        return one_hot(np.maximum(((x - l) / d - 1) // 2, 0).astype(np.uint64), num_classes)


# In[166]:


def decategorize(x, temperature=1):
    l = -1
    h = 1
    d = (h - l) / (2 * num_classes)
    
    buckets = np.linspace(-1 + d, 1 - d, num_classes, dtype=x.dtype)
    
    if soft:
        epsilon = 1e-12
        return inv_mu_law(softmax(np.log(x + epsilon) / temperature).dot(buckets))
    else:
        return inv_mu_law(l + (2 * x.argmax(-1) + 1) * d)


# ## Load Data

# In[15]:


filenames = sorted(glob(os.path.join(DIR_DATA, 'wavs', '*.wav')))


# Play sample audio

# In[16]:


rate = wavfile.read(randpick(filenames))[0]


# In[ ]:


temp_filenames = []
total_mins = 0

for filename in tqdm(filenames):
    rate, aud = wavfile.read(filename)
    total_mins += (len(aud) / rate) / 60
    temp_filenames.append(filename)
    if total_mins >= training_mins: break
        
filenames = temp_filenames

del temp_filenames, total_mins


# In[ ]:


data = []
for filename in tqdm(filenames):
    data.append(wavfile.read(filename)[1])

data = np.hstack(data)
data = data.astype(np.float32) / 2 ** 15


# In[183]:


batches_per_epoch = int(np.ceil(len(data) / (batch_size * frame_size)))


# ## Create Data Generator

# In[23]:


def __get_static_data(start_idx):
    x = data[start_idx:start_idx + batch_size * frame_size + 1]
    x = categorize(x)
    
    _input = x[:-1].reshape(batch_size, frame_size, num_classes)
    target = x[1:].reshape(batch_size, frame_size, num_classes)
    
    return _input, target


# In[24]:


def __get_random_data():
    _input = np.zeros((batch_size * frame_size, num_classes), dtype=data.dtype)
    target = _input.copy()
    
    for i in range(batch_size):
        rand_idx = np.random.randint(len(data) - frame_size - 1)
        x = data[rand_idx:rand_idx + frame_size + 1]
        x = categorize(x)
        _input[i * frame_size:i * frame_size + frame_size, :] = x[:-1]
        target[i * frame_size:i * frame_size + frame_size, :] = x[1:]
        
    return _input.reshape(batch_size, frame_size, num_classes), target.reshape(batch_size, frame_size, num_classes)


# Parameter ```epoch``` is a float which signifies the state of the generator.
# 
# For example, if ```epoch = 0.5```, the data is read starting from roughly the middle.
# 
# If none, data is randomly sequenced.

# In[25]:


def data_generator(epoch=None):
    if epoch is not None:
        start_idx = int(epoch * len(data))
        if start_idx + batch_size * frame_size + 1 > len(data): start_idx = 0
        
    while True:
        if epoch is None:
            _input, target = __get_random_data()
        else:
            _input, target = __get_static_data(start_idx)
            start_idx += batch_size * frame_size
            if start_idx + batch_size * frame_size + 1 > len(data): start_idx = 0

        _input = np.transpose(_input, [0, 2, 1])
        _input = Variable(cuda(torch.from_numpy(_input)), requires_grad=False)
        
        target = target.reshape((-1, num_classes))
        target = Variable(cuda(torch.from_numpy(target)), requires_grad=False)

        yield _input, target


# # Define Model

# A Residual Block which is repeated many times over as a stack

# In[26]:


class ResBlock(nn.Module):
    def __init__(self, dilation=1, res_connect=True):
        super().__init__()
        self.conv_tanh = CausalConv1d(residual_channels, dilation_channels, 2, dilation, bias=bias)
        self.conv_sigmoid = CausalConv1d(residual_channels, dilation_channels, 2, dilation, bias=bias)
        self.conv_skip = CausalConv1d(dilation_channels, skip_channels, 1, bias=bias)
        
        self.res_connect = res_connect
        if res_connect: self.conv_res = CausalConv1d(dilation_channels, residual_channels, 1, bias=bias)
        
    def forward(self, x):
        x_tanh = F.tanh(self.conv_tanh(x))
        x_sigmoid = F.sigmoid(self.conv_sigmoid(x))
        out = x_tanh * x_sigmoid
        skip = self.conv_skip(out)
        res = self.conv_res(out) + x if self.res_connect else skip
        return res, skip


# In[158]:


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = CausalConv1d(num_classes, residual_channels, 1, bias=bias)
        self.conv2 = CausalConv1d(skip_channels, end_channels, 1, bias=bias)
        self.conv3 = CausalConv1d(end_channels, num_classes, 1, bias=bias)
        
        dilations = list(itertools.chain(*[[2 ** i for i in range(num_layers)] for _ in range(num_stacks)]))
        dilations.pop()
        self.res_layers = nn.ModuleList([ResBlock(dilation) for dilation in dilations])
        self.res_layers.append(ResBlock(2 ** (num_layers - 1), res_connect=False))
        
    def forward(self, x):
        res = self.conv1(x)
        
        skip_total = 0
        for res_layer in self.res_layers:
            res, skip = res_layer(res)
            skip_total += skip
            
        out = F.relu(skip_total)
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        if soft: out = F.softmax(out, 1)
        return out


# In[28]:


def cross_entropy(_input, target):
    if soft:
        return -(target * torch.log(_input + 1e-12)).sum(-1).mean()
    else:
        return nn.CrossEntropyLoss()(_input, target.max(-1)[1])


# In[184]:


model = cuda(Model())
history = {'epochs': 0, 'batches': 0, 'running_epoch': [], 'time': [], 'loss': []}


# In[185]:


optimizer = optim.Adam(model.parameters())


# # Train Model

# In[31]:


def load_model():
    global model, history
    if not os.path.exists(os.path.join(DIR_CHECKPOINTS, 'model.pt')): return
    model.load_state_dict(torch.load(os.path.join(DIR_CHECKPOINTS, 'model.pt')))
    
    history = pickle_load(os.path.join(DIR_CHECKPOINTS, 'history.p'))


# In[32]:


def save_model():
    torch.save(model.state_dict(), os.path.join(DIR_CHECKPOINTS, 'model.pt'))
    pickle_dump(os.path.join(DIR_CHECKPOINTS, 'history.p'), history)


# In[33]:


def print_tree(output):
    g_fn = [output.grad_fn]
            
    while True:
        print(g_fn, '\n')
        new_g_fn = []
        for g in g_fn:
            if g is None: continue
            next_fn = g.next_functions
            for fn, _ in next_fn:
                new_g_fn.append(fn)
        g_fn = new_g_fn
        if len(g_fn) == 0: break


# In[102]:


def show_zero_grads(module=None, percent_only=True):
    if module is None:
        if not percent_only:
            for c in model.children():
                for p in c.parameters():
                    if p.grad is None: continue
                    x = np.where(p.grad.data.cpu().numpy() == 0)
                    if len(x[0]) > 0: 
                        return "{} {} {}% {}".format(c, len(x[0]), int(100 * len(x[0]) / len(p.view(-1))), '%', p.size())
        else:
            z_count = 0
            count = 0
            for c in model.children():
                for p in c.parameters():
                    if p.grad is None: continue
                    x = np.where(p.grad.data.cpu().numpy() == 0)
                    z_count += len(x[0])
                    count += len(p.view(-1))
            return 100 * z_count / count
                    
    else:
        for p in module.parameters():
            x = np.where(p.grad.data.cpu().numpy() == 0)
            return 100 * len(x[0]) / len(p.view(-1))


# In[186]:


def optimize(epochs=1, lr=1e-3, save_every=1):
    load_model()
    model.train()
    optimizer.param_groups[0]['lr'] = lr
    starting_epoch = history['epochs']
    start_time = time()
    
    if shuffle:
        gen = data_generator()
    elif len(history['running_epoch']) != 0:
        gen = data_generator(np.modf(history['running_epoch'][-1])[0])
    else:
        gen = data_generator(0)

    running_loss = []
    saved_once = False
    for epoch in tqdm(range(starting_epoch, starting_epoch + epochs)):
        prog_bar = tqdm(range(history['batches'], batches_per_epoch))
        for batch in prog_bar:
            _input, target = next(gen)
            _input = torch.transpose(model(_input), 1, 2).contiguous().view(-1, 256)
            
            loss = cross_entropy(_input, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.data.cpu().numpy()[0])
            if saved_once: running_loss.pop(0)
                
            prog_bar.set_description('loss:{:.2f}'.format(sum(running_loss) / len(running_loss)))
            history['batches'] += 1
            if time() - start_time > save_every * 60:
                history['running_epoch'].append(epoch + float(batch) / batches_per_epoch)
                history['time'].append(time())
                history['loss'].append(sum(running_loss) / len(running_loss))
                saved_once = True
                save_model()
                start_time = time()
            
        history['epochs'] += 1
        history['batches'] = 0

    save_model()


# In[188]:


optimize(2)