from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
from utils import preprocess_gradients
from layer_norm_lstm import LayerNormLSTMCell
from layer_norm import LayerNorm1D


class MetaOptimizer(nn.Module):

    def __init__(self, model, num_layers, hidden_size):
        super(MetaOptimizer, self).__init__()
        self.meta_model = model

        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(3, hidden_size)
        self.ln1 = LayerNorm1D(hidden_size)

        self.lstms = []
        for i in range(num_layers):
            self.lstms.append(LayerNormLSTMCell(hidden_size, hidden_size))

        self.linear2 = nn.Linear(hidden_size, 1)
        self.linear2.weight.data.mul_(0.1)
        self.linear2.bias.data.fill_(0.0)

    def cuda(self):
        super(MetaOptimizer, self).cuda()
        for i in range(len(self.lstms)):
            self.lstms[i].cuda()

    def reset_lstm(self, keep_states=False, model=None, use_cuda=False):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)

        if keep_states:
            for i in range(len(self.lstms)):
                self.hx[i] = Variable(self.hx[i].data)
                self.cx[i] = Variable(self.cx[i].data)
        else:
            self.hx = []
            self.cx = []
            for i in range(len(self.lstms)):
                self.hx.append(Variable(torch.zeros(1, self.hidden_size)))
                self.cx.append(Variable(torch.zeros(1, self.hidden_size)))
                if use_cuda:
                    self.hx[i], self.cx[i] = self.hx[i].cuda(), self.cx[i].cuda()

    def forward(self, x):
        # Gradients preprocessing
        x = F.tanh(self.ln1(self.linear1(x)))

        for i in range(len(self.lstms)):
            if x.size(0) != self.hx[i].size(0):
                self.hx[i] = self.hx[i].expand(x.size(0), self.hx[i].size(1))
                self.cx[i] = self.cx[i].expand(x.size(0), self.cx[i].size(1))

            self.hx[i], self.cx[i] = self.lstms[i](x, (self.hx[i], self.cx[i]))
            x = self.hx[i]

        x = self.linear2(x)
        return x.squeeze()

    def meta_update(self, model_with_grads, loss):
        # First we need to create a flat version of parameters and gradients
        grads = []

        for module in model_with_grads.children():
            grads.append(module._parameters['weight'].grad.data.view(-1))
            grads.append(module._parameters['bias'].grad.data.view(-1))

        flat_params = self.meta_model.get_flat_params()
        flat_grads = preprocess_gradients(torch.cat(grads))

        inputs = Variable(torch.cat((flat_grads, flat_params.data), 1))

        # Meta update itself
        flat_params = flat_params + self(inputs)

        self.meta_model.set_flat_params(flat_params)

        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model


class FastMetaOptimizer(nn.Module):

    def __init__(self, model, num_layers, hidden_size):
        super(FastMetaOptimizer, self).__init__()
        self.meta_model = model

        self.linear1 = nn.Linear(6, 2) # size是D*6，所以这里输入才是6，输出是2，正好和return时候的split(1,1)相对应
        self.linear1.bias.data[0] = 1

    def forward(self, x):
        # Gradients preprocessing
        x = torch.sigmoid(self.linear1(x))
        return x.split(1, 1)

    def reset_lstm(self, keep_states=False, model=None, use_cuda=False):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)

        if keep_states:
            self.f = Variable(self.f.data)
            self.i = Variable(self.i.data)
        else:
            self.f = Variable(torch.zeros(1, 1))
            self.i = Variable(torch.zeros(1, 1))
            if use_cuda:
                self.f = self.f.cuda()
                self.i = self.i.cuda()

    def meta_update(self, model_with_grads, loss):
        """
        :param: model_with_grads: 需要生成参数的模型（目标模型）
        :param: loss: 目标模型的loss
        :return：更新参数后的模型
        """
        # First we need to create a flat version of parameters and gradients
        grads = []

        for module in model_with_grads.children():
            # print(module) # 类似Linear(in_features=32, out_features=10, bias=True)的module
            grads.append(module._parameters['weight'].grad.data.view(-1).unsqueeze(-1))
            # view(-1)直接将数据变为1行，unsqueeze(-1)又在最后一维增加一个维度
            grads.append(module._parameters['bias'].grad.data.view(-1).unsqueeze(-1))
            # print(len(grads))  # 第一次是2，第二轮循环是4

        flat_params = self.meta_model.get_flat_params().unsqueeze(-1)
        flat_grads = torch.cat(grads)
        # print(flat_grads.shape) # torch.Size([25450, 1]) 28*28*32+32+32*10+10，所以对于不同维度的梯度，这里直接拍平了处理
        # TODO:flat_grads就是获取的梯度，改动从这里下手

        # 设flat_params.size(0)=D
        self.i = self.i.expand(flat_params.size(0), 1)
        self.f = self.f.expand(flat_params.size(0), 1)

        loss = loss.expand_as(flat_grads)
        inputs = Variable(torch.cat((preprocess_gradients(flat_grads), flat_params.data, loss), 1)) # size:D*4
        inputs = torch.cat((inputs, self.f, self.i), 1)  # size: D*6
        self.f, self.i = self(inputs)  # 相当于调用forward函数了，self.f和self.i的size都变回D*1了

        # Meta update itself
        flat_params = self.f * flat_params - self.i * Variable(flat_grads)  # 这里都是按元素相乘，更新目标模型的参数
        flat_params = flat_params.view(-1)  # 又变成1D

        self.meta_model.set_flat_params(flat_params)

        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model


class TransformerMetaOptimizer(nn.Module):

    def __init__(self, model, num_layers, hidden_size):
        super(TransformerMetaOptimizer, self).__init__()
        self.meta_model = model

        self.linear1 = nn.Linear(6, 2) # size是D*6，所以这里输入才是6，输出是2，正好和return时候的split(1,1)相对应
        self.linear1.bias.data[0] = 1

    def forward(self, x):
        # Gradients preprocessing
        x = torch.sigmoid(self.linear1(x))
        return x.split(1, 1)

    def reset_lstm(self, keep_states=False, model=None, use_cuda=False):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)

        if keep_states:
            self.f = Variable(self.f.data)
            self.i = Variable(self.i.data)
        else:
            self.f = Variable(torch.zeros(1, 1))
            self.i = Variable(torch.zeros(1, 1))
            if use_cuda:
                self.f = self.f.cuda()
                self.i = self.i.cuda()

    def meta_update(self, model_with_grads, loss):
        """
        :param: model_with_grads: 需要生成参数的模型（目标模型）
        :param: loss: 目标模型的loss
        :return：更新参数后的模型
        """
        # First we need to create a flat version of parameters and gradients
        grads = []

        for module in model_with_grads.children():
            grads.append(module._parameters['weight'].grad.data.view(-1).unsqueeze(-1))
            # view(-1)直接将数据变为1行，unsqueeze(-1)又在最后一维增加一个维度
            grads.append(module._parameters['bias'].grad.data.view(-1).unsqueeze(-1))

        flat_params = self.meta_model.get_flat_params().unsqueeze(-1)
        flat_grads = torch.cat(grads)
        # TODO:flat_grads就是获取的梯度，改动从这里下手

        # 设flat_params.size(0)=D
        self.i = self.i.expand(flat_params.size(0), 1)
        self.f = self.f.expand(flat_params.size(0), 1)

        loss = loss.expand_as(flat_grads)
        inputs = Variable(torch.cat((preprocess_gradients(flat_grads), flat_params.data, loss), 1)) # size:D*4
        inputs = torch.cat((inputs, self.f, self.i), 1)  # size: D*6
        self.f, self.i = self(inputs)  # 相当于调用forward函数了，self.f和self.i的size都变回D*1了

        # Meta update itself
        flat_params = self.f * flat_params - self.i * Variable(flat_grads)  # 这里都是按元素相乘，更新目标模型的参数
        flat_params = flat_params.view(-1)  # 又变成1D

        self.meta_model.set_flat_params(flat_params)

        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model


# A helper class that keeps track of meta updates
# It's done by replacing parameters with variables and applying updates to
# them.


class MetaModel:

    def __init__(self, model):
        self.model = model

    def reset(self):
        for module in self.model.children():
            module._parameters['weight'] = Variable(
                module._parameters['weight'].data)
            module._parameters['bias'] = Variable(
                module._parameters['bias'].data)

    def get_flat_params(self):
        params = []

        for module in self.model.children():
            params.append(module._parameters['weight'].view(-1))
            params.append(module._parameters['bias'].view(-1))

        return torch.cat(params)

    def set_flat_params(self, flat_params):
        """Restore original shapes
        :param: flat_params 1D的向量
        :return: 原来形状的tensor"""
        offset = 0
        for i, module in enumerate(self.model.children()):
            weight_shape = module._parameters['weight'].size()
            bias_shape = module._parameters['bias'].size()

            weight_flat_size = reduce(mul, weight_shape, 1)
            bias_flat_size = reduce(mul, bias_shape, 1)

            module._parameters['weight'] = flat_params[
                                           offset:offset + weight_flat_size].view(*weight_shape)
            module._parameters['bias'] = flat_params[
                                         offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(
                *bias_shape)

            offset += weight_flat_size + bias_flat_size

    def copy_params_from(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelA.data.copy_(modelB.data)

    def copy_params_to(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelB.data.copy_(modelA.data)
