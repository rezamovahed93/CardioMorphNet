# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:39:21 2024

@author: reza
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

### ConvLSTMCell ==============================================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, device, dropuout_rate):
        
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        # self.height, self.width, self.depth = input_size
        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.device = device 
        
        self.conv = nn.Conv3d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.dropout = nn.Dropout(p=dropuout_rate)
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.dropout(self.conv(combined))
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next
    
    # def init_hidden(self, batch_size):
    #     return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width, self.depth)).to(self.device),
    #             Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width, self.depth)).to(self.device))

    def init_hidden(self, batch_size, depth):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width, depth)).to(self.device),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width, depth)).to(self.device))

### ConvLSTM ==================================================================
class ConvLSTM(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, device, droput_rates, batch_first=False, bias=True):
        super(ConvLSTM, self).__init__()

        # self.height, self.width, self.depth = input_size
        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.device = device
        self.droput_rates = droput_rates

        cell_list = []
        for i in range(self.num_layers):
            if i == 0:
                cur_input_dim = self.input_dim
            else:
                cur_input_dim = self.hidden_dim[i-1]

            # cell_list.append(ConvLSTMCell(input_size= (self.height, self.width, self.depth),
            #                               input_dim= cur_input_dim,
            #                               hidden_dim= self.hidden_dim[i],
            #                               kernel_size= self.kernel_size[i],
            #                               bias= self.bias))
            
            cell_list.append(ConvLSTMCell(input_size= (self.height, self.width),
                                          input_dim= cur_input_dim,
                                          hidden_dim= self.hidden_dim[i],
                                          kernel_size= self.kernel_size[i],
                                          bias= self.bias, device=self.device, 
                                         dropuout_rate = self.droput_rates[i]))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state):
        """
        input_tensor: todo 4-D Tensor either of shape (t, c, h, w)
        Returns: layer_output, new_hidden_state,
        """
        new_hidden_state = []
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h_cur, c_cur = self.cell_list[layer_idx](cur_layer_input, hidden_state[layer_idx])
            new_hidden_state.append([h_cur, c_cur])
            cur_layer_input = h_cur
        return h_cur, new_hidden_state

    def _init_hidden(self, batch_size, depth):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, depth))
        return init_states
    

