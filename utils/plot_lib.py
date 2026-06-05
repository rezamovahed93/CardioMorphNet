# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:04:02 2024

@author: reza
"""

import matplotlib.pyplot as plt
import numpy as np 

def plot_fun(X, x_axis_label, y_axis_label, title, filename): 
    plt.plot(np.arange(1, X.size + 1), X)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.savefig(filename)
    # plt.savefig('example_plot.jpg', format='jpg', quality=95)  # quality parameter is optional
    plt.close()
    
