# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 20:14:55 2017

@author: Philip
"""

import numpy as np
import matplotlib.pyplot as plt



def plot_font_size(typ='normal'):
    if typ == 'normal':
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12
    elif typ == 'poster':
        SMALL_SIZE = 14
        MEDIUM_SIZE = 16
        BIGGER_SIZE = 18
    
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title





if __name__ == "__main__":
    fig, axes = plt.subplots()
    
    x_test = np.linspace(0,10,100)
    y1_test = np.exp(-x_test)
    y2_test = np.sin(x_test)
    test_data_list = [(y1_test, 'exp(-x)'), (y2_test,'sin(x)')]
    
#    print(zip([1,2],test_data_list))
    
    for num, (data, lab) in zip([1,2],test_data_list):
        print(num,data,lab)
#    ax = multi_scales_plot(axes, test_data_list)
