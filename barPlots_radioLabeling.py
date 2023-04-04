# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 12:35:58 2022

@author: colli
"""

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir("c:\\\\Users\\colli\\Documents\\Python Scripts")
mpl.rcParams['figure.dpi'] = 300
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 10})
    
diWater = np.array([0.9747, 0.9803, 0.9820, 0.9803, .9849])
tenMMolar = np.array([0.9075, 0.9314, 0.9125, 0.8581, .9836])
hundredMMolar = np.array([0.9340, 0.9366, 0.9199, 0.9221, .9816])
diWaterRed = np.ones(5) - diWater
tenMMolarRed = np.ones(5) - tenMMolar
hundredMMolarRed = np.ones(5) - hundredMMolar
times = ['10 [min]', '25 [min]', '45 [min]', '60 [min]', 'Washed']
    
greys = plt.cm.Greys(np.linspace(0,1, 5))
reds = plt.cm.Reds(np.linspace(0,1, 5))
with sns.axes_style("white"):
    sns.set_style("ticks")
    sns.set_context("talk")
    
    # plot details
    bar_width = 0.25
    epsilon = .015
    epsilon = .03
    line_width = 1
    opacity = 0.7
    pos_bar_positions = np.arange(len(diWater))
    neg_bar_positions = pos_bar_positions + bar_width
    hundredmMPositions = neg_bar_positions + bar_width
   # washPositions = hundredMPositions + bar_width

    # make bar plots
    diwater_bar = plt.bar(pos_bar_positions, diWater, bar_width,
                              color=greys[1],
                              label='Di Water Sorbed')
    diwater_bar = plt.bar(pos_bar_positions, diWaterRed, bar_width-epsilon,
                              bottom=diWater,
                              alpha=opacity,
                              color=reds[1],
                              edgecolor=reds[1],
                              linewidth=line_width,
                              label='Di Water Unsorbed')
    tenMMolar_bar = plt.bar(neg_bar_positions, tenMMolar, bar_width,
                              color=greys[2],
                              label='10 mM NaCl Sorbed')
    tenMMolar_bar = plt.bar(neg_bar_positions, tenMMolarRed, bar_width-epsilon,
                              bottom=tenMMolar,
                              color=reds[2],
                              edgecolor=reds[2],
                              ecolor=reds[2],
                              linewidth=line_width,
                              label='10 mM NaCl Unsorbed')
    hundredMMolar_bar = plt.bar(hundredmMPositions, hundredMMolar, bar_width,
                              color=greys[3],
                              label='100 mM NaCl Sorbed')
    hundredMMolar_bar = plt.bar(hundredmMPositions, hundredMMolarRed, bar_width-epsilon,
                              bottom=hundredMMolar,
                              color=reds[3],
                              edgecolor=reds[3],
                              ecolor=reds[3],
                              linewidth=line_width,
                              label='100 mM NaCl Unsorbed')

    plt.xticks(neg_bar_positions, times, size=12)
    plt.yticks(size=12)
    plt.ylabel('Fractional $^{64}$Cu$^{2+}$ [-]', size=12)
    plt.legend(loc ='best', prop={'size': 12})
    #plt.legend(loc='best')
    sns.despine()