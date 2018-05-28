#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 12:30:30 2018

@author: pmirbach
"""

import matplotlib.pyplot as plt
import pickle

from plot_scripts import plot_font_size


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('lines', linewidth=2.2)
plot_font_size('poster')


# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)    




pickle_file = open('data.pkl', mode='rb')

(uVec, Energy, Energy_var_Ht, Energy_var_Ht2, Energy_gutzwiller, Energy_baeriswyl) = pickle.load(pickle_file)



fig, ax = plt.subplots(figsize=(18.0, 12.0))
#mng = plt.get_current_fig_manager()
#mng.window.showMaximized()


ax.plot(uVec, Energy, '-', color='black', label='Exact solution')
ax.plot(uVec[1:], Energy_var_Ht[1:], '-', color=tableau20[4], 
        label=r'$\alpha \cdot$ single + $\beta \cdot H_t \cdot $ single')
ax.plot(uVec[1:], Energy_var_Ht2[1:], '-', color=tableau20[0] , 
        label=r'$\alpha \cdot$ single + $\beta \cdot H_t \cdot $ single + $\gamma \cdot H_t^2 \cdot $ single')
ax.plot(uVec, Energy_gutzwiller,'--', color=tableau20[2] ,  label=r'Gutzwiller')
ax.plot(uVec, Energy_baeriswyl,'--', color=tableau20[6], label=r'Baeriswyl')


ax.set(title='Hubbard 4 site / half filling - variation',
       xlabel=r'$\frac{U}{t}$', ylabel=r'Energy')

ax.legend(loc='best')


fig.savefig('Hub4_var.png')


plt.show()