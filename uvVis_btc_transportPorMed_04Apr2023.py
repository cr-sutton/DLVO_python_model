# -*- coding: utf-8 -*-
"""
@author: Collin Sutton
"""

# Import libraries

import pandas as pd
from pandas import DataFrame
import os 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt

#import flopy
import sys
from scipy.special import erf as erf
from scipy.special import erfc as erfc
from scipy.special import erfinv as erfinv

def offset_calc(concArr):
    offset = np.average(concArr[5:600]) #600 so that only the time before the breakthrough is considered
    corrected = concArr - offset
    return (offset, corrected)

def molarity_calc(C_clay):
    ## use spectral analysis to convert kaolinite abs to molarity
    filename = "abs_kao_trend.csv"
    file = pd.read_csv(filename) #data file contain raw C output data of valid experiments
    abs_kao = file['abs'].to_numpy()[0:]
    m_kao = file['M'].to_numpy()[0:]
    M_abs_kao = np.polyfit(m_kao, abs_kao, 1) #gives m then y intercept  ###mx+b

    ####determine M for breakthrough of kao1
    molaritykao = C_clay/M_abs_kao[0]
    totalMkao = np.trapz(molaritykao)
    #totalMkao = (totalMkao/60/1000)*1000000 #[umol] #multiple by q [L/t] for the whole experiment (amount of water flowed through the experiment time / the whole time length)
    #/60 becuase 1ml/min to seconds and everything cancels due to 1:1 and then /1000 because of L to mL then *1000 for the conversion to umol
    totalMkao = (totalMkao/60/1000) #[mol] #multiple by q [L/t] for the whole experiment (amount of water flowed through the experiment time / the whole time length)
    #/60 becuase 1ml/min to seconds and everything cancels due to 1:1 and then /1000 because of L to mL then *1000 for the conversion to mmol
    kaoNormalM = molaritykao/0.15
    return (totalMkao, kaoNormalM)


# ADE w reactions 
def ADEwReactions_type1_fun(x, t, v, D, mu, C0, t0, Ci):
    # We are not focused on sorption so R can be set to one (equivalent to kd = 0)
    R = 1
    # 'u' term identical in equation c5 and c6 (type 3 inlet)
    u = v*(1+(4*mu*D/v**2))**(1/2)
    
    # Note that the '\' means continued on the next line
    Atrf = np.exp(-mu*t/R)*(1- (1/2)* erfc((R*x - v*t)/(2*(D*R*t)**(1/2))) - \
        (1/2)*np.exp(v*x/D)*erfc((R*x + v*t)/(2*(D*R*t)**(1/2))))
    
    # term with B(x, t)
    Btrf = 1/2*np.exp((v-u)*x/(2*D))* erfc((R*x - u*t)/(2*(D*R*t)**(1/2))) \
        + 1/2*np.exp((v+u)*x/(2*D))* erfc((R*x + u*t)/ \
        (2*(D*R*t)**(1/2)))
    
    # if a pulse type injection
    if t0 > 0:
        tt0 = t - t0
        
        indices_below_zero = tt0 <= 0
        # set values equal to 1 (but this could be anything)
        tt0[indices_below_zero] = 1
    
        Bttrf = 1/2*np.exp((v-u)*x/(2*D))* erfc((R*x - u*tt0)/(2*(D*R*tt0)**(1/2))) \
            + 1/2*np.exp((v+u)*x/(2*D))* erfc((R*x + u*tt0)/ \
            (2*(D*R*tt0)**(1/2)))
        
        # Now set concentration at those negative times equal to 0
        Bttrf[indices_below_zero] = 0
        
        C_out = Ci*Atrf + C0*Btrf - C0*Bttrf
        
    else: # if a continous injection then ignore the Bttrf term (no superposition)
        C_out = Ci*Atrf + C0*Btrf
        
    
    # Return the concentration (C) from this function
    return C_out
    
os.chdir("c:\\\\Users\\colli\\Documents\\Python Scripts")
mpl.rcParams['figure.dpi'] = 300
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({'font.size': 18})


filename = "uvVis_data.csv"
file = pd.read_csv(filename) #data file contain raw C output data of valid experiments
col_names = file.columns
print("column names:", col_names)

# Replace NaN values in df with 0
file[:] = file[:].fillna(0)

# manually assign value to variable
t_2469s = file["Time ( sec. )"].to_numpy()

t_2469s_pulse = t_2469s[5:2469] # Crop the first 5 sec to remove the 5sec lag between UVvis and actual injection
print(t_2469s_pulse)

#read in tracer and clay data
C_solute1 = file['tracer'].to_numpy()[5:2469]

#apply offset function
offsetTracer1, C_tracer1_corrected = offset_calc(C_solute1)

coppers = plt.cm.copper(np.linspace(0,1, 5))
# Initial concentration 
C0_solute1 = 1.415
#C0_solute1 = np.trapz(C_solute1,x=t_2469s_pulse)/60
C0_kao1 =  0.15 #0.15M Kaolinite

  # Normalized experimental BTC of solute
C_solute1_norm = C_tracer1_corrected/C0_solute1


# manually assign value to variable
t_5325s = file["Time2 ( sec. )"].to_numpy()

t_5325s_pulse = t_5325s[5:5325] # Crop the first 5 sec to remove the 5sec lag between UVvis and actual injection
t_highSaline = t_5325s[5:2400]
t_lowSaline = t_5325s[0:2925]

#determine molarity for clay breakthrough and normalzie
C_kao_highSaline_18Aug22 = file['18Aug2022_kaoHiLowSaline'].to_numpy()[5:2400]
C_kao_lowSaline_18Aug22 = file['18Aug2022_kaoHiLowSaline'].to_numpy()[2400:5325]
C_Cukao_highSaline_22Aug22 = file['22Aug2022_CukaoHiLowSaline'].to_numpy()[5:2400]
C_Cukao_lowSaline_22Aug22 = file['22Aug2022_CukaoHiLowSaline'].to_numpy()[2400:5325]

offsetC_kao_high18Aug22, C_kao_high18Aug22_corrected = offset_calc(C_kao_highSaline_18Aug22)
offsetC_kao_low18Aug22, C_kao_low18Aug22_corrected = offset_calc(C_kao_lowSaline_18Aug22)
offsetC_Cukao_high22Aug22, C_Cukao_high22Aug22_corrected = offset_calc(C_Cukao_highSaline_22Aug22)
offsetC_Cukao_low22Aug22, C_Cukao_low22Aug22_corrected = offset_calc(C_Cukao_lowSaline_22Aug22)
totalKaohigh18Aug2022_M,C_kaohigh18Aug2022_M = molarity_calc(C_kao_high18Aug22_corrected)
totalKaolow18Aug2022_M,C_kaolow18Aug2022_M = molarity_calc(C_kao_low18Aug22_corrected)
totalCuKaohigh22Aug2022_M,C_Cukaohigh22Aug2022_M = molarity_calc(C_Cukao_high22Aug22_corrected)
totalCuKaoLow2Aug2022_M,C_Cukaolow22Aug2022_M = molarity_calc(C_Cukao_low22Aug22_corrected)


gammaPV = [0.742350172007967,0.781278290783994,0.820206409560022,0.859134528336049,0.898062647112077,0.936990765888104,0.975918884664132,2.9994517874746,3.07054967107248,3.14164755467037,3.21274543826825,3.28384332186614,3.35494120546402,3.4260390890619,3.49713697265979,3.56823485625767,3.63933273985555,3.71043062345344,3.78152850705132,3.8526263906492,3.92372427424709]
#gammaCountsN = [2.70833333333333E-07,4.25833333333333E-05,0.000317229166666667,0.0003703125,0.000290541666666667,0.0001426875,7.85208333333333E-05,0,1.12916666666667E-05,0.0000065625,2.59166666666667E-05,0.000052875,2.58958333333333E-05,3.02916666666667E-05,2.09583333333333E-05,2.30208333333333E-05,1.82083333333333E-05,0.000017375,0.0000155,0.00000975,8.29166666666667E-06]
gammaCountsN = [13,2044,15227,17775,13946,6849,3769,0,542,315,1244,2538,1243,1454,1006,1105,874,834,744,468,398]
gammaPV1 = [0.742350172007967,0.781278290783994,0.820206409560022,0.859134528336049,0.898062647112077,0.936990765888104,0.975918884664132]
gammaCountsN1 = [13,2044,15227,17775,13946,6849,3769]
gammaPV2 = [2.9994517874746,3.07054967107248,3.14164755467037,3.21274543826825,3.28384332186614,3.35494120546402,3.4260390890619,3.49713697265979,3.56823485625767,3.63933273985555,3.71043062345344,3.78152850705132,3.8526263906492,3.92372427424709]
gammaCountsN2 = [0,542,315,1244,2538,1243,1454,1006,1105,874,834,744,468,398]
gammaCountsN1 = np.array(gammaCountsN1)
gammaPV1 = np.array(gammaPV1)
gammaCountsN2 = np.array(gammaCountsN2)
gammaPV2 = np.array(gammaPV2)

    
#%%    
L = 9.75
d= 1*2.54 # column diameter (cm)
A = 3.14*((d/2)**2) # cross-sectional area (cm^2) 
col_vol = A*L  # col length =10 cm# column volume (ml)
print ("column volume:", col_vol)

    # tubing 
d_tube = 0.75/10 # tubing inner diameter:0.75 mm (cm)
A_tube = 3.14*((d_tube/2)**2) #area of tubing

# porosity - TO BE FOUND
#pore_vol = 18 #ml
phi = 0.375
print("porosity:",phi)

    # Sediment properties
# define a single particle size (diameter)
dp = 1* 1e-3 # cm (~10 um)
dp_meter = dp/100 #convert cm to m (for eta calculation)

# alpha is the attachment efficiency (stickiness)
# alpha =0.018 # for now assume fully favorable conditions (every particle that hits a grain of sand sticks)
# alphaSaline = 0.025
alpha =0.0045 # for now assume fully favorable conditions (every particle that hits a grain of sand sticks)
alphaSaline = 0.012
alphaSaline = 0.0083
alphaSaline = 0.0162
alphaLowSaline = 0.008

# collector (i.e. sand grain) diameter
gd = 4.2* 1e-2 # cm

# flow ratE (ml/sec)
Q =1/60 # convert ml/min to ml/sec
print("Flow rate:", Q)
# advection velocity
v = Q/(A*phi) # cm/sec
v_tube = Q/A_tube # no column so phi =1
v_meter_per_sec = v/100 #convert cm/sec to m/sec (for eta calculation)
v_tube_meter_per_sec = v_tube/100
print("advection velocity (m/sec):",v_meter_per_sec)

pulse_time = 1/Q # seconds # pulse time = volumn of solute loaded/flow rate
print("pulse time:", pulse_time, " sec")

# dispersivity:
al = 0.059 # cm ->
#al = 0.1 # for clay
# dispersion
D = al*v

# define observation location
x =L #= 9.75 # cm 

# Normalized analytical solute (set mu=0)
C_solute_ADE1 = ADEwReactions_type1_fun(x, t_2469s_pulse, v, D, 0, C0_solute1, pulse_time, 0)

# Normalzed analytical kaolinite (set mu = kc)
C_kao_ADEHighSaline= ADEwReactions_type1_fun(x, t_2469s_pulse, v, D, 0.0019, 0.15, pulse_time, 0)
C_kao_ADEHighSaline2= ADEwReactions_type1_fun(x, t_2469s_pulse, v, D, 0.0021, 0.15, pulse_time, 0)


import matplotlib.gridspec as gridspec

########################

#########plot for kaolinite transport paper##############
mpl.rcParams['figure.dpi'] = 300
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 20})
# fig = plt.subplots(figsize=(16,7), dpi=300)
#fig = plt.subplots(figsize=(32,14), dpi=300)
fig = plt.figure()
plt.gcf().set_size_inches(15, 8)
gs = gridspec.GridSpec(2, 2)
##top plot
ax1 = fig.add_subplot(gs[0, 0])
ax2 = ax1.twinx()
ax1.plot((t_2469s_pulse/60)/18.41, C_solute1_norm,   c='black', label = 'Fluorescein Tracer')
ax1.plot((t_2469s_pulse/60)/18.41, C_solute_ADE1,'--', c= 'k', label = 'Analytical ADE Model')
ax2.plot(gammaPV1, gammaCountsN1, 'o',markersize=6, c='gray', label = 'Gamma Counts')
#ax2.errorbar(gammaPV1, gammaCountsN1,yerr=gammaCountErrors1, capsize=10, c='gray')
ax2.plot(gammaPV1, gammaCountsN1, c='gray')
#ax1.legend(loc ='best', prop={'size': 10})
# labels = ['Kaolinite - 100mM to 1mM NaCl', 'Kaolinite + Cu - 100mM to 1mM NaCl', 'Normalized Gamma Counts']
lines = []
labels = []
for ax in fig.axes:
    Line, Label = ax.get_legend_handles_labels()
    # print(Label)
    lines.extend(Line)
    labels.extend(Label)
#fig.legend(lines, labels, loc='upper left', bbox_to_anchor=(.48, 0., 0.5, 0.89), prop={'size': 16})
fig.legend(lines, labels, loc='upper left', bbox_to_anchor=(.55, 0., 0.5, 0.89), prop={'size': 16})

#ax1.set_xlabel('Pore Volumes Injected', size=20)
#ax1.set_ylabel('$C/C_0$ [-]', size=20)
# plt.xticks(size=14)
# plt.yticks(size=14)
# ax2.set_ylabel('Gamma Counts', size=20)
# plt.xticks(size=14)
# plt.yticks(size=24)
# ax1.tick_params(axis="y", size=16)
# ax2.tick_params(axis="y", size=16)

ax2.set_ylim([-990000,20000000])
ax2.set_xticks([])
# ax2.set_yticks([])
ax2.set_ylabel('Gamma Counts', size=20)
#ax2.text(3.7, 0.0011, '1mM NaCl Transition', ha='center', va='center', fontsize=20)
#ax2.text(0.9, 0.0011, '100mM NaCl Injection', ha='center', va='center', fontsize=20)
# plt.axhline(y = 0, color = 'r', linestyle = '-')
# plt.axhline(y = 0.2, color = 'r', linestyle = '-')
#plt.axvline(x = 1, color = 'r', linestyle = '-')
plt.xlim([0,2.165])
ax1.tick_params(axis="both", size=20)
#plt.title('Spectrop

###bottom plot
ax3 = fig.add_subplot(gs[1, :])
ax4 = ax3.twinx()
lns1 = ax3.plot((t_highSaline/60)/18.41, C_kaohigh18Aug2022_M,c='black', label = 'Kaolinite - 100mM to 1mM NaCl')
ax3.plot((((t_lowSaline/60)/18.41)+((t_highSaline[-1]/60)/18.41)), C_kaolow18Aug2022_M,c='black')
lns2 = ax3.plot((t_highSaline/60)/18.41, C_Cukaohigh22Aug2022_M,c=coppers[3], label = 'Kaolinite + Cu - 100mM to 1mM NaCl')
ax3.plot((((t_lowSaline/60)/18.41)+((t_highSaline[-1]/60)/18.41)), C_Cukaolow22Aug2022_M,c=coppers[3])
#ax3.plot((t_2469s_pulse/60)/18.41, C_kaoHighSalineADE_M,'--',c='gray', label = 'ADE Kaolinite - 100mM NaCl')
#ax3.plot((t_2469s_pulse/60)/18.41, C_kaoHighSalineADE_M2,'--',c='pink', label = 'ADE Kaolinite - 100mM NaCl')
plt.axvline(x=2.1727, ymin=0.1, ymax=.95,c='black',linestyle='dashed', linewidth=2)
# plt.plot(t_lowSaline+t_highSaline[-1], C_kaolow18Aug2022_M,c='black')
# plt.plot(t_lowSaline+t_highSaline[-1], C_Cukaolow22Aug2022_M,c=coppers[3])

lns3 = ax4.plot(gammaPV1, gammaCountsN1, 'o',markersize=6, c='gray', label = 'Gamma Counts')
#ax4.errorbar(gammaPV1, gammaCountsN1,yerr=gammaCountPropErrors1, capsize=5, c='gray')
ax4.plot(gammaPV1, gammaCountsN1, c='gray')
ax4.plot(gammaPV2, gammaCountsN2, 'o',markersize=6, c='gray')
#ax4.errorbar(gammaPV2, gammaCountsN2,yerr=gammaCountPropErrors2, capsize=5, c='gray')
ax4.plot(gammaPV2, gammaCountsN2, c='gray')


# fig.legend([ax1, ax2], labels=labels,loc="upper right")
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax3.legend(lns, labs, loc='upper left', bbox_to_anchor=(0.65, 0.65, 0.5, 0.89), prop={'size': 16})


ax3.set_xlabel('Pore Volumes Injected', size=20)
#ax3.set_ylabel('$C/C_0$ [-]', size=20)
ax3.text(3.7, 0.000295, '1mM NaCl Transition', ha='center', va='center', fontsize=20)
ax3.text(0.9, 0.000295, '100mM NaCl Injection', ha='center', va='center', fontsize=20)
# plt.xticks(size=14)
# plt.yticks(size=14)
ax4.set_ylabel('Gamma Counts', size=20)
# plt.xticks(size=14)
# plt.yticks(size=24)
ax3.tick_params(axis="y", size=16)
ax3.tick_params(axis="both", size=20)
# ax4.tick_params(axis="y", size=16)
ax4.set_ylim([-400,5000])
#ax4.set_ylim([-0.000085,0.001045])
#ax4.set_yticks([])
#ax2.text(3.7, 0.0011, '1mM NaCl Transition', ha='center', va='center', fontsize=20)
ax1.text(-.6, -0.092, 'C/C0 [-]', rotation=90, fontsize=20)
#plt.axhline(y = 0, color = 'r', linestyle = '-')
##plt.axhline(y = 0.001, color = 'r', linestyle = '-')
plt.xlim([0,4.75])
#plt.savefig('KaoliniteBTC_spectro_AGUPoster.png', transparent= True)
plt.show() 
plt.close()
