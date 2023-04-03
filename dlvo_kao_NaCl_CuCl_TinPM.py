# -*- coding: utf-8 -*-
"""
@author: Collin Sutton, UW Madison
"""
import numpy as np
import matplotlib.pyplot as plt
# for analytical solutions
from math import exp as exp
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Times New Roman']})
plt.rcParams['font.size'] = 16

###DLVO method taken from "Mobilization of Attached Clustered Colloids in Porous Media" https://doi.org/10.1029/2018WR024504
###Calculations are of a finite thickness particle (kaolinite) and a comparatively infinite thickness grain (quartz sand), hence FTIT on the def names

##### (EQUATION S21)
def dlvo(vlvw_FTIT, vedl_FTIT, vbr_FTIT):
    vdlvo = (vlvw_FTIT+vedl_FTIT+vbr_FTIT)
    return vdlvo

# FT-plate-IT-plate
#van der Waals energies calculation(EQUATION S6)
def vlvw_FTIT(ah, h):
    vlvw_FTIT = -(ah/(12*np.pi*(h**2)))
    return vlvw_FTIT

#electrical double layer energies calculation (EQUATION S3 from Image interpretation for kaolinite detachment from solid substrate: type curves, stochastic model)
#need to do separate calculations for gammaS, gammaG, and k
#gammaS and gammaG are the reduced surface potential of particles (kaolinite) and grain(sand)
def vedl_FTIT(epsilon0, epsilonr, k, kB, T, gammaS, gammaG, z, ee, h):
    vedl_FTIT = 32*epsilon0*epsilonr*k*gammaS*gammaG*(((kB*T)/(z*ee))**2)*(exp(-k*h))
    return vedl_FTIT

## for Born Potential Energies Calculation ## (EQUATION S8)
##### sigmaB [m] is the Born collision parameter - for the commonly used sigmaB = 5 A, the resulting acceptable minimum separation distance, at h=h0 (i.e. contact) is estimated to be h0 = 2.5 A = 0.25 nm which compares well to h0 = 4-10 A (born can be neglected if h>1nm) 
def vbr_FTIT(ah, sigma, h):
    vbr_FTIT = (ah*(sigma**6))/(360*np.pi*(h**8))
    return vbr_FTIT


#gamma calculation for kaolinite  ## (after EQUATION S3)
def gammaS_FTIT(z, ee, zetaKao, kB, T):
    gammaS = np.tanh((z*ee*zetaKao)/(4*kB*T))
    return gammaS

#gamma calculation for sand grain ## (after EQUATION S3)
def gammG_FTIT(z, ee, zetaGlass, kB, T):
    gammaG = np.tanh((z*ee*zetaGlass)/(4*kB*T))
    return gammaG

#Debye-Huckel reciprical double layer length ## ( EQUATION S4)
def k_FTIT(ee, epsilonr, epsilon0, kB, T, zi, ni):
    sigmaZiNi = (zi**2)*ni + ((zi**2)*ni)
    k = np.sqrt(((ee**2)/(epsilonr*epsilon0*kB*T))*sigmaZiNi)
    return k

#Debye-Huckel reciprical double layer length ## ( EQUATION S4)
def k_FTIT2(ee, epsilonr, epsilon0, kB, T, zi, ni, zi2, niCu, niCl2):
    sigmaZiNi = (zi**2)*ni + (zi**2)*ni + (zi2**2)*niCu + (zi2**2)*niCl2
    k = np.sqrt(((ee**2)/(epsilonr*epsilon0*kB*T))*sigmaZiNi)
    return k

#DLVO parameters
ah = 1.49E-20 #[J]
kB = 1.38E-23 #[J/K]
T = 298.15 #[degree K] = 25 degrees C
# atomic collision constant with a typical value of 0.5 nm
sigma = 5E-10 #[m] 
epsilon0 = 8.85E-12 # [C^2/(N*m^2)] #permativity of the solution
epsilonr = 78.46 #(relative permativity of water) [-]
z = 2 #electron valency ## assume z:z electron valency for NaCl - so z+ = 1 (Na) and z- = 1 (Cl)
zCu = 4
ee = 1.6E-19 #elemental charge [C]
Na = 6.02E23 # avogadro's number [1/mol]
zi = float(1)   ## assume z:z electron valency for NaCl - so z+ = 1 (Na) and z- = 1 (Cl)
zi2 = float(2) ## assume z electron valency for CuCl2 - so z+ = 2 (Cu) and z- = 2 (Cl2)

hArr = np.linspace(1E-10, 3E-7, num=100000)
hlen = len(hArr)
vlvw = np.zeros([hlen])
vedl = np.zeros([hlen])
vbr = np.zeros([hlen])
vdlvo = np.zeros([hlen])
vdlvo_0_1NaCl = np.zeros([hlen])
vdlvo_0_01NaCl = np.zeros([hlen])
vdlvo_0_001NaCl = np.zeros([hlen])
vdlvo_0_1NaCl_Cu = np.zeros([hlen])
vdlvo_0_001NaCl_Cu = np.zeros([hlen])

##0.1 Molar NaCl
zetaKao = -38 / 1000 # [V] zeta must be in Volts - unit from zeta sizer is mV
zetaGlass = -57.3 / 1000 # [V] zeta must be in Volts - unit from zeta sizer is mV
epsilonM = 80.1 # Line 80 of SI at 20C
ni = 0.1*1000*Na
gammaS_0_1NaCl = gammaS_FTIT(z, ee, zetaKao, kB, T)
gammaG_0_1NaCl = gammG_FTIT(z, ee, zetaGlass, kB, T)
k_0_1NaCl = k_FTIT(ee, epsilonr, epsilon0, kB, T, zi, ni) #[1/m]
for i in range(0,hlen):
    h = hArr[i]
    vlvw[i] = vlvw_FTIT(ah, h)
    vedl[i] = vedl_FTIT(epsilon0, epsilonr, k_0_1NaCl, kB, T, gammaS_0_1NaCl, gammaG_0_1NaCl, z, ee, h)
    vbr[i] = vbr_FTIT(ah, sigma, h)
    vdlvo[i] = dlvo(vlvw[i], vedl[i], vbr[i])
    vdlvo_0_1NaCl[i] = vdlvo[i]
    
    
#0.001 Molar NaCl
zetaKao3 = -61.6 / 1000
zetaSand3 = -73.6 / 1000 #from implications of Cation Exchange on Clay Release and Colloid-Facilitated Transport in Porous Media - doi:10.2134/jeq2010.0156
ni3 = 0.001*1000*Na
gammaS_0_001NaCl = gammaS_FTIT(z, ee, zetaKao3, kB, T)
gammaG_0_001NaCl = gammG_FTIT(z, ee, zetaSand3, kB, T)
k_0_001NaCl = k_FTIT(ee, epsilonr, epsilon0, kB, T, zi, ni3) #[1/m]
for i in range(0,hlen):
    h = hArr[i]
    vlvw[i] = vlvw_FTIT(ah, h)
    vedl[i] = vedl_FTIT(epsilon0, epsilonr, k_0_001NaCl, kB, T, gammaS_0_001NaCl, gammaG_0_001NaCl, z, ee, h)
    vbr[i] = vbr_FTIT(ah, sigma, h)
    vdlvo[i] = dlvo(vlvw[i], vedl[i], vbr[i])
    vdlvo_0_001NaCl[i] = vdlvo[i]
    
    
##0.1 Molar NaCl with 2.64 mM CuCl2
zetaKaoCu = -25 / 1000 # [V] zeta must be in Volts - unit from zeta sizer is mV
zetaGlass = -57.3 / 1000 # [V] zeta must be in Volts - unit from zeta sizer is mV
epsilonM = 80.1 # Line 80 of SI at 20C
ni = 0.1*1000*Na
niCu = 0.00264*1000*Na
niCl2 =  0.00264*2*1000*Na
gammaS_0_1NaCl = gammaS_FTIT(zCu, ee, zetaKaoCu, kB, T)
gammaG_0_1NaCl = gammG_FTIT(zCu, ee, zetaGlass, kB, T)
k_0_1NaCl_Cu = k_FTIT2(ee, epsilonr, epsilon0, kB, T, zi, ni, zi2, niCu, niCl2) #[1/m]
for i in range(0,hlen):
    h = hArr[i]
    vlvw[i] = vlvw_FTIT(ah, h)
    vedl[i] = vedl_FTIT(epsilon0, epsilonr, k_0_1NaCl_Cu, kB, T, gammaS_0_1NaCl, gammaG_0_1NaCl, zCu, ee, h)
    vbr[i] = vbr_FTIT(ah, sigma, h)
    vdlvo[i] = dlvo(vlvw[i], vedl[i], vbr[i])
    vdlvo_0_1NaCl_Cu[i] = vdlvo[i]   
    
#0.001 Molar NaCl with 2.64 mM CuCl2
zetaKaoCu2 = -60 / 1000
zetaSand3 = -73.6 / 1000 #from implications of Cation Exchange on Clay Release and Colloid-Facilitated Transport in Porous Media - doi:10.2134/jeq2010.0156
ni3 = 0.001*1000*Na
gammaS_0_001NaCl_Cu = gammaS_FTIT(zCu, ee, zetaKaoCu2, kB, T)
gammaG_0_001NaCl = gammG_FTIT(zCu, ee, zetaSand3, kB, T)
k_0_001NaCl_Cu = k_FTIT2(ee, epsilonr, epsilon0, kB, T, zi, ni3, zi2, niCu, niCl2) #[1/m]
for i in range(0,hlen):
    h = hArr[i]
    vlvw[i] = vlvw_FTIT(ah, h)
    vedl[i] = vedl_FTIT(epsilon0, epsilonr, k_0_001NaCl_Cu, kB, T, gammaS_0_001NaCl_Cu, gammaG_0_001NaCl, zCu, ee, h)
    vbr[i] = vbr_FTIT(ah, sigma, h)
    vdlvo[i] = dlvo(vlvw[i], vedl[i], vbr[i])
    vdlvo_0_001NaCl_Cu[i] = vdlvo[i]


vdlvo_0_1NaCl_kBT = vdlvo_0_1NaCl*(2E-5*2E-5)/(kB*T) #surface area of kaolinite (20 um)^2 / kB*T 
vdlvo_0_01NaCl_kBT = vdlvo_0_01NaCl*(2E-5*2E-5)/(kB*T) #surface area of kaolinite (20 um)^2 / kB*T 
vdlvo_0_001NaCl_kBT = vdlvo_0_001NaCl*(2E-5*2E-5)/(kB*T) #surface area of kaolinite (20 um)^2 / kB*T 
vdlvo_0_1NaCl_Cu_kBT = vdlvo_0_1NaCl_Cu*(2E-5*2E-5)/(kB*T) #surface area of kaolinite (20 um)^2 / kB*T 
vdlvo_0_001NaCl_Cu_kBT = vdlvo_0_001NaCl_Cu*(2E-5*2E-5)/(kB*T) #surface area of kaolinite (20 um)^2 / kB*T 


# Raw plot without cropping [units in kbT]
coppers = plt.cm.copper(np.linspace(0,1, 5))
plt.figure(dpi=300)
plt.plot(hArr, vdlvo_0_1NaCl_kBT, c="black", label="0.1M NaCl")
plt.plot(hArr, vdlvo_0_001NaCl_kBT, c="black", linestyle="dashed", label="0.001M NaCl")
plt.plot(hArr, vdlvo_0_1NaCl_Cu_kBT, color=coppers[3], label="0.1M NaCl + Cu")
plt.plot(hArr, vdlvo_0_001NaCl_Cu_kBT, color=coppers[3], linestyle="dashed", label="0.001M NaCl + Cu")
plt.legend(loc ='best', prop={'size': 10})
plt.xscale("log")
plt.xlabel('Separation Distance [m]')
plt.ylabel('U [kbT]')
plt.title('DLVO Kaolinite System') ,plt.show(),plt.close()


# Raw plot with cropping [units in kbT]
plt.figure(dpi=300)
plt.plot(hArr, vdlvo_0_1NaCl_kBT, c="black", label="0.1M NaCl")
plt.plot(hArr, vdlvo_0_001NaCl_kBT, c="black", linestyle="dashed", label="0.001M NaCl")
plt.plot(hArr, vdlvo_0_1NaCl_Cu_kBT, color=coppers[3], label="0.1M NaCl + Cu")
plt.plot(hArr, vdlvo_0_001NaCl_Cu_kBT, color=coppers[3], linestyle="dashed", label="0.001M NaCl + Cu")
plt.legend(loc ='best', prop={'size': 10})
plt.xscale("log")
plt.ylim([-250000000, 100000000])
plt.xlim([1E-10, 3E-7])
plt.xlabel('Separation Distance [m]')
plt.ylabel('U [kbT]')
plt.title('DLVO Kaolinite System') ,plt.show(),plt.close()


# Raw plot with cropping [units in Jm^-2]
plt.figure(dpi=300)
plt.plot(hArr, vdlvo_0_1NaCl, c="black", label="0.1M NaCl")
plt.plot(hArr, vdlvo_0_001NaCl, c="black", linestyle="dashed", label="0.001M NaCl")
plt.plot(hArr, vdlvo_0_1NaCl_Cu, color=coppers[3], label="0.1M NaCl + Cu")
plt.plot(hArr, vdlvo_0_001NaCl_Cu, color=coppers[3], linestyle="dashed", label="0.001M NaCl + Cu")
plt.legend(loc ='best', prop={'size': 10})
plt.xscale("log")
plt.ylim([-0.0025, 0.001])
plt.xlim([1E-10, 3E-7])
plt.xlabel('Separation Distance [m]')
plt.ylabel('U [J/m^2]')
plt.title('DLVO Kaolinite System') ,plt.show(),plt.close()


##Next 5 plots are commented out - show individual plots of primary and secondary minimum for different IS solutions
# plt.figure(dpi=300)
# plt.plot(hArr, vdlvo_0_1NaCl, c="black", label="0.1M NaCl")
# plt.plot(hArr, vdlvo_0_1NaCl_Cu, color=coppers[3], label="0.1M NaCl + Cu")
# plt.legend(loc ='best', prop={'size': 10})
# plt.xscale("log")
# plt.ylim([-0.0025, 0.001])
# plt.xlim([1E-10, 3E-7])
# plt.xlabel('Separation Distance [m]')
# plt.ylabel('U [J/m^2]')
# plt.title('DLVO High IS') ,plt.show(),plt.close()


# fig = plt.figure(dpi=300)
# ax = fig.add_subplot(111)
# ax.plot(hArr, vdlvo_0_1NaCl, c="black", label="0.1M NaCl")
# ax.plot(hArr, vdlvo_0_1NaCl_Cu, color=coppers[3], label="0.1M NaCl + Cu")
# plt.legend(loc ='best', prop={'size': 10})
# plt.xscale("log")
# ax.spines['bottom'].set_position('zero')
# plt.ylim([-0.00001, 0.000005])
# plt.xlim([3E-9, 8E-8])
# plt.xlabel('Separation Distance [m]')
# plt.ylabel('U [J/m^2]')
# plt.title('Secondary Minimum for High IS') ,plt.show(),plt.close()

# plt.figure(dpi=300)
# plt.plot(hArr, vdlvo_0_001NaCl, c="black", linestyle="dashed", label="0.001M NaCl")
# plt.plot(hArr, vdlvo_0_001NaCl_Cu, color=coppers[3], linestyle="dashed", label="0.001M NaCl + Cu")
# plt.legend(loc ='best', prop={'size': 10})
# plt.xscale("log")
# plt.ylim([-0.0025, 0.001])
# plt.xlim([1E-10, 3E-7])
# plt.xlabel('Separation Distance [m]')
# plt.ylabel('U [J/m^2]')
# plt.title('DLVO Low IS') ,plt.show(),plt.close()

# plt.figure(dpi=300)
# plt.plot(hArr, vdlvo_0_001NaCl, c="black", linestyle="dashed", label="0.001M NaCl")
# plt.plot(hArr, vdlvo_0_001NaCl_Cu, color=coppers[3], linestyle="dashed", label="0.001M NaCl + Cu")
# plt.legend(loc ='best', prop={'size': 10})
# plt.xscale("log")
# plt.ylim([-0.0002, 0.00025])
# plt.xlim([8E-10, 6E-8])
# plt.xlabel('Separation Distance [m]')
# plt.ylabel('U [J/m^2]')
# plt.title('DLVO Low IS Energy Barrier') ,plt.show(),plt.close()

# ######### for changing x axis location
# fig = plt.figure(dpi=300)
# ax = fig.add_subplot(111)
# ax.plot(hArr, vdlvo_0_1NaCl, c="black", label="0.1M NaCl")
# ax.plot(hArr, vdlvo_0_001NaCl, c="black", linestyle="dashed", label="0.001M NaCl")
# ax.plot(hArr, vdlvo_0_1NaCl_Cu, color=coppers[3], label="0.1M NaCl + Cu")
# ax.plot(hArr, vdlvo_0_001NaCl_Cu, color=coppers[3], linestyle="dashed", label="0.001M NaCl + Cu")
# plt.legend(loc ='best', prop={'size': 10})
# plt.xscale("log")
# plt.ylim([-0.0000015, 0.0000001])
# plt.xlim([1E-8, 3E-7])
# ax.spines['bottom'].set_position('zero')
# plt.xlabel('[m]')
# plt.ylabel('U [J/m^2]')
# plt.title('Secondary Minimum for low IS') ,plt.show(),plt.close()


#zoom in of entire system with legend and labels
fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
ax.plot(hArr, vdlvo_0_1NaCl_kBT, c="black", label="0.1M NaCl")
#plt.plot(hArr, vdlvo_0_01NaCl_kBT, c="red", label="0.01M NaCl")
ax.plot(hArr, vdlvo_0_001NaCl_kBT, c="black", linestyle="dashed", label="0.001M NaCl")
ax.plot(hArr, vdlvo_0_1NaCl_Cu_kBT, color=coppers[3], label="0.1M NaCl + Cu")
ax.plot(hArr, vdlvo_0_001NaCl_Cu_kBT, color=coppers[3], linestyle="dashed", label="0.001M NaCl + Cu")
plt.legend(loc ='best', prop={'size': 10})
plt.xscale("log")
ax.spines['bottom'].set_position('zero')
plt.ylim([-1.5E7, 3E7])
plt.xlim([7E-10, 8E-8])
plt.xlabel('Separation Distance [m]')
plt.ylabel('U [kbT]')
plt.title('DLVO Kaolinite System') ,plt.show(),plt.close()

##zoom in of secondary minimum for high IS
# fig = plt.figure(dpi=300)
# ax = fig.add_subplot(111)
# ax.plot(hArr, vdlvo_0_1NaCl_kBT, c="black", label="0.1M NaCl")
# # plt.plot(hArr, vdlvo_0_01NaCl, c="red", label="0.01M NaCl")
# #plt.plot(hArr, vdlvo_0_001NaCl, c="black", linestyle="dashed", label="0.001M NaCl")
# ax.plot(hArr, vdlvo_0_1NaCl_Cu_kBT, color=coppers[3], label="0.1M NaCl + Cu")
# #plt.plot(hArr, vdlvo_0_001NaCl_Cu, c="gold", linestyle="dashed", label="0.001M NaCl + Cu")
# plt.legend(loc ='best', prop={'size': 10})
# plt.xscale("log")
# ax.spines['bottom'].set_position('zero')
# plt.ylim([-1.5E6, 1.5E6])
# plt.xlim([3E-9, 8E-8])
# plt.xlabel('Separation Distance [m]')
# plt.ylabel('U [kBT]')
# plt.title('Secondary Minimum for High IS') ,plt.show(),plt.close()

##zoom in of low IS energy barrier
# fig = plt.figure(dpi=300)
# ax = fig.add_subplot(111)
# #plt.plot(hArr, vdlvo_0_1NaCl, c="black", label="0.1M NaCl")
# # plt.plot(hArr, vdlvo_0_01NaCl, c="red", label="0.01M NaCl")
# ax.plot(hArr, vdlvo_0_001NaCl_kBT, c="black", linestyle="dashed", label="0.001M NaCl")
# #plt.plot(hArr, vdlvo_0_1NaCl_Cu, c="gold", label="0.1M NaCl + Cu")
# ax.plot(hArr, vdlvo_0_001NaCl_Cu_kBT, color=coppers[3], linestyle="dashed", label="0.001M NaCl + Cu")
# plt.legend(loc ='best', prop={'size': 10})
# plt.xscale("log")
# ax.spines['bottom'].set_position('zero')
# plt.ylim([-1.5E7, 3E7])
# plt.xlim([8E-10, 6E-8])
# plt.xlabel('Separation Distance [m]')
# plt.ylabel('U [kBT]')
# plt.title('DLVO Low IS Energy Barrier') ,plt.show(),plt.close()





###############manuscript plot#####################
from matplotlib.patches import Rectangle

figure, axes = plt.subplots(1, 2, figsize=(16,8), dpi=300)
axes[0].plot(hArr, vdlvo_0_1NaCl_kBT, c="black", label="100 mM NaCl")
axes[0].plot(hArr, vdlvo_0_001NaCl_kBT, c="black", linestyle="dashed", label="1 mM NaCl")
axes[0].plot(hArr, vdlvo_0_1NaCl_Cu_kBT, color=coppers[3], label="100 mM NaCl + Cu")
axes[0].plot(hArr, vdlvo_0_001NaCl_Cu_kBT, color=coppers[3], linestyle="dashed", label="1 mM NaCl + Cu")
axes[0].axhline(y=0,xmin=0,xmax=3,c="black",linewidth=0.5,zorder=0)
axes[1].plot(hArr, vdlvo_0_1NaCl_kBT, c="black", label="100 mM NaCl")
axes[1].plot(hArr, vdlvo_0_001NaCl_kBT, c="black", linestyle="dashed", label="1 mM NaCl")
axes[1].plot(hArr, vdlvo_0_1NaCl_Cu_kBT, color=coppers[3], label="100 mM NaCl + Cu")
axes[1].plot(hArr, vdlvo_0_001NaCl_Cu_kBT, color=coppers[3], linestyle="dashed", label="1 mM NaCl + Cu")
axes[1].axhline(y=0,xmin=0,xmax=3,c="black",linewidth=0.5,zorder=0)

axes[0].add_patch(Rectangle((7E-10, -1.5E7), 7.93e-8, 45000000,
             edgecolor = 'black',
             facecolor = 'none',
             lw=4))

axes[0].set_ylim([-250000000, 100000000])
axes[0].set_xscale('log')
axes[1].legend(loc ='lower right', prop={'size': 16})
axes[1].set_ylim([-1.5E7, 3E7])
axes[1].set_xlim([7E-10, 8E-8])
axes[1].set_xscale('log')

for axes in axes.flat:
    axes.set(xlabel='Separation Distance [m]', ylabel='U [$k_B T$]')
    axes.tick_params(axis='both', labelsize=18)
    plt.rcParams['axes.labelsize'] = 20
    
ax.xaxis.get_label().set_fontsize(20)
ax.yaxis.get_label().set_fontsize(20)
    
#plt.savefig('dlvo_AGU2022', transparent=True)


