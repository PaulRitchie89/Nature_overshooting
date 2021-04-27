# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:13:40 2021

@author: Paul Ritchie

Animations for passing through a SN (in double well) for a fast timescale
system (a0 = 2) and a slow timescale system (a0 = 1/30)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
from matplotlib.gridspec import GridSpec

def f(x, p, a0, kappa):
    """
    Generic double saddle-node bifurcation model
    """    
    f = a0*(kappa*(-(x**3) + x) - p)    
    return (f)

def U(x, p, a0, kappa):
    """
    Stability landscape
    """    
    U = a0*(kappa*((x**4)/4 - (x**2)/2) + p*x)    
    return (U)

def Temperature(t, Tcurr, r):
    """
    Linear temperature forcing
    """    
    T = Tcurr + r*t    
    return (T)

# Time parameters
tstart = 0                              # Start time
tend = 125                              # End time
dt = 0.01                               # Spacing between time intervals
n = int((tend - tstart)/dt)             # Number of time intervals
t = np.linspace(tstart, tend, n+1)      # Time values
freq = 20

# System parameters
a0s = [2, 1/30]                         # Inverse timescale parameter
kappa = 3                               # Curvature parameter

# Original forcing parameters
p0 = 0                                  # Start value of forcing
pbif = 2*kappa/(3*np.sqrt(3))           # Saddle-node bifurcaton value

# Temperature forcing parameters
Tcurr = 1                               # Starting temperature
Tbif = 2                                # Temperature threshold/bifurcation
r = 0.02                                # Initial rate of warming

# Assume original forcing proportional to temperature 
scaling = (pbif - p0)/(Tbif - Tcurr)

# Arrays of original forcing and temperature forcing 
p = pbif + (Temperature(t,Tcurr,r)-Tbif)*scaling
T = Temperature(t,Tcurr,r)

# Equilibria
xx = np.linspace(-3, 3, 601)
TT = Tbif + (-xx*(kappa*(xx**2-1))-pbif)/scaling

# Initialise variable 
x = np.zeros(n+1)
x2 = np.zeros(n+1)
x[0] = 1
x2[0] = 1

# Shift and scale factors so that state variable lies between 0 and 1
state_fac = 1.7
state_scal = 3

# Initialise figure
fontsize = 12
fig = plt.figure(figsize=(9, 9))



gs=GridSpec(12,2)

ax1=fig.add_subplot(gs[4:7,0])
ax2=fig.add_subplot(gs[4:7,1])
ax3=fig.add_subplot(gs[7:10,0])
ax4=fig.add_subplot(gs[7:10,1])
ax5=fig.add_subplot(gs[0:3,:])


ax1.set_xlim(1, 3.5)
ax1.set_ylim(0, 100*(1.3+state_fac)/3)
ax1.set_ylabel('System state (%)', fontsize = fontsize)
ax1.tick_params(axis='both', which='major', labelsize=10)
ax1.set_xticklabels([])
ax2.set_xlim(-20, 120)
ax2.set_ylim(-9, 5)
ax2.set_ylabel('Stability \n landscape', fontsize = fontsize)
ax2.tick_params(axis='both', which='major', labelsize=10)
ax2.set_xticklabels([])
ax3.set_xlim(1, 3.5)
ax3.set_ylim(0, 100*(1.3+state_fac)/3)
ax3.set_xlabel('Global warming ($^oC$)', fontsize = fontsize)
ax3.set_ylabel('System state (%)', fontsize = fontsize)
ax3.tick_params(axis='both', which='major', labelsize=10)
ax4.set_xlim(-20, 120)
ax4.set_ylim(-0.15, 0.05)
ax4.set_xlabel('System state (%)', fontsize = fontsize)
ax4.set_ylabel('Stability \n landscape', fontsize = fontsize)
ax4.tick_params(axis='both', which='major', labelsize=10)
ax5.set_xlabel('Time (years)',fontsize=12)
ax5.set_ylabel('Global warming \n ($^oC$)',fontsize=12)
ax5.tick_params(axis='both', which='major', labelsize=10)
ax5.set_xlim(tstart,tend)
ax5.set_ylim(0.5,3.5)
sns.despine()
fig.tight_layout()

caption = (r"$\bf{Video}$" + " " + r"$\bf1:}$" + " Comparison between fast and "
     "slow onset tipping elements as shown in Figure 1 of Ritchie et al. (2021). "
     "Valleys or wells in the stability landscapes represent stable states of the "
     "system and the ball indicates the current state the system resides in. "
     "As the tipping point threshold is approached the right-hand well becomes "
     "shallower before vanishing at the threshold causing the ball to transition "
     "to the left-hand well.")

plt.figtext(0.01, 0.08, caption, fontsize=10, va='top', ha='left', wrap=True)
plt.figtext(0.5, 0.99, 'Linear global warming increase', weight="bold", fontsize=10, va='top', ha='center', wrap=True)
plt.figtext(0.5, 0.68, 'Fast tipping element', weight="bold", fontsize=10, va='top', ha='center', wrap=True)
plt.figtext(0.5, 0.418, 'Slow tipping element', weight="bold", fontsize=10, va='top', ha='center', wrap=True)

# stop

ims = []

for i in range(n):
    # Solve ODE with Euler method
    x[i+1] = x[i] + dt*f(x[i], p[i], a0s[0], kappa)
    x2[i+1] = x2[i] + dt*f(x2[i], p[i], a0s[1], kappa)
    # Plotting and generate frames for animation
    if i % freq == 0:
        im1, = ax1.plot(TT, 100*(xx+state_fac)/state_scal, 'k--')
        im2, = ax1.plot(TT[np.where(xx>1/(np.sqrt(3)))], 100*(xx[np.where(xx>1/(np.sqrt(3)))]+state_fac)/state_scal, 'k',linewidth=1.5)
        im3, = ax1.plot(TT[np.where(xx<-1/(np.sqrt(3)))], 100*(xx[np.where(xx<-1/(np.sqrt(3)))]+state_fac)/state_scal, 'k',linewidth=1.5)
        im4, = ax1.plot(T[0:i+1], 100*(x[0:i+1]+state_fac)/state_scal,'tab:blue',linewidth=2.5)
        im5, = ax1.plot(T[i], 100*(x[i]+state_fac)/state_scal,'tab:blue',marker='.', Markersize = 16)
        im6, = ax2.plot(100*(xx+state_fac)/state_scal, U(xx, p[i], a0s[0], kappa), 'k')
        im7, = ax2.plot(100*(x[i]+state_fac)/state_scal, U(x[i], p[i], a0s[0], kappa), 'tab:blue',marker='.', MarkerSize = 16)
        im8, = ax3.plot(TT, 100*(xx+state_fac)/state_scal, 'k--')
        im9, = ax3.plot(TT[np.where(xx>1/(np.sqrt(3)))], 100*(xx[np.where(xx>1/(np.sqrt(3)))]+state_fac)/state_scal, 'k',linewidth=1.5)
        im10, = ax3.plot(TT[np.where(xx<-1/(np.sqrt(3)))], 100*(xx[np.where(xx<-1/(np.sqrt(3)))]+state_fac)/state_scal, 'k',linewidth=1.5)
        im11, = ax3.plot(T[0:i+1], 100*(x2[0:i+1]+state_fac)/state_scal,'tab:orange',linewidth=2.5)
        im12, = ax3.plot(T[i], 100*(x2[i]+state_fac)/state_scal,'tab:orange',marker='.', Markersize = 16)
        im13, = ax4.plot(100*(xx+state_fac)/state_scal, U(xx, p[i], a0s[1], kappa), 'k')
        im14, = ax4.plot(100*(x2[i]+state_fac)/state_scal, U(x2[i], p[i], a0s[1], kappa), 'tab:orange',marker='.', MarkerSize = 16)
        im15, = ax5.plot([tstart,tend],[Tbif, Tbif], 'k--',linewidth=1.5)
        im16, = ax5.plot(t[:i+1], Temperature(t[:i+1],Tcurr,r),c='k',linewidth=2.5)
        im17, = ax5.plot(t[i], Temperature(t[i],Tcurr,r),c='k',marker='.', Markersize = 16)
        ims.append([im1, im2, im3, im4, im5, im6, im7, im8, im9, im10, im11, im12, im13, im14, im15, im16, im17])
# stop  
plt.rcParams['animation.ffmpeg_path'] = 'PATH_TO_FFMPEG\\bin\\ffmpeg'
ani = animation.ArtistAnimation(fig, ims, interval = 1000)
FFwriter = animation.FFMpegWriter(fps=50)
ani.save('Nature_video1.mp4', writer = FFwriter)
plt.show()