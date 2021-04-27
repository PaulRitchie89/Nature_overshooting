# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:02:35 2021

@author: Paul Ritchie

Script to animate realistic temperature overshoots in the Cessi version of the
Stommel model
"""

#import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib import animation

def Cessi(y,F,gamma,mu,beta):

    """
    Cessi version of Stommel 2 box model for AMOC
    """
        
    dydt = beta*(gamma*F - y*(1+mu*((1-y)**2)))

    return(dydt)

def U(Q, F, A, beta, mu, gamma):
    """
    Stability landscape
    """    
    U = beta*(2*A*(Q**3)/3 - Q**2 - 4*(mu**2)*(((A*Q-1)/mu)**1.5)*(3*A*Q + 2 - 5*gamma*F)/(15*(A**2)))  
    return (U)


def Q(y,mu,td,beta2):
    """
    Transport function Q
    """
    V = 300*4.5*8250 # Ocean volume (km**3)
    return (V*1000*(1+mu*((1-y)**2))/td/beta2)    

    
def Temperature(t,Tcurr,Tstab,r,mu0,mu1):
    """
    Realistic temperature profile from Huntingford et al. 2017
    """
    mu = mu0 + mu1*t
    gamma = r - mu0*(Tstab-Tcurr)
    
    TempT = Tcurr + gamma*t - (1 - np.exp(-mu*t))*(gamma*t - (Tstab-Tcurr))
    
    return (TempT)

# Temperature profile parameters
Tcurr = 1   # Start level of warming
Tbif = 4    # Tipping point threshold of warming
r = 0.02    # Current rate of global warming (K/yr)
Tstabs = [1.5, 1.5]         # Levels to stabilise warming at
mu0s = [0.0018, -0.0013]    # Transition timescale parameter
mu1s = [2E-7, 0.000007]     # Transition timescale parameter

# Freshwater parameters
Fstart = 8.2E-5 # Freshwater initial level
Fbif = 9.65E-5  # Freshwater thipping threshold

## Set up for initial value problem solver
tspan = [0, 20]     # Time span (centuries)
h = 0.01            # Time step
t = np.arange(tspan[0],tspan[1]+h,h) # Time array
nt = len(t)         # Number of time points
freq = 2

td = 180            # Diffusion timescale
gamma = 1.343E4     # Scaling to non-dimensional freshwater
mu = 6.2            # Ratio of diffusive and advective timescales
beta = 100*1/td     # Timescale (beta=1 => 1 time unit is 180 years)
beta2 = 3600*24*365 # Seconds in a year


scaling = (Fbif - Fstart)/(Tbif - Tcurr)    # Scaling between freshwater and temperature

# Calculate equilibria
yeq = np.linspace(0,1.5,1001)
Feq = yeq*(1 + mu*((1-yeq)**2))/gamma
Teq = Tbif + (Feq-Fbif)/scaling

# Setting up figure
fig = plt.figure(figsize=(11, 9))

gs=GridSpec(12,2)
ax1=fig.add_subplot(gs[4:7,0])
ax2=fig.add_subplot(gs[4:7,1])
ax3=fig.add_subplot(gs[7:10,0])
ax4=fig.add_subplot(gs[7:10,1])
ax5=fig.add_subplot(gs[0:3,:])


ax1.set_ylabel('Flow strength ($Sv$)',fontsize=12)
ax1.tick_params(axis='both', which='major', labelsize=10)
ax1.set_xticklabels([])
ax1.set_xlim(0,7)
ax1.set_ylim(1,10)
ax2.set_ylabel('Stability \n landscape',fontsize=12)
ax2.tick_params(axis='both', which='major', labelsize=10)
ax2.set_xticklabels([])
ax2.set_xlim(1.96,10)
ax2.set_ylim(-7,6)
ax3.set_xlabel('Global warming ($^oC$)',fontsize=12)
ax3.set_ylabel('Flow strength ($Sv$)',fontsize=12)
ax3.tick_params(axis='both', which='major', labelsize=10)
ax3.set_xlim(0,7)
ax3.set_ylim(1,10)
ax4.set_xlabel('Flow strength ($Sv$)',fontsize=12)
ax4.set_ylabel('Stability \n landscape',fontsize=12)
ax4.tick_params(axis='both', which='major', labelsize=10)
ax4.set_xlim(1.96,10)
ax4.set_ylim(-7,6)
ax5.set_xlabel('Time (years)',fontsize=12)
ax5.set_ylabel('Global warming \n ($^oC$)',fontsize=12)
ax5.tick_params(axis='both', which='major', labelsize=10)
ax5.set_xlim(100*tspan[0],100*tspan[-1])
ax5.set_ylim(0.5,6.5)

sns.despine()
fig.tight_layout()

caption = (r"$\bf{Video}$" + " " + r"$\bf2:}$" + " Illustration of overshooting "
     "a tipping point threshold in a model for the Atlantic Meridional Overturning "
     "Circulation (AMOC) as shown in Figure 2 of Ritchie et al. (2021). "
     "Valleys or wells in the stability landscapes represent stable states of the "
     "system and the ball indicates the current state the system resides in. "
     "As the tipping point threshold is approached the right-hand well becomes "
     "shallower before vanishing at the tipping point threshold. If the warming is "
     "reversed too slowly the ball transitions to the left. However, a sufficiently "
     "fast reversal and the right-hand well will reform in time to catch the ball "
     "and prevent tipping.")

plt.figtext(0.01, 0.102, caption, fontsize=10, va='top', ha='left', wrap=True)
plt.figtext(0.5, 0.99, 'Global warming scenarios', weight="bold", fontsize=10, va='top', ha='center', wrap=True)
plt.figtext(0.5, 0.68, 'AMOC under unsafe overshoot', weight="bold", fontsize=10, va='top', ha='center', wrap=True)
plt.figtext(0.5, 0.418, 'AMOC under safe overshoot', weight="bold", fontsize=10, va='top', ha='center', wrap=True)

ims = []    # Initialise empty array for storing video frames 

colours = ['tab:blue','tab:orange']

xx = np.linspace(-2,12,10001)   # Array of flow strengths (used for stability landscape) 

# Initialise salinity arrays
X = np.zeros(len(t)+1)
X2 = np.zeros(len(t)+1)

# Initial salinity level
X[0] = 0.24077               
X2[0] = 0.24077

# Convert temperature scenarios to freshwater scenarios
F = Fbif + (Temperature(100*t,Tcurr,Tstabs[0],r,mu0s[0],mu1s[0])-Tbif)*scaling
F2 = Fbif + (Temperature(100*t,Tcurr,Tstabs[1],r,mu0s[1],mu1s[1])-Tbif)*scaling

# Solve ODE with Euler method for both scenarios
for i in range(len(t)):
    X[i+1] = X[i] + h*Cessi(X[i],F[i],gamma,mu,beta)
    X2[i+1] = X2[i] + h*Cessi(X2[i],F2[i],gamma,mu,beta) 
    
    # # Plotting and generate frames for animation
    if i % freq == 0:  
        im1, = ax1.plot(Teq, Q(yeq,mu,td,beta2), 'k--',linewidth=1.5)
        im2, = ax1.plot(Teq[np.where(yeq<0.427)], Q(yeq,mu,td,beta2)[np.where(yeq<0.427)], 'k',linewidth=1.5)
        im3, = ax1.plot(Teq[np.where(yeq>0.906)], Q(yeq,mu,td,beta2)[np.where(yeq>0.906)], 'k',linewidth=1.5)
        im4, = ax1.plot(Temperature(100*t[:i+1],Tcurr,Tstabs[0],r,mu0s[0],mu1s[0]), Q(X[:i+1],mu,td,beta2),c=colours[0],linewidth=2.5)
        im5, = ax1.plot(Temperature(100*t[i],Tcurr,Tstabs[0],r,mu0s[0],mu1s[0]), Q(X[i],mu,td,beta2),c=colours[0],marker='.', Markersize = 16)
        im6, = ax2.plot(xx,U(xx, F[i], td*beta2/(1000*300*4.5*8250), beta, mu, gamma),'k')
        im7, = ax2.plot(Q(X[i],mu,td,beta2),U(Q(X[i],mu,td,beta2), F[i], td*beta2/(1000*300*4.5*8250), beta, mu, gamma),c=colours[0],marker='.', Markersize = 16)
        im8, = ax3.plot(Teq, Q(yeq,mu,td,beta2), 'k--',linewidth=1.5)
        im9, = ax3.plot(Teq[np.where(yeq<0.427)], Q(yeq,mu,td,beta2)[np.where(yeq<0.427)], 'k',linewidth=1.5)
        im10, = ax3.plot(Teq[np.where(yeq>0.906)], Q(yeq,mu,td,beta2)[np.where(yeq>0.906)], 'k',linewidth=1.5)
        im11, = ax3.plot(Temperature(100*t[:i+1],Tcurr,Tstabs[1],r,mu0s[1],mu1s[1]), Q(X2[:i+1],mu,td,beta2),c=colours[1],linewidth=2.5)
        im12, = ax3.plot(Temperature(100*t[i],Tcurr,Tstabs[1],r,mu0s[1],mu1s[1]), Q(X2[i],mu,td,beta2),c=colours[1],marker='.', Markersize = 16)
        im13, = ax4.plot(xx,U(xx, F2[i], td*beta2/(1000*300*4.5*8250), beta, mu, gamma),'k')
        im14, = ax4.plot(Q(X2[i],mu,td,beta2),U(Q(X2[i],mu,td,beta2), F2[i], td*beta2/(1000*300*4.5*8250), beta, mu, gamma),c=colours[1],marker='.', Markersize = 16)
        im19, = ax5.plot([100*tspan[0],100*tspan[-1]],[Tbif, Tbif], 'k--',linewidth=1.5)
        im15, = ax5.plot(100*t[:i+1], Temperature(100*t[:i+1],Tcurr,Tstabs[0],r,mu0s[0],mu1s[0]),c=colours[0],linewidth=2.5)
        im16, = ax5.plot(100*t[i], Temperature(100*t[i],Tcurr,Tstabs[0],r,mu0s[0],mu1s[0]),c=colours[0],marker='.', Markersize = 16)
        im17, = ax5.plot(100*t[:i+1], Temperature(100*t[:i+1],Tcurr,Tstabs[1],r,mu0s[1],mu1s[1]),c=colours[1],linewidth=2.5)
        im18, = ax5.plot(100*t[i], Temperature(100*t[i],Tcurr,Tstabs[1],r,mu0s[1],mu1s[1]),c=colours[1],marker='.', Markersize = 16)
        ims.append([im1, im2, im3, im4, im5, im6, im7, im8, im9, im10, im11, im12, im13, im14, im15, im16, im17, im18, im19])

# Create video
plt.rcParams['animation.ffmpeg_path'] = 'PATH_TO_FFMPEG\\bin\\ffmpeg'
ani = animation.ArtistAnimation(fig, ims, interval = 1000)
FFwriter = animation.FFMpegWriter(fps=50)
ani.save('Nature_video2.mp4', writer = FFwriter)
plt.show()