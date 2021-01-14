from itertools import islice

import math
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def window(seq, n=2, prepend=False):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    if prepend:
        seq2 = [None]*(n-1) + seq
    else:
        seq2 = seq
    it = iter(seq2)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result

true_x = [4]
true_y = [0]
speed_x = 0
speed_y = 1
timestep = 1
var_x = 0.5
var_y = 0
length = 50

# Generate true data
for i in range(length):
    true_x.append(true_x[-1] + speed_x * timestep)
    true_y.append(true_y[-1] + speed_y * timestep)

# Generate noisy data
meas_x = [x + math.sqrt(var_x) * np.random.randn() for x in true_x]
meas_y = [y + math.sqrt(var_y) * np.random.randn() for y in true_y]

# ML estimation of position
ML_x = [np.mean(x) for x in meas_x]
ML_y = [np.mean(y) for y in meas_y]

# MAP estimation of position, only for x axis, using Kalman filter, 
for i, mx in enumerate(meas_x):
    # Initially, we have no prior, use only the measurements
    if i == 0:
        MAP_x    = [meas_x[0]]
        MAP_meanx = [meas_x[0]]
        MAP_varx = [var_x]
    else:
        prior_mean = MAP_meanx[i-1]
        prior_var  =1.2 * MAP_varx[i-1]   # add an expanding (forgetting) factor?
        meas_mean  = np.mean(mx)
        meas_var   = var_x             # measurements variance is a constant

        # Multiply w(r|Theta)*w(Theta)
        # See https://ccrma.stanford.edu/~jos/sasp/Product_Two_Gaussian_PDFs.html
        posterior_mu  = (meas_mean * prior_var + prior_mean*meas_var)/(meas_var + prior_var)
        posterior_var = meas_var * prior_var / (meas_var + prior_var)

        # MAP estimation = Maximum = Mean
        MAP_x.append(posterior_mu)

        # Save mean and variance for next step
        MAP_meanx.append(posterior_mu)
        MAP_varx.append(posterior_var)

MAP_y = meas_y

fig, ax = plt.subplots(frameon=False)
line_ML,   = ax.plot(ML_x, ML_y, marker='o', color='r', markersize=5)
line_MAP,  = ax.plot(MAP_x, MAP_y, marker='o', color='g', markersize=5)
line_true, = ax.plot(true_x, true_y, marker='o', color='b', linestyle='--', markersize=5)
lines = (line_ML, line_MAP, line_true)
ax.legend(['ML estimated pos.','MAP estimated pos.','True position'], loc='upper right')

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Define view parameters
ax.set_xlim(2,6)
vh = 5

# Meshgrid of indices
# xx = range(0,vh)
# tt = np.linspace(-vh, 102, 1000)
# XX, TT = np.meshgrid(xx, tt)
# YY = XX+TT

# block = amp.blocks.Line(X, Y)
# anim = amp.Animation([block], timeline) # pass in the timeline instead

# anim.controls()
# anim.save_gif('images/line4') # save animation for docs
# plt.show()


def setview(ymin):
    ax.set_ylim(ymin, ymin+vh)
    plt.draw()
    return lines

ani = animation.FuncAnimation(fig, setview, np.linspace(-vh+2, length-vh+2, 500), interval=20, repeat=False)
plt.show()

