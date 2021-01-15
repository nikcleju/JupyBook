from itertools import islice

import math
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import scipy.stats


np.random.seed(123)

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


#===============
# Setup
#===============

# Problem parameters
x_start = 4         # Starting x position
y_start = 0         # Starting y position
speed_x = 0         # Speed along x direction
speed_y = 1         # Speed along x direction
timestep = 1
var_x = 0.5         # Noise variance for x measurement
var_y = 0           # Noise variance for y measurement
timelength = 50     # Total number of timesteps


# Initialize data
x_true_list = []          # True position
y_true_list = []  
x_meas_list = []          # Measurements (noisy)
y_meas_list = []
x_ML_list = []            # ML estimations
y_ML_list = []
x_MAP_list = []           # ML estimations
y_MAP_list = []

# Run at every time step
for i in range(timelength):
    
    if i == 0:
        # Initially, true position = starting position
        x_true = x_start
        y_true = y_start
    else:
        # Current true position = Last true position + speed * timestep
        x_true = x_true_list[-1] + speed_x * timestep
        y_true = y_true_list[-1] + speed_y * timestep

    # Take measurement of position = true position + noise
    x_meas = x_true + math.sqrt(var_x) * np.random.randn()
    y_meas = y_true + math.sqrt(var_y) * np.random.randn()

    # ML estimation of current position = average of the current measurements
    # If we have a single measurement, that value it is
    x_ML = np.mean(x_meas)
    y_ML = np.mean(y_meas)

    # Bayesian estimation of position, only for x axis, using Kalman filter, 
    if i == 0:
        # At the initial time, we have no prior, so we can use only the measurements (ML estimation)
        x_MAP = x_ML
        post_x_mean = [x_ML]
        post_x_var  = [var_x]

    else:
        # Apply Bayesian estimation, using measurements and a prior

        # Get mean and variance of the prior w(Theta) from the last step
        x_prior_mean = post_x_mean[i-1] + speed_x * timestep   # moved with speed_x 
        x_prior_var  = post_x_var[i-1]                         # TODO: add an expanding (forgetting) factor?

        # Get the likelihood w(r|Theta) mean and variance
        x_meas_mean  = x_ML
        x_meas_var   = var_x             # measurement' variance is a constant

        # Compute posterior w(Theta|r) = multiply w(r|Theta) * w(Theta)
        # A Gaussian * a Gaussian = a Gaussian, with resulting mean and variance given below
        # See https://ccrma.stanford.edu/~jos/sasp/Product_Two_Gaussian_PDFs.html
        post_mean  = (x_meas_mean * x_prior_var + x_prior_mean * x_meas_var)/(x_meas_var + x_prior_var)
        post_var   = x_meas_var * x_prior_var / (x_meas_var + x_prior_var)

        # MAP estimation = Maximum of posterior distribution = The mean, if it is a Gaussian distribution
        x_MAP = post_mean

        # Save posterior parameters for next step
        post_x_mean.append(post_mean)     # Mean and variance of posterior 
        post_x_var.append(post_var)       #  to use at the next step        

    # We don't do MAP estimation for y direction, yet
    y_MAP = y_ML

    # Save data from all time steps in lists 
    x_true_list.append(x_true)  # True position
    y_true_list.append(y_true)

    x_meas_list.append(x_meas)  # Measurements
    y_meas_list.append(y_meas)

    x_ML_list.append(x_ML)      # ML estimations
    y_ML_list.append(y_ML)

    x_MAP_list.append(x_MAP)    # MAP estimations
    y_MAP_list.append(y_MAP)



#=========================
# Animation
#=========================

# Draw full figure
fig, ax = plt.subplots(frameon=False)
line_ML,   = ax.plot(x_ML_list,   y_ML_list,   marker='o', color='r', markersize=5)
line_MAP,  = ax.plot(x_MAP_list,  y_MAP_list,  marker='o', color='g', markersize=5)
line_true, = ax.plot(x_true_list, y_true_list, marker='o', color='b', markersize=5, linestyle='--')
lines = (line_ML, line_MAP, line_true)
ax.legend(['ML estimated pos.','MAP estimated pos.','True position'], loc='upper right')
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Define view parameters
ax.set_xlim(2,6)
vh = 5

# Draw car icon
img = mpimg.imread('caricon.png')
a=plt.axes([0.47, 0.4, 0.08, 0.2], frameon=False)
plt.xticks([])
plt.yticks([])
plt.imshow(img)

def setview(ymin):
    ax.set_ylim(ymin, ymin+vh)
    plt.draw()
    return lines

# Define and run animation
y_max = y_start + speed_y*timelength
ani = animation.FuncAnimation(fig, setview, np.linspace(-vh+2, y_max-vh+2, 300), interval=20, repeat=False)
plt.show()

# Save animation
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani.save('MLvsMAPposition.mp4', writer=writer)


#=========================
# Explanation figure 1
#=========================

# # Draw figure again
# fig, ax = plt.subplots()
# line_ML,   = ax.plot(x_ML_list,  y_ML_list,   marker='o', color='r', linestyle=' ',  markersize=5)
# line_MAP,  = ax.plot(x_MAP_list,  y_MAP_list,  marker='o', color='g', linestyle=' ',  markersize=5)
# line_true, = ax.plot(x_true_list, y_true_list, marker='o', color='b', linestyle='--', markersize=5)
# lines = (line_ML, line_MAP, line_true)

# ax.legend(['ML estimated pos.','MAP estimated pos.','True position'], loc='upper right')
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

# # Define view parameters
# ax.set_xlim(2,6)
# vh = 5
# ax.set_ylim(-0.25,1.75)
# #plt.yticks([])
# plt.title('First time moment')

# # Show only first measurement
# line_ML.set_data(x_ML_list[0],   y_ML_list[0])
# #line_MAP.set_data(x_MAP_list[], y_MAP_list[])
# line_MAP.set_data([], [])

# # # Show true distribution 
# # # Add smaller plot on top: see https://matplotlib.org/examples/pylab_examples/axes_demo.html
# # a1 = plt.axes([.411, .13, .2, .2], frameon=False)
# # x_axis = np.arange(-3, 3, 0.01)
# # plt.plot(x_axis, scipy.stats.norm.pdf(x_axis,0,1))
# # plt.xticks([])
# # plt.yticks([])
# # ax.annotate('True distribution', xy=(4.2, 0.2), xytext=(4.6, 0.3),
# #             arrowprops=dict(facecolor='black', width=0.5, shrink=0.00),
# #             )

# plt.savefig('MLvsMAPposition_fig1.png')
# plt.show()



#=========================
# Explanation figure 2
# Add smaller plot on top: see https://matplotlib.org/examples/pylab_examples/axes_demo.html
#=========================

# # Draw figure again
# fig, ax = plt.subplots()
# line_ML,   = ax.plot(x_ML_list,  y_ML_list,   marker='o', color='r', linestyle=' ',  markersize=5)
# line_MAP,  = ax.plot(x_MAP_list,  y_MAP_list,  marker='o', color='g', linestyle=' ',  markersize=5)
# line_true, = ax.plot(x_true_list, y_true_list, marker='o', color='b', linestyle='--', markersize=5)
# lines = (line_ML, line_MAP, line_true)

# ax.legend(['ML estimated pos.','MAP estimated pos.','True position'], loc='upper right')
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

# # Define view parameters
# ax.set_xlim(2,6)
# vh = 5
# ax.set_ylim(-0.25,1.75)
# plt.title('Second time moment')

# # Show only first measurement
# line_ML.set_data(x_ML_list[:2],   y_ML_list[:2])
# line_MAP.set_data(x_MAP_list[1], y_MAP_list[1])

# plt.savefig('MLvsMAPposition_fig2.png')
# plt.show()