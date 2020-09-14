# plot multiple lines in one figure
import matplotlib.pyplot as plt
def lines(pertable, fc, per):
    pertable['datasize'] = ['1/10', '2/10', '3/10', '4/10', '5/10',
                         '6/10', '7/10', '8/10', '9/10', '10/10']
    x1 = pertable['datasize']
    y1 = pertable[::0]
    fig1 = plt.plot(x1, y1, label = pertable.columns[0])
    y2 = pertable[::1]
    fig1 = plt.plot(x1, y2, label = pertable.columns[1])
    y3 = pertable[::2]
    fig1 = plt.plot(x1, y3, label = pertable.columns[2])
    
    fig1 = plt.xlabel('datasize')
    # Set the y axis label of the current axis.
    fig1 = plt.ylabel(f"{per}")
    fig1 = plt.title(f"{per} of algorithms based on {fc} features")
    fig1 = plt.legend()
    return fig1

import numpy as np

# Some example data to display
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('A single plot')

fig, axs = plt.subplots(1,3)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(x, y)
axs[1].plot(x, -y)
axs[2].plot(x, -y)


fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x, y)
axs[0, 0].set_title('Axis [0, 0]')
axs[0, 1].plot(x, y, 'tab:orange')
axs[0, 1].set_title('Axis [0, 1]')
axs[1, 0].plot(x, -y, 'tab:green')
axs[1, 0].set_title('Axis [1, 0]')
axs[1, 1].plot(x, -y, 'tab:red')
axs[1, 1].set_title('Axis [1, 1]')

for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Aligning x-axis using sharex')
ax1.plot(x, y)
ax2.plot(x + 1, -y)

















