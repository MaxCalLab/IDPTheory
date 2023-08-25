##  Mike Phillips, 10/30/2021
##  Nice file for saving typical plot options
##  - mostly for internal work; _not_ for publication

import matplotlib.pyplot as plt

FINAL = True

##  Plot styles
#plt.style.use("dark_background")
plt.style.use("seaborn-notebook")
##plt.rc("lines", linewidth=1)

##  Plot font sizes
if FINAL:
    SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    LARGE_SIZE = 18
    LW = 2.0
else:
    SMALL_SIZE = 10
    MEDIUM_SIZE = 13
    LARGE_SIZE = 16
    LW = 1.0

plt.rc('font', size=MEDIUM_SIZE)        # controls default text sizes
plt.rc('axes', titlesize=LARGE_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title
plt.rc('legend', edgecolor="inherit")   # ensure legend inherits color by default
plt.rc('xtick', bottom=True, top=True, direction="in")  # ensure ticks are on all sides
plt.rc('ytick', left=True, right=True, direction="in")  #
plt.rc('axes', linewidth=LW)            # adjust AXES linewidth
plt.rc('lines', linewidth=LW)           # adjust all plotted linewidths
