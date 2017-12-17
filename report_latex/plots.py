import numpy as np
from numpy import array, sin, cos, dot, random, zeros, ones, exp
from scipy.optimize import minimize, root
from scipy.linalg import solve, norm
from scipy.integrate import simps
from scipy.interpolate import lagrange
from math import pi
import sys
import matplotlib.pyplot as plt
from matplotlib import animation,rc

from IPython.display import HTML
import matplotlib.pyplot as plt

r=0.0001
k=1
last_steps_bonus = 100
penalize_u1_factor = 1
def cost_function(x,u,i,n) : 
    if i >= (n-3):
        return r/2*(penalize_u1_factor*u[0]**2+u[1]**2) + 1 - last_steps_bonus*exp(-k*cos(x[0]) + k*cos(x[1])-2*k)
    else:
        return r/2*(penalize_u1_factor*u[0]**2+u[1]**2) + 1 - exp(-k*cos(x[0]) + k*cos(x[1])-2*k) 


grid = 100
contour_points = np.zeros([100,100])
contour_x = np.linspace(-2*pi,2*pi,100)
contour_y = np.linspace(-pi,pi,100)
for x in range(grid):
    for y in range(grid):
        xx=contour_x[x]
        yy=contour_y[y]
        contour_points[y][x]=cost_function([xx,yy],[0,0],0,7)
plt.figure(figsize=(6, 6))
CS=plt.contour(contour_x,contour_y,contour_points,20)
plt.clabel(CS, inline=1, fontsize=10)
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter([pi, -pi], [0, 0], label="global minima", c = 'purple')
plt.scatter([0], [0], label="(0, 0)", c = 'red')
plt.legend(bbox_to_anchor=(1.3, 1.05))
plt.xlabel("$q_1$")
plt.ylabel("$q_2$")
plt.show()