import taichi as ti
import numpy as np
from numpy.random import default_rng
import time
from mode import MyMode
from demo import random_vector
import matplotlib.pyplot as plt

def get_orientation(mode, J1, J2):
    mode.random()
    mode.J1 = J1
    mode.J2 = J2
    tmp = []
    for i in range(3000):
        mode.step()

    for i in range(5):
        tmp.append(mode.orientation())
        for i in range(500):
            mode.step()
    return sum(tmp) / len(tmp)

def get_density(mode, J1, J2):
    mode.random()
    mode.J1 = J1
    mode.J2 = J2
    tmp = []
    for i in range(3000):
        mode.step()

    for i in range(5):
        tmp.append(mode.density())
        for i in range(500):
            mode.step()
    
    return sum(tmp) / len(tmp)

def J2_phase():
    J2_list = np.arange(0.5, 6.5, 0.5)
    for J2 in J2_list:
        val = get_orientation(myMode, myMode.J1, J2)
        orientation.append(val)

    plt.plot(J2_list, orientation, 'bo')
    plt.xlabel("J2")
    plt.ylabel("orientation")
    plt.show()

def J1_phase():
    J1_list = np.arange(0.0, 2.01, 0.2)
    for J1 in J1_list:
        val = get_orientation(myMode, J1, myMode.J2)
        orientation.append(val)

    plt.plot(J1_list, orientation, 'bo')
    plt.xlabel("J1")
    plt.ylabel("orientation")
    plt.show()

def J2_phase_d():
    J2_list = np.arange(5.5, 6.5, 0.5)
    for J2 in J2_list:
        val = get_density(myMode, myMode.J1, J2)
        print(val)
        density.append(val)

    plt.plot(J2_list, density, 'bo')
    plt.plot(J2_list, density)
    plt.xlabel("J2")
    plt.ylabel("density")
    plt.show()

if __name__ == "__main__":
    ti.init(arch=ti.gpu, random_seed=int(time.time()), default_fp=ti.f64)
    v = 1.0
    N = 1000
    rng = default_rng(seed=42)
    myMode = MyMode(N, 5e-3,
                    0.0, 0.0, 
                    100, v,
                    distant=0.2, topo_num=30,
                    pos=rng.random(size=(N, 2), dtype=np.float32),
                    vel=np.array([random_vector(2) * v for _ in range(N)], dtype=np.float32),
                    angle=0.5
                    )
    density = []
    orientation = []
    J1_phase()
    
    
    
    