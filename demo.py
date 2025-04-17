import taichi as ti
import numpy as np
import time
import random
import math
from numpy.random import default_rng
from mode import Boid
from mode import Viscek
from mode import MyMode
from mode import myViscek


def random_vector(n):
    components = [np.random.normal() for _ in range(n)]
    r = np.sqrt(sum(x * x for x in components))
    v = np.array([x / r for x in components])
    return v



if __name__ == "__main__":
    ti.init(arch=ti.gpu,  random_seed=int(time.time()), default_fp=ti.f64)

    WINDOW_HEIGHT = 540
    AR = 1
    WINDOW_WIDTH = AR * WINDOW_HEIGHT

    N = 2000

    gui = ti.GUI("flocking behavior", res=(WINDOW_WIDTH, WINDOW_HEIGHT))

    rule = 2
    search_mode = 0

    if rule == 0:
        rng = default_rng(seed=42)
        boid = Boid(N, 1e-2,
                    1.0, 1.0, 1.0,
                    1, 0.5,
                    distant=0.15, topo_num=20,
                    pos=rng.random(size=(N, 2), dtype=np.float32),
                    vel=np.array([random_vector(2) for _ in range(N)], dtype=np.float32),
                    angle=1.0
                    )
        i=0
        while gui.running:
            boid.get_neighbors(search_mode)
            boid.wrapped()
            boid.update()
            boid.edge()
            boid.render(gui, AR, filename=f'{i:05d}.png')
            i=i+1
    elif rule == 1:
        rng = default_rng(seed=42)
        viscek = Viscek(N, 1,
                        0.01, 0.002, 0.005, 0.008,  # r0, rb, re, ra
                        0.4, 1.0,
                        distant=0.2, topo_num=139,
                        pos=rng.random(size=(N, 2), dtype=np.float32),
                        vel=np.array([random_vector(2)*0.01 for _ in range(N)], dtype=np.float32),
                        angle=2.5
                        )

        while gui.running:
            viscek.get_neighbors(search_mode)
            viscek.wrapped()
            viscek.update()
            viscek.edge()
            viscek.render(gui, AR)
    elif rule == 2:
        v3 = 1.0
        rng = default_rng(seed=42)
        myMode = MyMode(N, 1e-3,
                        30.0, 0, 
                        1000, v3,
                        distant=0.1, topo_num=60,
                        pos=rng.random(size=(N, 2), dtype=np.float32),
                        vel=np.array([random_vector(2) * v3 for _ in range(N)], dtype=np.float32),
                        angle=2.5
                        )
        while gui.running:
            myMode.step()
            myMode.render(gui, AR)
    
    elif rule == 3:
        rng = default_rng(seed=42)
        myviscek = myViscek(N, 1,
                        1.0, 20,
                        distant=0.1,topo_num=20,
                        pos=rng.random(size=(N, 2), dtype=np.float32),
                        vel=np.array([random_vector(2)*0.01 for _ in range(N)], dtype=np.float32),
                        angle=0.5
                        )

        while gui.running:
            myviscek.get_neighbors(search_mode)
            myviscek.wrapped()
            myviscek.update()
            myviscek.edge()
            myviscek.render(gui, AR)
