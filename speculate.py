import math
import taichi as ti
import numpy as np
import cupy as cp
import time
import progressbar
from numpy.random import default_rng
from viscek import Viscek
from demo import random_vector
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def normalized(v):
    return v / v.norm()


@ti.func
def normalized_ti(v):
    return v / v.norm()


@ti.data_oriented
class Speculate:
    def __init__(self, flock, search_mode):
        self.search_mode = search_mode
        self.flock = flock
        self.nc_gauss = 0
        self.distant_gauss = 0.0
        self.nc = ti.field(dtype=ti.f64, shape=())
        # in topo search the neighbors number else the average neighbors number
        self.C_int = ti.field(dtype=ti.f64, shape=())
        self.J = ti.field(dtype=ti.f64, shape=())
        self.A = ti.field(dtype=ti.f64, shape=(self.flock.num, self.flock.num))
        self.A_np = np.ndarray((self.flock.num, self.flock.num), dtype=np.float64)
        self.n = ti.field(dtype=ti.f64, shape=(self.flock.num, self.flock.num))
        self.n.fill(0.0)
        self.entropy = ti.field(dtype=ti.f64, shape=())

    @ti.kernel
    def compute_nc(self):
        val = 0.0
        for i in range(self.flock.num):
            val += self.flock.neighbors_num[i]
        self.nc[None] = val / self.flock.num

    @ti.kernel
    def compute_J(self):
        self.J[None] = 1 / (self.nc[None] / 2 * (1 - self.C_int[None]))

    @ti.kernel
    def compute_C_int(self):
        ret = 0.0
        for i in range(self.flock.num):
            n = self.flock.neighbors_num[i]
            for index in range(n):
                j = self.flock.neighbors[i, index]
                ret += ti.math.dot(normalized_ti(self.flock.velocity[i]),
                                   normalized_ti(self.flock.velocity[j])
                                   )
        self.C_int[None] = ret / self.flock.num / self.nc[None]

    @ti.kernel
    def constract_n(self):
        for i in range(self.flock.num):
            for j in range(self.flock.num):
                self.n[i, j] = 0.0

        for i in range(self.flock.num):
            for index in range(self.flock.neighbors_num[i]):
                j = self.flock.neighbors[i, index]
                self.n[i, j] += 0.5
                self.n[j, i] += 0.5

    @ti.kernel
    def constract_A(self):
        for i in range(self.flock.num):
            for j in range(self.flock.num):
                if i == j:
                    temp = 0.0
                    for k in range(self.flock.num):
                        temp += self.n[i, k]
                    self.A[i, j] = temp - self.n[i, j]
                else:
                    self.A[i, j] = -self.n[i, j]

    def compute_entropy(self):
        A_gpu = cp.asarray(self.A_np)
        eig_gpu = cp.linalg.eigvalsh(A_gpu)
        eig = cp.asnumpy(eig_gpu)
        log_Z = self.flock.num * self.J[None] * self.nc[None] / 2
        for i in range(1, self.flock.num):
            if eig[i] > 0:
                log_Z -= np.log(self.J[None] * eig[i].real)
        self.entropy[None] = -log_Z + 0.5 * self.J[None] * self.flock.num * self.nc[None] * self.C_int[None]

    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.flock.num):
            for j in range(self.flock.num):
                np_arr[i, j] = src_arr[i, j]

    def update_flock(self, flock):
        self.flock = flock

    def change_nc_gauss(self, n):
        self.nc_gauss = n

    def change_distant_gauss(self, dis):
        self.distant_gauss = dis

    def recover_topo_num(self, n):
        self.flock.change_topo_num(n)

    def recover_distance(self, dis):
        self.flock.change_distant(dis)

    def wrapped(self):
        self.flock.change_topo_num(self.nc_gauss)
        self.flock.change_distant(self.distant_gauss)
        self.flock.get_neighbors(self.search_mode)
        self.constract_n()
        self.constract_A()
        self.copy_to_numpy(self.A_np, self.A)
        self.compute_nc()
        self.compute_C_int()
        self.compute_J()
        self.compute_entropy()


def simulate():
    viscek.random()
    for i in range(advanced_num):
        viscek.get_neighbors(search_mode)
        viscek.wrapped()
        viscek.update()
        viscek.edge()
    speculater.update_flock(viscek)


def nc_gauss(n_sim):
    for topo_num_gauss in range(begin, size + begin):
        speculater.change_nc_gauss(topo_num_gauss)
        speculater.wrapped()
        speculater.recover_topo_num(n_sim)
        entropy[topo_num_gauss - begin] = speculater.entropy[None]


def distance_gauss(dis_sim, step):
    for offset in range(size):
        dis_gauss = begin + offset * step
        speculater.change_distant_gauss(dis_gauss)
        speculater.wrapped()
        speculater.recover_distance(dis_sim)
        entropy[offset] = speculater.entropy[None]


def find_max_entropy(step):
    max_entropy = 0.0
    ret = 0
    for i in range(size):
        if entropy[i] > max_entropy:
            max_entropy = entropy[i]
            ret = i * step + begin
    return ret


def nc_speculate():
    nc_per_arg = np.zeros(sim_per_arg)
    nc_sim = np.arange(total_begin, total_begin + total_size)
    nc_mem = np.zeros(total_size, dtype=int)
    bar = progressbar.ProgressBar(max_value=total_size * sim_per_arg)

    for k in range(total_size):
        viscek.change_topo_num(nc_sim[k])

        for j in range(sim_per_arg):
            simulate()
            nc_gauss(nc_sim[k])
            nc_per_arg[j] = find_max_entropy(1)
            bar.update(k * sim_per_arg + j + 1)

        err[k] = np.std(nc_per_arg)
        nc_mem[k] = np.sum(nc_per_arg) / len(nc_per_arg)

    plt.errorbar(nc_sim, nc_mem, err, ecolor='k', elinewidth=0.5, marker='o', mfc='blue',
                 mec='k', mew=1, ms=10, alpha=1, capsize=5, capthick=3, linestyle="none")
    model = LinearRegression()
    Sim = nc_sim.reshape(-1, 1)
    Mem = nc_mem.reshape(-1, 1)
    model.fit(Sim, Mem)
    print("Coefficient (slope):", model.coef_[0][0])
    print("Intercept:", model.intercept_[0])
    plt.plot(nc_sim, model.predict(Sim), color="black", label='Regression Line')
    plt.xlabel("nc_sim")
    plt.ylabel("nc_mem")
    plt.show()


def dis_check():
    sim_dis = 0.30
    step = 0.01
    viscek.change_distant(sim_dis)
    simulate()
    distance_gauss(sim_dis, step)
    dist_sim = np.arange(begin, begin + size * step, step)
    print(find_max_entropy(step))
    plt.scatter(dist_sim, entropy)
    plt.show()


def distance_speculate():
    distance_per_arg = np.zeros(sim_per_arg)
    step = 0.02
    distance_sim = np.arange(total_begin, total_begin + total_size * step - 1e-8, step)
    distance_mem = np.zeros(total_size, dtype=float)
    bar = progressbar.ProgressBar(max_value=total_size * sim_per_arg)

    for k in range(total_size):
        viscek.change_distant(distance_sim[k])

        for j in range(sim_per_arg):
            simulate()
            distance_gauss(distance_sim[k], step)
            distance_per_arg[j] = find_max_entropy(step)
            bar.update(k * sim_per_arg + j + 1)

        err[k] = np.std(distance_per_arg)
        distance_mem[k] = np.sum(distance_per_arg) / len(distance_per_arg)
    plt.errorbar(distance_sim, distance_mem, err, ecolor='k', elinewidth=0.5, marker='o', mfc='blue',
                 mec='k', mew=1, ms=10, alpha=1, capsize=5, capthick=3, linestyle="none")
    model = LinearRegression()
    Sim = distance_sim.reshape(-1, 1)
    Mem = distance_mem.reshape(-1, 1)
    model.fit(Sim, Mem)
    print("Coefficient (slope):", model.coef_[0][0])
    print("Intercept:", model.intercept_[0])
    plt.plot(distance_sim, model.predict(Sim), color="black", label='Regression Line')
    plt.xlabel("distance_sim")
    plt.ylabel("distance_mem")
    plt.show()


if __name__ == "__main__":
    ti.init(arch=ti.gpu, random_seed=int(time.time()), default_fp=ti.f64)
    N = 1024
    advanced_num = 5000
    search_mode = 0
    total_size = 10
    total_begin = 0.10
    sim_per_arg = 50
    begin = 0.04
    size = 20
    err = np.zeros(total_size)
    rng = default_rng(seed=42)
    viscek = Viscek(N, 1e-2,
                    0.01, 0.002, 0.005, 0.008,  # r0, rb, re, ra
                    0.4, 1.0,
                    distant=0.15, topo_num=5,
                    pos=rng.random(size=(N, 2), dtype=np.float32),
                    vel=np.array([random_vector(2) for _ in range(N)], dtype=np.float32),
                    angle=2.0
                    )
    speculater = Speculate(viscek, search_mode)
    entropy = np.zeros(size)
    distance_speculate()
    # dis_check()
    # nc_speculate()
