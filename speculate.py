import taichi as ti
import numpy as np
import cupy as cp
import time
import progressbar
from numpy.random import default_rng
from mode import Viscek
from mode import Boid
from mode import MyMode
from demo import random_vector
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.optimize import bisect
import scipy.optimize as opt


def normalized(v):
    return v / v.norm()


@ti.func
def normalized_ti(v):
    return v / v.norm()


@ti.data_oriented
class Speculate:
    def __init__(self, flock, search):
        self.search_mode = search
        self.flock = flock
        self.nc_gauss = flock.topo_num
        self.distant_gauss = flock.distant
        self.angle_gauss = flock.angle
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
    def compute_C_int_angle(self):
        ret = 0.0
        for i in range(self.flock.num):
            n = self.flock.neighbors_num[i]
            for index in range(n):
                j = self.flock.neighbors[i, index]
                ret += ti.math.dot(normalized_ti(self.flock.velocity[i]),
                                   normalized_ti(self.flock.position[j] - self.flock.position[i])
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
                self.n[i, j] = 1.0

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
        for i in range(self.flock.num):
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

    def change_angle_gauss(self, angle):
        self.angle_gauss = angle

    def recover_topo_num(self, n):
        self.flock.change_topo_num(n)

    def recover_distance(self, dis):
        self.flock.change_distant(dis)

    def recover_angle(self, ang):
        self.flock.change_angle(ang)

    def wrapped(self):
        self.flock.change_topo_num(self.nc_gauss)
        self.flock.change_distant(self.distant_gauss)
        self.flock.change_angle(self.angle_gauss)
        self.flock.get_neighbors(self.search_mode)
        self.constract_n()
        self.constract_A()
        self.copy_to_numpy(self.A_np, self.A)
        self.compute_nc()
        self.compute_C_int_angle()
        self.compute_J()
        self.compute_entropy()


@ti.data_oriented
class MySpeculate:
    def __init__(self, mymode, search):
        self.search_mode = search
        self.mymode = mymode
        self.nc_gauss = mymode.topo_num
        self.distant_gauss = mymode.distant
        self.angle_gauss = mymode.angle
        self.Q0 = ti.field(dtype=ti.f64, shape=())
        self.Q3 = ti.field(dtype=ti.f64, shape=())
        self.J = ti.field(dtype=ti.f64, shape=())
        self.g = ti.field(dtype=ti.f64, shape=())
        self.A = ti.field(dtype=ti.f64, shape=(self.mymode.num))
        self.N = ti.field(dtype=ti.f64, shape=(self.mymode.num, self.mymode.num))
        self.N_np = np.ndarray((self.mymode.num, self.mymode.num), dtype=np.float64)
        self.eig_N = np.ndarray((self.mymode.num), dtype=np.float64)
        self.n = ti.field(dtype=ti.f64, shape=(self.mymode.num, self.mymode.num))
        self.m = ti.field(dtype=ti.f64, shape=(self.mymode.num, self.mymode.num))
        self.n.fill(0.0)
        self.m.fill(0.0)
        self.A.fill(0.0)
        self.entropy = ti.field(dtype=ti.f64, shape=())
   

    @ti.kernel
    def compute_Q0(self):
        self.Q0[None] = 0.0
        #for i in range(self.mymode.num):
        #    for j in range(self.mymode.num):
        #        self.Q0[None] += self.n[i,j] * \
        #            (self.mymode.velocity[i] - self.mymode.velocity[j]).norm() ** 2 \
        #            /(4*self.mymode.v0 ** 2 *self.mymode.num*self.mymode.topo_num)
        
        for i in range(self.mymode.num):
            self.Q0[None] += self.A[i]*(self.mymode.velocity[i].norm() - self.mymode.velocity[i].y)/self.mymode.v0/(self.mymode.num*self.mymode.topo_num)
        # print("Q0: ", self.Q0[None])

    @ti.kernel
    def compute_Q3(self):
        self.Q3[None]=0.0
        for i in range(self.mymode.num):
            self.Q3[None] += 1/self.mymode.num/self.mymode.v0**2*(self.mymode.velocity[i].norm()-self.mymode.v0)**2
        # print("Q3: ",self.Q3[None])


    @ti.kernel
    def constract_n(self):
        for i in range(self.mymode.num):
            for j in range(self.mymode.num):
                self.n[i, j] = 0.0

        for i in range(self.mymode.num):
            for index in range(self.mymode.neighbors_num[i]):
                j = self.mymode.neighbors[i, index]
                self.n[i, j] += 0.5
                self.n[j, i] += 0.5
    
    @ti.kernel
    def constract_m(self):
        for i in range(self.mymode.num):
            for j in range(self.mymode.num):
                self.m[i, j] = 0.0

        for i in range(self.mymode.num):
            for index in range(self.mymode.neighbors_num[i]):
                j = self.mymode.neighbors[i, index]
                self.m[i, j] = 1.0
    
    @ti.kernel
    def constract_N(self):
        for i in range(self.mymode.num):
            for j in range(self.mymode.num):
                if i == j:
                    temp = 0.0
                    for k in range(self.mymode.num):
                        temp += self.n[i, k]
                    self.N[i, j] = temp - self.n[i, j]
                else:
                    self.N[i, j] = -self.n[i, j]

    @ti.kernel
    def compute_A(self):
        for i in range(self.mymode.num):
            self.A[i]= self.mymode.topo_num + 1
        
        for i in range(self.mymode.num):
            for j in range(self.mymode.num):
                self.A[i] -= self.m[i, j]
    
    def compute_eig(self):
        N_gpu = cp.asarray(self.N_np)
        eig_gpu = cp.linalg.eigvalsh(N_gpu)
        self.eig_N = cp.asnumpy(eig_gpu)

    def func(self, J):
        val = 0.0
        for a in range(2, N):
            print(self.eig_N[a].real, self.Q3[None], self.mymode.topo_num, self.Q0[None])
            val += 1/(3+J*(self.eig_N[a].real*self.Q3[None] - 2*self.mymode.topo_num*self.Q0[None]))
        val -= self.mymode.num
        return val
    
    def compute_Jg(self):
        J = bisect(self.func, 0.1, 10)
        self.J[None] = J
        self.g[None] = (3-2*self.mymode.topo_num*self.J[None]*self.Q0[None])/self.Q3[None]
        print("J_guass :", self.J[None], "  g guass", self.g[None])
        print(self.func(1))

    def compute_Jg_simple(self):
        self.J[None] = 1/2/self.mymode.topo_num/self.Q0[None]
        self.g[None] = 1/self.Q3[None]
        print("J_guess: ", self.J, "g_guess: ", self.g)
        print(self.angle_gauss)


    def compute_entropy_simple(self):
        self.entropy[None] = 0.0
        for i in range(self.mymode.num):
            if self.A[i] > 0:
                self.entropy[None] += 0.5*ti.log(self.J[None]*(self.A[i]))
        for i in range(2, self.mymode.num):
            self.entropy[None] += 0.5*ti.log(self.g[None])
        self.entropy[None] -= self.J[None]*self.mymode.topo_num*self.mymode.num*self.Q0[None]
        self.entropy[None] -= self.g[None]*self.mymode.num/2*self.Q3[None]

    def compute_entropy(self):
        self.entropy[None] = 0.0
        for i in range(self.mymode.num):
            self.entropy[None] += 0.5*ti.log(self.J[None]*(self.A[i]+self.eig_N[i].real))
        for i in range(2, self.mymode.num):
            self.entropy[None] += 0.5*ti.log(self.g[None]+self.J[None]*self.eig_N[i].real)
        self.entropy[None] -= self.J[None]*self.mymode.topo_num*self.mymode.num*self.Q0[None]
        self.entropy[None] -= self.g[None]*self.mymode.num/2*self.Q3[None]

    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.mymode.num):
            for j in range(self.mymode.num):
                np_arr[i, j] = src_arr[i, j]

    def update_flock(self, mymode):
        self.mymode = mymode

    def change_nc_gauss(self, n):
        self.nc_gauss = n

    def change_distant_gauss(self, dis):
        self.distant_gauss = dis

    def change_angle_gauss(self, angle):
        self.angle_gauss = angle

    def recover_topo_num(self, n):
        self.mymode.change_topo_num(n)

    def recover_distance(self, dis):
        self.mymode.change_distant(dis)

    def recover_angle(self, ang):
        self.mymode.change_angle(ang)

    def wrapped(self):
        self.mymode.change_topo_num(self.nc_gauss)
        self.mymode.change_distant(self.distant_gauss)
        self.mymode.change_angle(self.angle_gauss)
        self.mymode.get_neighbors(self.search_mode)
        self.constract_m()
        self.constract_n()
        self.compute_A()
        self.constract_N()
        self.copy_to_numpy(self.N_np, self.N)
        self.compute_Q0()
        self.compute_Q3()
        self.compute_eig()
        self.compute_Jg_simple()
        self.compute_entropy_simple()


def simulate(mode):
    mode.random()
    for i in range(advanced_num):
        mode.get_neighbors(search_mode)
        mode.wrapped()
        mode.update()
        mode.edge()
    speculater.update_flock(mode)


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


def angle_gauss(ang_sim, step_gauss):
    for offset in range(size):
        ang_gauss = begin + offset * step_gauss
        speculater.change_angle_gauss(ang_gauss)
        speculater.wrapped()
        speculater.recover_angle(ang_sim)
        entropy[offset] = speculater.entropy[None]


def find_max_entropy(step):
    max_entropy = -10000.0
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
    dist_sim = np.arange(begin, begin + size * step - 1e-8, step)
    print(find_max_entropy(step))
    plt.scatter(dist_sim, entropy)
    plt.xlabel("distance_sim")
    plt.ylabel("entropy")
    plt.show()


def angle_check(mode):
    step_gauss = 0.05
    sim_ang = 1.0
    mode.change_angle(sim_ang)
    simulate(mode)
    angle_gauss(sim_ang, step_gauss)
    an_sim = np.arange(begin, begin + size * step_gauss, step_gauss)
    print(find_max_entropy(step_gauss))
    plt.scatter(an_sim, entropy)
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


def angle_speculate(mode):
    angle_per_arg = np.zeros(sim_per_arg)
    step = 0.1
    step_gauss = 0.05
    angle_sim = np.arange(total_begin, total_begin + total_size * step - 1e-8, step)
    angle_mem = np.zeros(total_size, dtype=float)
    bar = progressbar.ProgressBar(max_value=total_size * sim_per_arg)

    for k in range(total_size):
        mode.change_angle(angle_sim[k])

        for j in range(sim_per_arg):
            simulate(mode)
            angle_gauss(angle_sim[k], step_gauss)
            angle_per_arg[j] = find_max_entropy(step_gauss)
            bar.update(k * sim_per_arg + j + 1)

        err[k] = np.std(angle_per_arg)
        angle_mem[k] = np.sum(angle_per_arg) / len(angle_per_arg)
    print("sim", angle_sim, "mem", print(angle_mem), "err", print(err))
    plt.errorbar(angle_sim, angle_mem, err, ecolor='k', elinewidth=0.5, marker='o', mfc='blue',
                 mec='k', mew=1, ms=10, alpha=1, capsize=5, capthick=3, linestyle="none")
    model = LinearRegression()
    Sim = angle_sim.reshape(-1, 1)
    Mem = angle_mem.reshape(-1, 1)
    model.fit(Sim, Mem)
    print("Coefficient (slope):", model.coef_[0][0])
    print("Intercept:", model.intercept_[0])
    plt.plot(angle_sim, model.predict(Sim), color="black", label='Regression Line')
    plt.xlabel("angle_sim")
    plt.ylabel("angle_mem")
    plt.show()

def angle_hist(mode):
    angle_per_arg = np.zeros(sim_per_arg)
    step = 0.1
    step_gauss = 0.05
    angle_sim = 1.5
    bar = progressbar.ProgressBar(max_value=sim_per_arg)

    mode.change_angle(angle_sim)

    for j in range(sim_per_arg):
        simulate(mode)
        angle_gauss(angle_sim, step_gauss)
        angle_per_arg[j] = find_max_entropy(step_gauss)
        bar.update(j + 1)
    print(angle_per_arg)
    plt.hist(angle_per_arg, edgecolor='black')
    plt.show()

if __name__ == "__main__":
    ti.init(arch=ti.gpu, random_seed=int(time.time()), default_fp=ti.f64)
    N = 1024
    advanced_num = 5000
    search_mode = 1
    total_size = 12
    total_begin = 1.0
    sim_per_arg = 60
    begin = 0.1
    size = 60
    err = np.zeros(total_size)
    rng = default_rng(seed=42)
    viscek = Viscek(N, 1e-2,
                    0.01, 0.002, 0.005, 0.008,  # r0, rb, re, ra
                    0.4, 1.0,
                    distant=0.1, topo_num=20,
                    pos=rng.random(size=(N, 2), dtype=np.float32),
                    vel=np.array([random_vector(2) for _ in range(N)], dtype=np.float32),
                    angle=3.0
                    )
    boid = Boid(N, 1e-2,
                2.0, 2.0, 2.0,
                1, 0.5,
                distant=0.05, topo_num=20,
                pos=rng.random(size=(N, 2), dtype=np.float32),
                vel=np.array([random_vector(2) for _ in range(N)], dtype=np.float32),
                angle=3.0
                )
    myMode = MyMode(N, 5e-3,
                        1, 0,
                        10, 1.0,
                        distant=0.1, topo_num=50,
                        pos=rng.random(size=(N, 2), dtype=np.float32),
                        vel=np.array([random_vector(2)*1.0 for _ in range(N)], dtype=np.float32),
                        angle=1.0
                        )
    mode = myMode
    speculater = MySpeculate(mode, search_mode)
    entropy = np.zeros(size)
    angle_check(mode)
    # distance_speculate()
    # angle_speculate(mode)
    # angle_hist(mode)
    # dis_check()
    # nc_speculate()
