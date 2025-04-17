import taichi as ti
import sys
import numpy as np
from numpy.random import default_rng
from flock import Flock
import math



@ti.func
def random_vector(var=1):
    val = ti.sqrt(var)
    # angle = ti.random(ti.f32) * 2 * math.pi
    v = ti.Vector([val * ti.randn(), val * ti.randn()])
    return v


@ti.func
def normalized(v):
    return v / v.norm()


@ti.data_oriented
class Viscek(Flock):
    def __init__(self, num, dt, r0, rb, re, ra, J, beta, distant=None, topo_num=None, pos=None, vel=None, acc=None, angle=None):
        super().__init__(num, dt, distant, topo_num, pos, vel, acc, angle)
        self.v0 = ti.field(ti.f64, shape=())
        self.r0 = r0
        self.rb = rb
        self.re = re
        self.ra = ra
        self.J = J
        self.beta = beta
        self.forward_vel = ti.Vector.field(n=2, dtype=ti.f64, shape=self.num)

    @ti.kernel
    def check(self):
        norm = self.velocity[0].norm()
        self.v0[None] = norm

    @ti.kernel
    def clear_acc(self):
        for i in range(self.num):
            self.acceleration[i] = ti.Vector([0.0 for _ in range(2)])

    @ti.func
    def compute_distant_force(self, r_norm) -> ti.f64:
        val = 0.0
        if r_norm < self.rb:
            val = -10000
        elif self.rb < r_norm < self.ra:
            val = (r_norm - self.re)/4/(self.ra - self.re)
        elif self.ra < r_norm < self.r0:
            val = 1.0
        return val

    @ti.kernel
    def compute_vel(self):
        for i in range(self.num):
            sum_vel = ti.Vector([0.0 for _ in range(2)])
            sum_force = ti.Vector([0.0 for _ in range(2)])
            n = self.neighbors_num[i]
            for index in range(n):
                j = self.neighbors[i, index]
                sum_vel += self.velocity[j]
                r = self.position[j] - self.position[i]
                sum_force += self.compute_distant_force(r.norm()) * normalized(r)
            self.forward_vel[i] = self.v0[None] * normalized(
                self.J / self.v0[None] * sum_vel +
                self.beta * sum_force +
                random_vector())

        for i in range(self.num):
            self.velocity[i] = self.forward_vel[i]

    def wrapped(self):
        self.check()
        self.clear_acc()
        self.compute_vel()

@ti.data_oriented
class myViscek(Flock):
    def __init__(self, num, dt, J, nc, distant=None, topo_num=None, pos=None, vel=None, acc=None, angle=None):
        super().__init__(num, dt, distant, topo_num, pos, vel, acc, angle)
        self.v0 = ti.field(ti.f64, shape=())
        self.J = J
        self.nc = nc
        self.forward_vel = ti.Vector.field(n=2, dtype=ti.f64, shape=self.num)

    @ti.kernel
    def check(self):
        norm = self.velocity[0].norm()
        self.v0[None] = norm

    @ti.kernel
    def clear_acc(self):
        for i in range(self.num):
            self.acceleration[i] = ti.Vector([0.0 for _ in range(2)])

    @ti.kernel
    def compute_vel(self):
        for i in range(self.num):
            n = self.neighbors_num[i]
            self.forward_vel[i] = self.v0[None] * normalized(
                self.J * (self.nc - n) * ti.Vector([0.0, 1.0]) +
                random_vector())

        for i in range(self.num):
            self.velocity[i] = self.forward_vel[i]

    def wrapped(self):
        self.check()
        self.clear_acc()
        self.compute_vel()

@ti.func
def set_mag(v, mag):
    return (v / v.norm()) * mag


@ti.func
def limit(v, mag):
    norm = v.norm()
    return (v / norm) * mag if norm > 0 and norm > mag else v


@ti.data_oriented
class Boid(Flock):
    def __init__(self, num, dt, ali, sep, coh, max_spd, max_acc, distant=None, topo_num=None, pos=None, vel=None, acc=None, angle=None):
        super().__init__(num, dt, distant, topo_num, pos, vel, acc, angle)
        self.ali = ali
        self.sep = sep
        self.coh = coh
        self.max_spd = max_spd
        self.max_acc = max_acc

    @ti.func
    def clear_acc(self):
        for i in range(self.num):
            self.acceleration[i] = ti.Vector([0.0 for _ in range(2)])

    @ti.kernel
    def compute_force(self):
        self.clear_acc()
        for i in range(self.num):
            alignment = ti.Vector([0.0 for _ in range(2)])
            separation = ti.Vector([0.0 for _ in range(2)])
            cohesion = ti.Vector([0.0 for _ in range(2)])
            n = self.neighbors_num[i]
            for index in range(n):
                j = self.neighbors[i, index]
                alignment += self.velocity[j]
                separation += (self.position[i] - self.position[j]) / self.distant
                cohesion += self.position[j]
            if n > 0:
                alignment = limit(
                    set_mag((alignment / n), self.max_spd) - self.velocity[i],
                    self.max_acc) * self.ali
                separation = limit(
                    set_mag((separation / n), self.max_spd) - self.velocity[i],
                    self.max_acc) * self.sep
                cohesion = limit(
                    set_mag(((cohesion / n) - self.position[i]), self.max_spd) -
                    self.velocity[i], self.max_acc) * self.coh
                self.acceleration[i] += alignment
                self.acceleration[i] += separation
                self.acceleration[i] += cohesion
                self.velocity[i] = limit(self.velocity[i], self.max_spd)

    def wrapped(self):
        self.compute_force()



@ti.data_oriented
class MyMode(Flock):
    def __init__(self, num, dt, J1, J2, g, v0, distant=None, topo_num=None, pos=None, vel=None, acc=None, angle=None):
        super().__init__(num, dt, distant, topo_num, pos, vel, acc, angle)
        self.J1 = J1
        self.J2 = J2
        self.g = g
        self.v0 = v0
        self.dt = dt
    
    @ti.kernel
    def compute_force(self, J1:ti.f64, J2:ti.f64):
        for i in range(self.num):
            F0 = J1/self.v0 * (self.topo_num+1 - self.neighbors_num[i])*(ti.Vector([0, 1.0]) - normalized(self.velocity[i]))
            F1 = self.g/self.v0 ** 2 * (self.v0-self.velocity[i].norm()) * normalized(self.velocity[i])
            F2 = random_vector(2 / self.dt)
            # F3 = ti.Vector([0.0, 0.0])
            # for index in range(self.neighbors_num[i]):
            #    j = self.neighbors[i, index]
            #    F3 += J2/2/self.v0 * (normalized(self.velocity[j])-normalized(self.velocity[i]))
            self.acceleration[i] = F0 + F1 + F2


    def wrapped(self):
        self.compute_force(self.J1, self.J2)

    def step(self):
        self.get_neighbors(1)
        self.wrapped()
        self.update()
        self.edge()
        