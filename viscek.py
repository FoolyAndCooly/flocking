import taichi as ti
import sys
import numpy as np
from numpy.random import default_rng
from flock import Flock
import math


@ti.func
def random_vector():
    angle = ti.random(ti.f32) * 2 * math.pi
    v = ti.Vector([ti.cos(angle), ti.sin(angle)])
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

