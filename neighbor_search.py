import taichi as ti
import numpy as np


@ti.data_oriented
class NeighborSearch:
    def __init__(self,
                 neighbor_num_max,
                 num,
                 position,
                 distant,
                 topo_num,
                 angle=None,
                 velocity=None
                 ):
        self.neighbor_num_max = num
        self.num = num
        self.position = position
        self.distant0 = distant
        self.topo_num = topo_num
        self.view = angle
        self.neighbors = ti.field(int, shape=(num, neighbor_num_max))
        self.neighbors.fill(-1)
        self.neighbors_num = ti.field(int, shape=num)
        self.neighbors_num.fill(0)
        self.velocity = velocity

        # topo_search temp
        self.distant_temp = ti.field(dtype=ti.f64, shape=(num, num-1))
        self.index = ti.field(int, shape=(num, neighbor_num_max))
        self.indices = ti.field(int, shape=(num, num-1))

        # grid
        self.support_radius = self.distant0
        self.grid_size = 2 * self.support_radius
        self.grid_per_line = int(np.ceil(1.0 / self.grid_size))
        self.grid_num = self.grid_per_line ** 2
        self.max_grid_particles_num = 10000
        self.grid_particles_num = ti.field(int, shape=self.grid_num)
        self.grid_particles = ti.field(int, shape=(self.grid_num, self.max_grid_particles_num))
        self.grid_particles_num.fill(0)
        self.grid_particles.fill(-1)
        self.offset = ti.field(int, shape=(self.num, self.grid_num))
        self.width = ti.field(int, shape=self.num)
        self.count = ti.field(int, shape=self.num)

    @ti.func
    def pos_to_index(self, pos):
        ret = 0
        for i in range(2):
            j = 1 - i
            ret *= self.grid_per_line
            ret += int(pos[j] / self.grid_size)
        return ret

    @ti.kernel
    def allocate_particles_to_grid(self):
        for p in range(self.num):
            cell = self.pos_to_index(self.position[p])
            offset = ti.atomic_add(self.grid_particles_num[cell], 1)
            self.grid_particles[cell, offset] = p

    @ti.func
    def grid_around(self, cell, width, p_i):
        center_row = int(cell / self.grid_per_line)
        center_col = cell % self.grid_per_line
        for i in range(center_row - width, center_row + width + 1):
            for j in range(center_col - width, center_col + width + 1):
                row = i % self.grid_per_line
                col = j % self.grid_per_line
                index = (i - (center_row - width)) * (2 * width + 1) + (j - (center_col - width))
                self.offset[p_i, index] = row * self.grid_per_line + col

    @ti.func
    def width_check(self, width, p_i):
        center_cell = self.pos_to_index(self.position[p_i])
        cnt = 0
        radius = self.support_radius + self.grid_size * (width - 1)
        self.grid_around(center_cell, width, p_i)
        for k in range((1 + 2 * width) ** 2):
            cell = self.offset[p_i, k]
            for j in range(self.grid_particles_num[cell]):
                p_j = self.grid_particles[cell, j]
                distance = (self.position[p_i] - self.position[p_j]).norm()
                if p_i != p_j and distance < radius\
                        and self.is_in_view(p_i, p_j):
                    self.distant_temp[p_i, cnt] = distance
                    self.index[p_i, cnt] = p_j
                    cnt += 1
        return cnt

    @ti.kernel
    def grid_distant_search(self):
        for p_i in range(self.num):
            center_cell = self.pos_to_index(self.position[p_i])
            width = 1
            self.grid_around(center_cell, width, p_i)
            self.neighbors_num[p_i] = self.width_check(1, p_i)
            for cnt in range(self.neighbors_num[p_i]):
                self.neighbors[p_i, cnt] = self.index[p_i, cnt]

    def grid_distant_search_wrapped(self):
        self.allocate_particles_to_grid()
        self.grid_distant_search()

    @ti.kernel
    def grid_topo_search(self):
        for p_i in range(self.num):
            self.count[p_i] = 0
            self.width[p_i] = 0
            while self.count[p_i] < self.topo_num:
                self.width[p_i] += 1
                self.count[p_i] = self.width_check(self.width[p_i], p_i)
            self.k_smallest_with_indices(p_i, self.topo_num, self.count[p_i])
            self.neighbors_num[p_i] = self.topo_num
            for j in range(self.topo_num):
                self.neighbors[p_i, j] = self.index[p_i, self.indices[p_i, j]]

    def grid_topo_search_wrapped(self):
        self.allocate_particles_to_grid()
        self.grid_topo_search()

    @ti.kernel
    def distant_search(self, distant: ti.f64, angle: ti.f64):
        for i in range(self.num):
            cnt = 0
            for j in range(self.num):
                val = ti.Vector([ti.min(ti.abs(self.position[i].x - self.position[j].x), 1-ti.abs(self.position[i].x - self.position[j].x)),
                                 ti.min(ti.abs(self.position[i].y - self.position[j].y), 1-ti.abs(self.position[i].y - self.position[j].y))
                                 ]).norm()
                if  val < distant and i != j \
                        and self.is_in_view(i, j, angle):
                    self.neighbors[i, cnt] = j
                    cnt += 1
            self.neighbors_num[i] = cnt

    @ti.func
    def is_in_view(self, i, j, view) -> ti.i32:
        flag = 1
        pos_jx=self.position[j].x
        pos_jy=self.position[j].y
        if view > 0.0:
            if ti.abs(self.position[i].x - self.position[j].x) > 1-ti.abs(self.position[i].x - self.position[j].x):
                if self.position[i].x<self.position[j].x:
                    pos_jx = self.position[j].x-1
                else:
                    pos_jx = 1+self.position[j].x
            if ti.abs(self.position[i].y - self.position[j].y) > 1-ti.abs(self.position[i].y - self.position[j].y):
                if self.position[i].y<self.position[j].y:
                    pos_jy = self.position[j].y-1
                else:
                    pos_jy = 1+self.position[j].y
            pos_j = ti.Vector([pos_jx, pos_jy])
            pos_i = self.position[i]
            r = pos_j - pos_i
            val1 = ti.abs(ti.math.cross(self.velocity[i], r))
            val2 = ti.math.dot(self.velocity[i], r)
            angle = ti.math.atan2(val1, val2)
            if angle < view:
                flag = 1
            else:
                flag = 0
        return flag

    @ti.func
    def partition_with_indices(self, m, left, right):
        pivot_value = self.distant_temp[m, self.indices[m, right]]
        i = left - 1

        for j in range(left, right):
            if self.distant_temp[m, self.indices[m, j]] <= pivot_value:
                i += 1
                t = self.indices[m, i]
                self.indices[m, i] = self.indices[m, j]
                self.indices[m, j] = t
        t = self.indices[m, i+1]
        self.indices[m, i+1] = self.indices[m, right]
        self.indices[m, right] = t
        return i + 1

    @ti.func
    def init_indices(self, i):
        for j in range(self.num - 1):
            self.indices[i, j] = j

    @ti.func
    def k_smallest_with_indices(self, i, topo_num, size):
        left, right = 0, size - 1
        self.init_indices(i)
        while left <= right:
            pivot_index = self.partition_with_indices(i, left, right)

            if pivot_index == topo_num - 1:
                break
            elif pivot_index > topo_num - 1:
                right = pivot_index - 1
            else:
                left = pivot_index + 1

    @ti.kernel
    def topo_search(self, topo_num: ti.i32, angle: ti.f64):
        for i in range(self.num):
            cnt = 0
            for j in range(self.num):
                val = ti.Vector([ti.min(ti.abs(self.position[i].x - self.position[j].x), 1-ti.abs(self.position[i].x - self.position[j].x)),
                                 ti.min(ti.abs(self.position[i].y - self.position[j].y), 1-ti.abs(self.position[i].y - self.position[j].y))
                                 ]).norm()
                if j != i and self.is_in_view(i, j, angle) and val < self.distant0:
                    self.distant_temp[i, cnt] = val
                    self.index[i, cnt] = j
                    cnt += 1
            # if i == 0:
                # print(cnt)
            if cnt > topo_num:
                self.k_smallest_with_indices(i, topo_num, cnt)
                for j in range(topo_num):
                    self.neighbors_num[i] = topo_num
                    self.neighbors[i, j] = self.index[i, self.indices[i, j]]
            else:
                for j in range(cnt):
                    self.neighbors_num[i] = cnt
                    self.neighbors[i, j] = self.index[i, j]

