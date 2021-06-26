#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from opttfunc import Eggholder

n_particles = 10
max_steps = 100
test_func = Eggholder()
x_ub = np.array([512, 512])
x_lb = np.array([-512, -512])

# Constants are taken from "Particle swarm optimization: developments, applications and resources"
w = 0.729
c_1 = 1.49445
c_2 = 1.49445


def pso():

    # initialize x and v with random variable
    x = (x_ub - x_lb) * np.random.rand(n_particles, test_func.dim) + x_lb
    v = (x_ub - x_lb) * np.random.rand(n_particles, test_func.dim) + x_lb

    # initialize inidivisual best
    x_hat = np.zeros((n_particles, test_func.dim))
    value_x_hat = np.inf * np.ones((n_particles,))

    # initialize global best
    x_hat_g = np.zeros((test_func.dim,))
    value_x_hat_g = np.inf

    for i_step in range(max_steps):

        # evaluate all perticles
        value = test_func(x)

        # update indivisual best
        indices_better_x = np.where(value < value_x_hat)
        x_hat[indices_better_x] = x[indices_better_x]
        value_x_hat[indices_better_x] = value[indices_better_x]

        # update global best
        idx_best_x = np.argmin(value)
        if value[idx_best_x] < value_x_hat_g:
            x_hat_g = x[idx_best_x]
            value_x_hat_g = value[idx_best_x]

        print(
            "Iteration {:>5d}: Current Best is {:.5f} at ({:.3f}, {:.3f})".format(
                i_step + 1, value_x_hat_g, x_hat_g[0], x_hat_g[1]
            )
        )

        # generate random value for update
        r_1 = np.random.rand(n_particles, 1)
        r_2 = np.random.rand(n_particles, 1)

        # compute next x, v
        x_next = np.clip(x + v, x_lb, x_ub)
        v_next = np.clip(
            w * v + c_1 * r_1 * (x_hat - x) + c_2 * r_2 * (x_hat_g - x), x_lb, x_ub
        )

        # update
        x = x_next
        v = v_next


if __name__ == "__main__":
    pso()
