#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import plotly.graph_objects as go
from opttfunc import Eggholder

n_particles = 500
max_steps = 1000
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

        # current status
        print(
            "Iteration {:>5d}: Current Best is {:.5f} at ({:.3f}, {:.3f})".format(
                i_step + 1, value_x_hat_g, x_hat_g[0], x_hat_g[1]
            )
        )
        if i_step in [0, 1, 9, 99, 999]:
            plot_current_particles(x)

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


def plot_current_particles(x: np.ndarray):

    n_grid = 500

    x_plot = np.linspace(x_lb[0] * 1.2, x_ub[0] * 1.2, n_grid)
    y_plot = np.linspace(x_lb[1] * 1.2, x_ub[1] * 1.2, n_grid)
    xx_plot, yy_plot = np.meshgrid(x_plot, y_plot)
    z_plot = test_func(
        np.concatenate(
            [np.reshape(xx_plot, (-1, 1)), np.reshape(yy_plot, (-1, 1))], axis=1
        )
    ).reshape((n_grid, n_grid))
    contour1 = go.Contour(z=z_plot, x=x_plot, y=y_plot, colorscale="gray", opacity=0.2)

    x_plot = np.linspace(x_lb[0], x_ub[0], n_grid)
    y_plot = np.linspace(x_lb[1], x_ub[1], n_grid)
    xx_plot, yy_plot = np.meshgrid(x_plot, y_plot)
    z_plot = test_func(
        np.concatenate(
            [np.reshape(xx_plot, (-1, 1)), np.reshape(yy_plot, (-1, 1))], axis=1
        )
    ).reshape((n_grid, n_grid))
    contour2 = go.Contour(z=z_plot, x=x_plot, y=y_plot, colorscale="gray", opacity=1.0)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x[:, 0],
                y=x[:, 1],
                mode="markers",
                marker=dict(size=10, color="blue"),
            ),
            go.Scatter(
                x=np.array([512]),
                y=np.array([404.2319]),
                mode="markers",
                marker=dict(size=20, color="red"),
            ),
            contour1,
            contour2,
        ]
    )
    fig.update_layout()
    fig.show()


if __name__ == "__main__":
    pso()
