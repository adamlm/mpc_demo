import casadi
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import time


obs_x = 7.5
obs_y = 5
obs_diam = 1


def simulate(cat_states, cat_controls, t, step_horizon, N, reference, save=False):
    def create_triangle(state=[0,0,0], h=1, w=0.5, update=False):
        x, y, th = state
        triangle = np.array([
            [h, 0   ],
            [0,  w/2],
            [0, -w/2],
            [h, 0   ]
        ]).T
        rotation_matrix = np.array([
            [np.cos(th), -np.sin(th)],
            [np.sin(th),  np.cos(th)]
        ])

        coords = np.array([[x, y]]) + (rotation_matrix @ triangle).T
        if update == True:
            return coords
        else:
            return coords[:3, :]

    def init():
        return path, horizon, current_state, target_state,

    def animate(i):
        # get variables
        x = cat_states[0, 0, i]
        y = cat_states[1, 0, i]
        th = cat_states[2, 0, i]

        # update path
        if i == 0:
            path.set_data(np.array([]), np.array([]))
        x_new = np.hstack((path.get_xdata(), x))
        y_new = np.hstack((path.get_ydata(), y))
        path.set_data(x_new, y_new)

        # update horizon
        x_new = cat_states[0, :, i]
        y_new = cat_states[1, :, i]
        horizon.set_data(x_new, y_new)

        # update current_state
        current_state.set_xy(create_triangle([x, y, th], update=True))

        # update target_state
        # xy = target_state.get_xy()
        # target_state.set_xy(xy)

        return path, horizon, current_state, target_state,

    # create figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))
    min_scale = min(reference[0], reference[1], reference[3], reference[4]) - 2
    max_scale = max(reference[0], reference[1], reference[3], reference[4]) + 2
    ax.set_xlim(left = min_scale, right = max_scale)
    ax.set_ylim(bottom = min_scale, top = max_scale)

    circle = plt.Circle((obs_x, obs_y), obs_diam/2, color='r')
    ax.add_patch(circle)

    # create lines:
    #   path
    path, = ax.plot([], [], 'k', linewidth=2)
    #   horizon
    horizon, = ax.plot([], [], 'x-g', alpha=0.5)
    #   current_state
    current_triangle = create_triangle(reference[:3])
    current_state = ax.fill(current_triangle[:, 0], current_triangle[:, 1], color='r')
    current_state = current_state[0]
    #   target_state
    target_triangle = create_triangle(reference[3:])
    target_state = ax.fill(target_triangle[:, 0], target_triangle[:, 1], color='b')
    target_state = target_state[0]

    sim = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=len(t),
        interval=step_horizon*100,
        # interval=3000,
        blit=True,
        repeat=True
    )
    plt.show()

    # if save == True:
    #     sim.save('./animation' + str(time()) +'.gif', writer='ffmpeg', fps=30)

    return


Q_x = 5
Q_y = 5
Q_theta = 0.1
R_v = 0.5
R_omega = 0.005

rob_diam = 0.3

dt = 0.1
N = 10
sim_time = 200

x_0 = 7.5
y_0 = 0
theta_0 = casadi.pi / 2
x_ref = 7.5
y_ref = 15
theta_ref = casadi.pi / 2

v_max = 2
v_min = -v_max
omega_max = casadi.pi/4
omega_min = -omega_max


def dm_to_array(dm):
    return np.array(dm.full())


def shift_timestep(h, time, state, control, f):
    delta_state = f(state, control[:, 0])
    next_state = casadi.DM.full(state + h * delta_state)
    next_time = time + h
    next_control = casadi.horzcat(control[:, 1:],
                                  casadi.reshape(control[:, -1], -1, 1))

    return next_time, next_state, next_control


x = casadi.SX.sym('x')
y = casadi.SX.sym('y')
theta = casadi.SX.sym('theta')
states = casadi.vertcat(x, y, theta)
n_states = states.numel()

v = casadi.SX.sym('v')
omega = casadi.SX.sym('omega')
controls = casadi.vertcat(v, omega)
n_controls = controls.numel()

X = casadi.SX.sym('X', n_states, N + 1)
U = casadi.SX.sym('U', n_controls, N)
P = casadi.SX.sym('P', 2 * n_states)
Q = casadi.diagcat(Q_x, Q_y, Q_theta)
R = casadi.diagcat(R_v, R_omega)

rhs = casadi.vertcat(v * casadi.cos(theta), v * casadi.sin(theta), omega)
f = casadi.Function('f', [states, controls], [rhs])

cost = 0
g = X[:, 0] - P[:n_states]

for k in range(N):
    state = X[:, k]
    control = U[:, k]
    cost = cost + (state - P[n_states:]).T @ Q @ (state - P[n_states:]) + \
            control.T @ R @ control
    next_state = X[:, k + 1]
    k_1 = f(state, control)
    k_2 = f(state + dt/2 * k_1, control)
    k_3 = f(state + dt/2 * k_2, control)
    k_4 = f(state + dt * k_3, control)
    predicted_state = state + dt/6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
    g = casadi.vertcat(g, next_state - predicted_state)

for k in range(N + 1):
    g = casadi.vertcat(g, -casadi.sqrt((X[0, k] - obs_x)**2 + (X[1, k] - obs_y)**2) + (rob_diam / 2 + obs_diam / 2))

opt_variables = casadi.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))

nlp_prob = {
    'f': cost,
    'x': opt_variables,
    'g': g,
    'p': P
}

opts = {
    'ipopt': {
        'sb': 'yes',
        'max_iter': 2000,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0,
}
solver = casadi.nlpsol('solver', 'ipopt', nlp_prob, opts)

lbx = casadi.DM.zeros((n_states * (N + 1) + n_controls * N, 1))
ubx = casadi.DM.zeros((n_states * (N + 1) + n_controls * N, 1))

lbx[0:n_states * (N + 1):n_states] = -casadi.inf
lbx[1:n_states * (N + 1):n_states] = -casadi.inf
lbx[2:n_states * (N + 1):n_states] = -casadi.inf

ubx[0:n_states * (N + 1):n_states] = casadi.inf
ubx[1:n_states * (N + 1):n_states] = casadi.inf
ubx[2:n_states * (N + 1):n_states] = casadi.inf

lbx[n_states * (N + 1):n_states * (N + 1) + n_controls * N:n_controls] = v_min
ubx[n_states * (N + 1):n_states * (N + 1) + n_controls * N:n_controls] = v_max
lbx[n_states * (N + 1) + 1:n_states * (N + 1) + n_controls * N:n_controls] = omega_min
ubx[n_states * (N + 1) + 1:n_states * (N + 1) + n_controls * N:n_controls] = omega_max

lbg = casadi.DM.zeros((n_states * (N + 1) + (N + 1), 1))
ubg = casadi.DM.zeros((n_states * (N + 1) + (N + 1), 1))

lbg[n_states * (N + 1):] = -casadi.inf
ubg[n_states * (N + 1):] = 0

args = {
    'lbg': lbg,
    'ubg': ubg,
    'lbx': lbx,
    'ubx': ubx
}

t_0 = 0
state_0 = casadi.DM([x_0, y_0, theta_0])
state_ref = casadi.DM([x_ref, y_ref, theta_ref])

t = casadi.DM(t_0)
u0 = casadi.DM.zeros((n_controls, N))
X0 = casadi.repmat(state_0, 1, N + 1)

mpc_iter = 0

cat_states = dm_to_array(X0)
cat_controls = dm_to_array(u0[:, 0])
times = np.array([[0]])


if __name__ == '__main__':
    while casadi.norm_2(state_0 - state_ref) > 1e-1 and mpc_iter * dt < sim_time:
        t1 = time.time()
        args['p'] = casadi.vertcat(state_0, state_ref)
        args['x0'] = casadi.vertcat(casadi.reshape(X0, n_states * (N + 1), 1),
                                    casadi.reshape(u0, n_controls * N, 1))

        sol = solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'],
                     lbg=args['lbg'], ubg=args['ubg'], p=args['p'])

        u = casadi.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        X0 = casadi.reshape(sol['x'][:n_states * (N + 1)], n_states, N + 1)

        cat_states = np.dstack((cat_states, dm_to_array(X0)))
        cat_controls = np.dstack((cat_controls, dm_to_array(u[:, 0])))

        t_0, state_0, u0 = shift_timestep(dt, t_0, state_0, u, f)
        X0 = casadi.horzcat(X0[:, 1:], casadi.reshape(X0[:, -1], -1, 1))

        t2 = time.time()

        times = np.vstack((times, t2 - t1))

        mpc_iter += 1

print(f'steady state error: {casadi.norm_2(state_0 - state_ref)}')
simulate(cat_states, cat_controls, times, dt, N,
         np.array([x_0, y_0, theta_0, x_ref, y_ref, theta_ref]), save=False)
