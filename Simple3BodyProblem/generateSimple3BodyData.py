import numpy as np
import csv
import torch                    
from scipy.integrate import solve_ivp
from multiprocessing import Pool, cpu_count

header = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'vx1', 'vy1', 'vx2', 'vy2', 'vx3', 'vy3']

def generate_three_positions():
    p1 = np.array([1.0, 0.0])
    theta = np.random.uniform(0, np.pi / 2)
    x = -min(0.5, np.cos(theta))
    z = np.sin(theta)
    p = np.array([x, z])
    s = np.random.uniform(0, 1)
    p2 = s * p
    p3 = -(p1 + p2)  # since all masses are 1
    return p1, p2, p3

def three_body_rhs(t, y, G=1):
    pos = y[:6].reshape(3, 2)
    vel = y[6:].reshape(3, 2)
    acc = np.zeros_like(pos)
    for i in range(3):
        for j in range(3):
            if i == j: continue
            r = pos[j] - pos[i]
            d = np.linalg.norm(r)
            acc[i] += G * r / (d**3 + 1e-10)
    return np.hstack((vel.flatten(), acc.flatten()))

def make_trajectory(p1, p2, p3, T=10.0, dt=0.0390625):
    N = int(T / dt)
    ts = np.linspace(0, T, N)
    v1 = v2 = v3 = np.zeros(2)
    y0 = np.hstack((p1, p2, p3, v1, v2, v3))
    sol = solve_ivp(three_body_rhs, (0, T), y0, t_eval=ts,
                    args=(1.0,), rtol=1e-9, atol=1e-9)
    return sol.t, sol.y.T

def generate_one_trajectory(idx, seed, T=10.0, dt=0.0390625):
    np.random.seed(seed)
    print(f"Generating trajectory {idx}")
    # 1) random initial config
    p1, p2, p3 = generate_three_positions()
    v1 = v2 = v3 = np.zeros(2)

    # 2) integrate
    ts, Ytrue = make_trajectory(p1, p2, p3, T, dt)

    # 3) build CSV row for initial conditions
    csv_row = [
        p1[0], p1[1],
        p2[0], p2[1],
        p3[0], p3[1],
        v1[0], v1[1],
        v2[0], v2[1],
        v3[0], v3[1],
    ]

    # 4) build X: [x1,y1, x2,y2, x3,y3, t]
    N = ts.shape[0]
    X = np.zeros((N, 7))
    X[:, :6] = np.hstack((p1, p2, p3))
    X[:, 6]  = ts

    return csv_row, X, Ytrue

def generate_dataset(num_trajectories=2000, output_csv='testingStartingConditionsSimple3Body.csv',
                     output_pt='testingDatasetSimple3Body.pt',
                     T=10.0, dt=0.0390625):
    # 1) launch worker pool
    seeds = np.random.SeedSequence().spawn(num_trajectories)
    seed_integers = [s.generate_state(1)[0] for s in seeds]

    args = [(i, seed_integers[i], T, dt) for i in range(num_trajectories)]
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(generate_one_trajectory, args)

    # 2) unzip results
    csv_rows, X_list, Y_list = zip(*results)

    # 3) write CSV of initial conditions
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_rows)

    # 4) stack full trajectories and save as tensors
    X_full = np.vstack(X_list)    # shape: (num_trajectories*N, 7)
    Y_full = np.vstack(Y_list)    # shape: (num_trajectories*N, 12)
    torch.save({
        'X': torch.from_numpy(X_full),
        'Y': torch.from_numpy(Y_full)
    }, output_pt)
    print(len(X_full))

    print(f"Saved {num_trajectories} trajectories:")
    print(f"  • initial conditions → '{output_csv}'")
    print(f"  • trajectory data    → '{output_pt}'")

if __name__ == "__main__":
    generate_dataset()
