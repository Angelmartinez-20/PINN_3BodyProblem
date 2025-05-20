import numpy as np
import csv
import torch
from scipy.integrate import solve_ivp
import os
from multiprocessing import Pool, cpu_count

# ===================== Header =====================
header = [
    'm1', 'm2', 'm3',
    'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3',
    'vx1', 'vy1', 'vz1', 'vx2', 'vy2', 'vz2', 'vx3', 'vy3', 'vz3'
]

# ===================== Generation Helpers =====================

def generate_three_positions_3d(min_dist=2.0):
    b1 = np.zeros(3)
    while True:
        b2 = np.random.uniform(-10, 10, 3)
        if np.linalg.norm(b2 - b1) >= min_dist:
            break
    while True:
        b3 = np.random.uniform(-10, 10, 3)
        if (np.linalg.norm(b3 - b1) >= min_dist and
            np.linalg.norm(b3 - b2) >= min_dist):
            break
    return b1, b2, b3

def three_body_rhs_3d(t, y, G, m):
    pos = y[:9].reshape(3, 3)
    vel = y[9:].reshape(3, 3)
    acc = np.zeros_like(pos)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            r = pos[j] - pos[i]
            d = np.linalg.norm(r)
            acc[i] += G * m[j] * r / (d**3 + 1e-10)
    return np.hstack((vel.flatten(), acc.flatten()))

def make_trajectory_3d(p1, p2, p3, v1, v2, v3, masses, T=60.0, dt=0.01):
    N = int(T / dt)
    ts = np.linspace(0, T, N)
    y0 = np.hstack((p1, p2, p3, v1, v2, v3))
    sol = solve_ivp(three_body_rhs_3d, (0, T), y0, t_eval=ts,
                    args=(1.0, masses), rtol=1e-9, atol=1e-9)
    return sol.t, sol.y.T

# ===================== Single Trajectory Generator =====================

def generate_one_trajectory(idx, seed, T, dt):
    np.random.seed(seed)
    print(f"Generating trajectory {idx}")

    p1, p2, p3 = generate_three_positions_3d()
    v1, v2, v3 = np.random.uniform(-0.5, 0.5, (3, 3))
    masses = np.random.uniform(1.0, 5.0, 3)

    ts, Ytrue = make_trajectory_3d(p1, p2, p3, v1, v2, v3, masses, T=T, dt=dt)

    # Build X
    N = ts.shape[0]
    X_traj = np.zeros((N, 22))
    for i, t in enumerate(ts):
        X_traj[i, :3]   = masses
        X_traj[i, 3:12] = np.hstack((p1, p2, p3))
        X_traj[i, 12:21]= np.hstack((v1, v2, v3))
        X_traj[i, 21]   = t

    # Row for CSV (only the initial condition)
    csv_row = [
        masses[0], masses[1], masses[2],
        *p1, *p2, *p3,
        *v1, *v2, *v3
    ]

    return csv_row, X_traj, Ytrue

# ===================== Dataset Generation =====================

def generate_dataset_3d(num_trajectories=1500, output_csv='testingStartingConditionsSimple3Body.csv',
                        output_pt='testingDatasetGeneral3Body.pt', T=50.0, dt=0.01):
    seeds = np.random.SeedSequence().spawn(num_trajectories)
    seed_integers = [s.generate_state(1)[0] for s in seeds]

    args = [(i, seed_integers[i], T, dt) for i in range(num_trajectories)]
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(generate_one_trajectory, args)

    csv_rows, X_list, Y_list = zip(*results)

    # Save CSV
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_rows)

    # Save Tensor
    X_full = np.vstack(X_list)
    Y_full = np.vstack(Y_list)
    torch.save({'X': torch.from_numpy(X_full), 'Y': torch.from_numpy(Y_full)}, output_pt)
    print(f"Saved {num_trajectories} trajectories to:")
    print(f"  •  CSV → {output_csv}")
    print(f"  •  PT  → {output_pt}")

if __name__ == '__main__':
    generate_dataset_3d()
