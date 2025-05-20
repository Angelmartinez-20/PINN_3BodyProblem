import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from generateGeneral3BodyData import generate_three_positions_3d, make_trajectory_3d

torch.set_default_dtype(torch.float64)
np.random.seed(100)

class PINN(nn.Module):
    def __init__(self, in_features=22, hidden=128, n_layers=10):
        super().__init__()
        layers = [nn.Linear(in_features, hidden), nn.ReLU()]
        for _ in range(n_layers):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, 18))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def compute_lagrangian(Y, masses):
    # Y: (N,18) rows are [pos(9), vel(9)]
    pos = Y[:, :9].reshape(-1,3,3)
    vel = Y[:, 9:].reshape(-1,3,3)
    T = 0.5 * np.sum(masses * np.sum(vel**2, axis=2), axis=1)
    U = np.zeros(len(Y))
    G = 1.0
    for i in range(3):
        for j in range(i+1, 3):
            r = np.linalg.norm(pos[:,i] - pos[:,j], axis=1)
            U -= G * masses[i] * masses[j] / r
    return T + U

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Load the trained model
    model = PINN().to(device)
    model.load_state_dict(torch.load(
        "../Models/finalModelGeneral3Body_1.pt",
        map_location=device))
    model.eval()

    # Generate one long trajectory to T=500 with dt=0.1
    p1,p2,p3 = generate_three_positions_3d()
    v1,v2,v3 = np.random.uniform(-0.5,0.5,(3,3))
    masses   = np.random.uniform(1.0,5.0,3)
    T, dt = 250.0, 0.1
    ts, Y_true = make_trajectory_3d(p1,p2,p3,v1,v2,v3,masses, T=T, dt=dt)
    N = len(ts)

    # Build inputs for the PINN
    init_state = np.hstack((p1,p2,p3,v1,v2,v3))  # (18,)
    init_rep   = np.repeat(init_state[None,:], N, axis=0)
    mass_rep   = np.repeat(masses[None,:],   N, axis=0)
    t_col      = ts[:,None]
    X_test     = np.hstack((mass_rep, init_rep, t_col))  # shape (N,22)

    # Run PINN
    X_tensor = torch.tensor(X_test, device=device)
    with torch.no_grad():
        Y_pred = model(X_tensor).cpu().numpy()  # (N,18)

    # Compute metrics at each time
    mse_per_step = ((Y_true - Y_pred)**2).mean(axis=1)              # (N,)
    E0 = compute_lagrangian(Y_true[[0], :], masses)[0]  # scalar
    E_pred       = compute_lagrangian(Y_pred, masses)             # (N,)
    drift_per_step = np.abs(E_pred - E0)                      # (N,)

    # Plot MSE vs time
    plt.figure(figsize=(8,4))
    plt.plot(ts, mse_per_step)
    plt.axvline(x=50, color='red', linestyle='--', label="Training Stopped") 
    plt.xlabel("Time")
    plt.ylabel("MSE")
    plt.title("MSE over Time")
    plt.legend() 
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("mse_vs_time.png", dpi=200)

    # Plot energy drift vs time
    plt.figure(figsize=(8,4))
    plt.plot(ts, drift_per_step)
    plt.axvline(x=50, color='red', linestyle='--', label="Training Stopped")  
    plt.xlabel("Time")
    plt.ylabel("|$E_{pred}(t)-E(0)$|")
    plt.title("Energy Drift over Time")
    plt.legend() 
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("energy_drift_vs_time.png", dpi=200)

    # 3D Trajectories side‑by‑side
    fig = plt.figure(figsize=(12,7))

    ax_true = fig.add_subplot(1, 2, 1, projection='3d')
    for i,c in enumerate(['red','blue','green']):
        ax_true.plot(
            Y_true[:,3*i+0],
            Y_true[:,3*i+1],
            Y_true[:,3*i+2],
            '-', color=c, label=f'Body {i+1}'
        )
    ax_true.set_title("RK45")
    ax_true.set_xlabel("X"); ax_true.set_ylabel("Y"); ax_true.set_zlabel("Z")
    ax_true.legend()

    ax_pred = fig.add_subplot(1, 2, 2, projection='3d')
    for i,c in enumerate(['red','blue','green']):
        ax_pred.plot(
            Y_pred[:,3*i+0],
            Y_pred[:,3*i+1],
            Y_pred[:,3*i+2],
            '-', color=c, label=f'Body {i+1}'
        )
    ax_pred.set_title("PINN λ=0.1")
    ax_pred.set_xlabel("X"); ax_pred.set_ylabel("Y"); ax_pred.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig("trajectory_comparison_3d.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()
