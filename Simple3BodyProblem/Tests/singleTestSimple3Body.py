import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from generateSimple3BodyData import generate_three_positions, make_trajectory

torch.set_default_dtype(torch.float64)
np.random.seed(39)

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(7, 128), nn.ReLU()]
        for _ in range(10):
            layers += [nn.Linear(128, 128), nn.ReLU()]
        layers.append(nn.Linear(128, 12))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def compute_lagrangian(Y, masses=[1.0, 1.0, 1.0]):
    pos = Y[:, :6].reshape(-1, 3, 2)
    vel = Y[:, 6:].reshape(-1, 3, 2)
        
    # Kinetic energy: T = 0.5 * m * v^2
    T = 0.5 * np.sum(masses * np.sum(vel**2, axis=2), axis=1)

    # Potential energy: U = -G * sum_i<j(m_i * m_j / r_ij)
    G = 1.0
    U = np.zeros(len(Y))
    for i in range(3):
        for j in range(i + 1, 3):
            r = np.linalg.norm(pos[:, i] - pos[:, j], axis=1)
            U -= G * masses[i] * masses[j] / r

    return T + U

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Load your trained PINN
    model = PINN().to(device)
    model.load_state_dict(torch.load(
        "../Models/finalModelSimple3Body_001warmup.pt",
        map_location=device))
    model.eval()

    # 2) Generate one random 3-body trajectory
    p1, p2, p3 = generate_three_positions()
    ts, Y_true = make_trajectory(p1, p2, p3, T=10.0, dt=0.0390625)
    N = len(ts)

    # 3) Build input features: [x1,y1,x2,y2,x3,y3, t]
    init_state = np.hstack((p1, p2, p3))       # (6,)
    init_rep   = np.repeat(init_state[None,:], N, axis=0)  # (N,6)
    t_col      = ts[:,None]                    # (N,1)
    X_test     = np.hstack((init_rep, t_col))  # (N,7)

    # 4) Model inference
    X_tensor = torch.tensor(X_test, device=device)
    with torch.no_grad():
        Y_pred = model(X_tensor).cpu().numpy()  # (N,12)

    # Metrics
    mse_per_step = ((Y_true - Y_pred)**2).mean(axis=1)
    mse_final    = mse_per_step[-1]
    mse_traj     = mse_per_step.mean()
    E0 = compute_lagrangian(Y_true[[0], :])
    ET = compute_lagrangian(Y_pred[[-1], :])
    energy_drift = abs(ET[0] - E0[0])

    print(f"Final-step MSE: {mse_final:.4f}")
    print(f"Mean-trajectory MSE: {mse_traj:.4f}")
    print(f"Mean energy diffrence: {energy_drift:.4f}")

    # Plot true vs. pred trajectories side by side
    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    colors = ['red','blue','green']
    labels = ['Body 1','Body 2','Body 3']

    # Left: numerical integrator
    ax = axes[0]
    for i,c,l in zip(range(3), colors, labels):
        ax.plot(Y_true[:,2*i], Y_true[:,2*i+1], '-', color=c, label=f'{l} True')
    ax.set_title("RK45 Integrator")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend()
    ax.axis('equal')

    # Right: PINN prediction
    ax = axes[1]
    for i,c,l in zip(range(3), colors, labels):
        ax.plot(Y_pred[:,2*i], Y_pred[:,2*i+1], '-', color=c, label=f'{l} Pred')
    ax.set_title("PINN Prediction")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend()
    ax.axis('equal')

    plt.suptitle("True vs. Predicted 2D Trajectories (3-Body)")
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig("trajectory_comparison_2d.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()
