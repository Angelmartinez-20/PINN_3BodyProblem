import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
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

    # Load model
    model = PINN().to(device)
    model.load_state_dict(torch.load("../Models/finalModelGeneral3Body_1.pt",
                                     map_location=device))
    model.eval()

    # Generate one random trajectory
    p1,p2,p3 = generate_three_positions_3d()
    v1,v2,v3 = np.random.uniform(-0.5,0.5,(3,3))
    masses   = np.random.uniform(1.0,5.0,3)
    ts, Y_true = make_trajectory_3d(p1,p2,p3,v1,v2,v3,masses,T=50.0,dt=0.01)
    N = len(ts)

    # Build input features
    init_state = np.hstack((p1,p2,p3,v1,v2,v3))  # (18,)
    init_rep   = np.repeat(init_state[None,:], N, axis=0)
    mass_rep   = np.repeat(masses[None,:],   N, axis=0)
    t_col      = ts[:,None]
    X_test     = np.hstack((mass_rep, init_rep, t_col))  # (N,22)

    # Predict
    X_tensor = torch.tensor(X_test, device=device)
    with torch.no_grad():
        Y_pred = model(X_tensor).cpu().numpy()

    # Metrics
    mse_per_step = ((Y_true - Y_pred)**2).mean(axis=1)
    mse_final    = mse_per_step[-1]
    mse_traj     = mse_per_step.mean()
    E0 = compute_lagrangian(Y_true[[0], :], masses)
    ET = compute_lagrangian(Y_pred[[-1], :], masses)
    energy_drift = abs(ET[0] - E0[0])

    print(f"Final-step MSE: {mse_final:.4f}")
    print(f"Mean-trajectory MSE: {mse_traj:.4f}")
    print(f"Mean energy diffrence: {energy_drift:.4f}")

    # 3D trajectory comparison
    fig = plt.figure(figsize=(10,7))
    ax  = fig.add_subplot(111, projection='3d')
    colors = ['red','blue','green']
    for i,c in enumerate(colors):
        ax.plot(Y_true[:,3*i+0], Y_true[:,3*i+1], Y_true[:,3*i+2],
                '--', color=c, label=f'Body {i+1} True')
        ax.plot(Y_pred[:,3*i+0], Y_pred[:,3*i+1], Y_pred[:,3*i+2],
                '-',  color=c, label=f'Body {i+1} Pred', alpha=0.6)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("True vs. Predicted Trajectories")
    ax.legend()
    plt.tight_layout()
    plt.savefig("traj_comparison.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()
