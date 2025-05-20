import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from ..generateGeneral3BodyData import generate_three_positions_3d, make_trajectory_3d

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

def evaluate_model(model, X_test, Y_true, masses, device):
    with torch.no_grad():
        Y_pred = model(X_test.to(device)).cpu().numpy()
    mse_per_step = ((Y_true - Y_pred)**2).mean(axis=1)
    traj_mse     = mse_per_step.mean()
    E0 = compute_lagrangian(Y_true[[0],:], masses)[0]
    ET = compute_lagrangian(Y_pred[[-1],:], masses)[0]
    drift = abs(ET - E0)
    return Y_pred, traj_mse, drift

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate one test trajectory
    p1,p2,p3 = generate_three_positions_3d()
    v1,v2,v3 = np.random.uniform(-0.5,0.5,(3,3))
    masses   = np.random.uniform(1.0,5.0,3)
    ts, Y_true = make_trajectory_3d(p1,p2,p3,v1,v2,v3,masses, T=50.0, dt=0.01)
    N = len(ts)

    # Build input X_test
    init_state = np.hstack((p1,p2,p3,v1,v2,v3))    # (18,)
    init_rep   = np.repeat(init_state[None,:], N, axis=0)
    mass_rep   = np.repeat(masses[None,:],   N, axis=0)
    t_col      = ts[:,None]
    X_np       = np.hstack((mass_rep, init_rep, t_col))  # (N,22)
    X_test     = torch.tensor(X_np, device=device)
    
    # Load and evaluate all three models
    models = {
        "NoPhysics":  "../Models/finalModelGeneral3Body_noPhysics.pt",
        "Warmup1":    "../Models/finalModelGeneral3Body_1.pt",
        "Warmup10":   "../Models/finalModelGeneral3Body10.pt",
    }
    results = {}
    for name, path in models.items():
        m = PINN().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        Yp, traj_mse, drift = evaluate_model(m, X_test, Y_true, masses, device)
        results[name] = (Yp, traj_mse, drift)

    # Plot 2×2 grid
    fig = plt.figure(figsize=(12,10))
    titles = [
        ("Runge Kutta 45",  Y_true,    None,    None),
        ("DNN",      *results["NoPhysics"]),
        ("PINN λ=0.1",      *results["Warmup1"]),
        ("PINN λ=10",     *results["Warmup10"]),
    ]
    for idx, (title, Ydata, traj_mse, drift) in enumerate(titles, start=1):
        ax = fig.add_subplot(2,2,idx, projection='3d')
        colors = ['red','blue','green']
        for i,c in enumerate(colors):
            linestyle = '-'
            ax.plot(
                Ydata[:,3*i+0], Ydata[:,3*i+1], Ydata[:,3*i+2],
                linestyle=linestyle, color=c,
                label=f'Body {i+1}', alpha=0.8
            )
        ax.set_title(
            f"{title}\n"
            + ("" if idx==1 else f"MSE={traj_mse:.3f}, drift={drift:.3f}")
        )
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        if idx==1:
            ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("bestModelsGeneralComparison.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()
