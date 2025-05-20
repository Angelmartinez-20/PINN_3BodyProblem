import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from generateSimple3BodyData import generate_three_positions, make_trajectory

torch.set_default_dtype(torch.float64)

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

def compute_lagrangian(Y, masses=[1.0,1.0,1.0]):
    # Y: (N,12) array, masses: length-3
    pos = Y[:,:6].reshape(-1,3,2)
    vel = Y[:,6:].reshape(-1,3,2)
    T = 0.5 * np.sum(masses * np.sum(vel**2,axis=2), axis=1)
    U = np.zeros(len(Y))
    G = 1.0
    for i in range(3):
        for j in range(i+1,3):
            r = np.linalg.norm(pos[:,i]-pos[:,j],axis=1)
            U -= G*masses[i]*masses[j]/r
    return T+U

def eval_model(model_path, ts, init_state, masses, device):
    # load model
    model = PINN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    N = len(ts)
    # build inputs
    rep = np.repeat(init_state[None,:], N, axis=0)  # (N,6)
    tcol= ts[:,None]
    X = np.hstack((rep,tcol))                       # (N,7)
    Xt= torch.tensor(X, device=device)
    with torch.no_grad():
        Yp = model(Xt).cpu().numpy()                # (N,12)
    return Yp

def metrics(Y_true, Y_pred, masses):
    mse_ts = ((Y_true - Y_pred)**2).mean(axis=1)
    mse_traj = mse_ts.mean()
    E0 = compute_lagrangian(Y_true[[0],:], masses)[0]
    ET = compute_lagrangian(Y_pred[[-1],:], masses)[0]
    energy_drift = abs(ET - E0)
    return mse_traj, energy_drift

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate one random trajectory
    p1,p2,p3 = generate_three_positions()
    ts, Y_true = make_trajectory(p1,p2,p3, T=10.0, dt=0.0390625)
    masses = [1.0,1.0,1.0]  # as per your simple data
    init_state = np.hstack((p1,p2,p3))  # (6,)

    # Paths & panel definitions
    panels = [
        ("RK45 Integrator", None),
        ("DNN", "../Models/finalModelSimple3Body_noPhysics.pt"),
        ("PINN λ=0.001 Warmup", "../Models/finalModelSimple3Body_001warmup.pt"),
        ("PINN λ=0.01", "../Models/finalModelSimple3Body_01.pt"),
    ]

    # Evaluate PINN models
    Y_preds = [None] * 4
    metrics_vals = [None] * 4
    Y_preds[0] = Y_true.copy()
    metrics_vals[0] = metrics(Y_true, Y_true, masses)
    for i in range(1,4):
        path = panels[i][1]
        Yp = eval_model(path, ts, init_state, masses, device)
        Y_preds[i] = Yp
        metrics_vals[i] = metrics(Y_true, Yp, masses)

    # Plot 2×2
    fig, axes = plt.subplots(2,2, figsize=(12,10))
    colors = ['red','blue','green']
    labels = ['Body 1','Body 2','Body 3']

    for idx, ax in enumerate(axes.flatten()):
        title, _ = panels[idx]
        Yp = Y_preds[idx]
        mse_traj, e_drift = metrics_vals[idx]

        for b,c,l in zip(range(3), colors, labels):
            x = Yp[:,2*b]
            y = Yp[:,2*b+1]
            style = '-' if idx>0 else '--'
            ax.plot(x,y, style, color=c, label=l)
        ax.set_title(f"{title}\nMSE traj={mse_traj:.3f}, Drift={e_drift:.3f}")
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.axis('equal')
        if idx==0:
            ax.legend(loc='upper right')

    plt.tight_layout()
    plt.suptitle("3‑Body True vs PINN Predictions", y=1.02, fontsize=16)
    plt.savefig("3body_comparison_2x2.png", dpi=200, bbox_inches='tight')
    plt.show()

if __name__=="__main__":
    main()
