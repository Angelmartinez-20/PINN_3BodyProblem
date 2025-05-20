import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

torch.set_default_dtype(torch.float64)

# PINN definition
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
        for j in range(i+1,3):
            r = np.linalg.norm(pos[:,i] - pos[:,j], axis=1)
            U -= G * masses[i] * masses[j] / r
    return T + U

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load trained model
    model = PINN().to(device)
    model.load_state_dict(torch.load(
        "Models/finalModelGeneral3Body10warmup.pt",
        map_location=device))
    model.eval()

    # Load test data
    data = torch.load("Data/testingDatasetGeneral3Body.pt")
    X_all = data['X']  # (n_samples, 22)
    Y_all = data['Y']  # (n_samples, 18)

    steps_per_traj = 5000
    n_traj = X_all.size(0) // steps_per_traj

    # Storage
    mse_time    = np.zeros((n_traj, steps_per_traj))
    mse_final   = np.zeros(n_traj)
    energy_drift  = np.zeros(n_traj)

    X_np = X_all.numpy()
    masses = X_np[0, 0:3]  # (3,)

    with torch.no_grad():
        for t in range(n_traj):
            s, e = t*steps_per_traj, (t+1)*steps_per_traj
            X = X_all[s:e].to(device)
            Yt = Y_all[s:e].numpy()
            Yp = model(X).cpu().numpy()

            # per‐step MSE
            mse_time[t] = ((Yt - Yp)**2).mean(axis=1)
            # final‐step MSE
            mse_final[t] = mse_time[t, -1]
            # energy error
            E0 = compute_lagrangian(Yt[[0], :], masses)
            ET = compute_lagrangian(Yp[[-1], :], masses)
            energy_drift[t] = abs(ET[0] - E0[0])

    # Aggregates
    avg_final_mse  = mse_final.mean()
    avg_traj_mse   = mse_time.mean(axis=1).mean()
    avg_energy_err = energy_drift.mean()

    print(f"Avg MSE per trajectory (all steps): {avg_traj_mse:.3f}")
    print(f"Avg final‐step MSE: {avg_final_mse:.3f}")
    print(f"Avg energy drift: {avg_energy_err:.3f}")

    # Plot
    # plt.figure(figsize=(8,5))
    # plt.plot(avg_mse_time, label="Avg MSE per step")
    # plt.yscale('log')
    # plt.xlabel("Time step")
    # plt.ylabel("MSE")
    # plt.title("Average MSE vs. Time on Test Set")
    # plt.tight_layout()
    # plt.savefig("avg_mse_per_step.png", dpi=200)
    # plt.show()

if __name__ == "__main__":
    main()
