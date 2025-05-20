import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

from generateSimple3BodyData import generate_three_positions, make_trajectory
torch.set_default_dtype(torch.float64)

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(7, 128), nn.ReLU()]
        for _ in range(10):
            layers += [nn.Linear(128, 128), nn.ReLU()]
        layers.append(nn.Linear(128, 12))  # outputs: [pos(6), vel(6)]
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

    # Load trained model
    model = PINN().to(device)
    model.load_state_dict(torch.load(
        "Models/finalModelSimple3Body_001-75warmup.pt",
        map_location=device))
    model.eval()

    # Load test data
    data = torch.load("Data/testingDatasetSimple3Body.pt")
    X_all = data['X']  
    Y_all = data['Y'] 

    steps_per_traj = 256
    n_traj = X_all.size(0) // steps_per_traj

    # Storage
    mse_time    = np.zeros((n_traj, steps_per_traj))
    mse_final   = np.zeros(n_traj)
    energy_drift = np.zeros(n_traj)

    X_np = X_all.numpy()

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
            # energy drift
            E0 = compute_lagrangian(Yt[[0], :])
            ET = compute_lagrangian(Yp[[-1], :])
            energy_drift[t] = abs(ET[0] - E0[0])

    # Aggregates
    avg_traj_mse   = mse_time.mean(axis=1).mean()
    avg_final_mse  = mse_final.mean()
    avg_energy_drift = energy_drift.mean()

    print(f"Avg MSE per trajectory (all steps): {avg_traj_mse:.3f}")
    print(f"Avg final‐step MSE:                {avg_final_mse:.3f}")
    print(f"Avg energy dift:                 {avg_energy_drift:.3f}")

    # Plot
    # plt.figure(figsize=(8,5))
    # plt.plot(avg_mse_time, label="Avg MSE per step")
    # plt.yscale('log')
    # plt.xlabel("Time step")
    # plt.ylabel("MSE")
    # plt.title("Average MSE vs. Time on Test Set")
    # plt.tight_layout()
    # # plt.savefig("avg_mse_per_step.png", dpi=200)
    # plt.show()

if __name__ == "__main__":
    main()