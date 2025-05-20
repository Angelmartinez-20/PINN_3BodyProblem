import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

torch.set_default_dtype(torch.float64)

# ===================== Helpers =====================

def accel_torch(pos, masses):
    B = pos.size(0)
    a = torch.zeros_like(pos)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            r = pos[:, j, :] - pos[:, i, :]
            d = torch.norm(r, dim=1, keepdim=True)
            a[:, i, :] += masses[:, j:j+1] * r / (d**3 + 1e-10)
    return a

# ===================== Model =====================

class PINN(nn.Module):
    def __init__(self, in_features=22, hidden=128, n_layers=10):
        super().__init__()
        layers = [nn.Linear(in_features, hidden), nn.ReLU()]
        for _ in range(n_layers):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, 18))  # 9 pos + 9 vel outputs
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ===================== Dataset =====================

class TrajectoryDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X  # X: (n_traj, steps, 22)
        self.Y = Y  # Y: (n_traj, steps, 18)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ===================== Training =====================

def train_model(model, optimizer, X, Y, device='cpu', log_file='training_log.log',
                steps_per_traj=6000, chunk_size=1000, num_epochs=200):
    # Prepare log file
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write('Epoch\tAvgLoss\tAvgDataLoss\tAvgPhysLoss\tWeight\n')

    # Reshape into trajectories
    n_traj = X.size(0) // steps_per_traj
    X = X.view(n_traj, steps_per_traj, -1)
    Y = Y.view(n_traj, steps_per_traj, -1)
    dataset = TrajectoryDataset(X, Y)

    loader = DataLoader(dataset, batch_size=chunk_size, shuffle=True,
                        pin_memory=(device != 'cpu'), num_workers=4)

    # Physics weighting schedule
    lambda_f_schedule = torch.linspace(0.001, 0.75, num_epochs-80)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_phys_loss = 0.0

        # Determine current weight
        if epoch < 80:
            current_weight = 0.0
        else:
            current_weight = 10.0 #lambda_f_schedule[epoch-80].item()

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            B, S, _ = xb.shape
            xb_flat = xb.view(B * S, -1)  # (B*S, 22)
            yb_flat = yb.view(B * S, -1)  # (B*S, 18)

            inp = xb_flat.clone().detach().requires_grad_(True)
            t_in = inp[:, -1].unsqueeze(1)  # (B*S, 1)

            # Forward pass
            U = model(inp)  # (B*S, 18)
            loss_data = F.mse_loss(U, yb_flat)

            # Physics loss calculation
            dr_dt = []
            dv_dt = []
            for i in range(U.shape[1]):  # 0 to 17
                # grad w.r.t. the full input leaf `inp`
                grad_inp = torch.autograd.grad(U[:, i].sum(), inp, create_graph=True)[0]   # (B*S, 22)
                g_i = grad_inp[:, -1].contiguous().view(-1)                            # take the time-column
                (dr_dt if i < 9 else dv_dt).append(g_i)

            dr_dt = torch.stack(dr_dt, dim=1)
            dv_dt = torch.stack(dv_dt, dim=1)    

            # Extract positions and masses
            pos_pred = U[:, :9].view(-1, 3, 3)  # (B*S, 3, 3)
            masses = inp[:, 0:3]             # (B*S, 3)
            acc_pred = accel_torch(pos_pred, masses).view(-1, 9)  # (B*S, 9)

            # Physics loss: dr_dt = velocities, dv_dt = accelerations
            loss_phys = F.mse_loss(dr_dt, U[:, 9:18]) + F.mse_loss(dv_dt, acc_pred)

            # Total loss
            loss = loss_data + current_weight * loss_phys

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_data_loss += loss_data.item()
            total_phys_loss += loss_phys.item()

        avg_loss = total_loss / len(loader)
        avg_data_loss = total_data_loss / len(loader)
        avg_phys_loss = total_phys_loss / len(loader)

        print(f"Epoch {epoch:03d}  AvgLoss={avg_loss:.6f}  "
            f"DataLoss={avg_data_loss:.6f}  PhysLoss={avg_phys_loss:.6f}  "
            f"Weight={current_weight:.4f}")

        with open(log_file, 'a') as f_log:
            f_log.write(f"{epoch:03d}\t{avg_loss:.6f}\t"
                        f"{avg_data_loss:.6f}\t{avg_phys_loss:.6f}\t"
                        f"{current_weight:.4f}\n")

# ===================== Main =====================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load('Data/trainingDatasetGeneral3Body.pt')
    X = data['X']       # (n_samples, 22)
    Y = data['Y']       # (n_samples, 18)
    print(X.shape, Y.shape)

    # prepare model and optimizer
    steps_per_traj = 5000
    n_traj = X.shape[0] // steps_per_traj
    chunk_size = max(1, n_traj // 100)
    print(f"Using steps_per_traj={steps_per_traj}, n_traj={n_traj}, chunk_size={chunk_size}")

    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, optimizer, X, Y, device=device, log_file='LogGeneral3Body10warmup_5000.txt',
                steps_per_traj=steps_per_traj, chunk_size=chunk_size, num_epochs=200)

    # save final model
    torch.save(model.state_dict(), '/research2/ammartinez20/finalModelGeneral3Body10warmup_5000.pt')
    print("Saved final model to 'finalModelGeneral3Body10warmup_5000.pt'")

if __name__ == '__main__':
    main()