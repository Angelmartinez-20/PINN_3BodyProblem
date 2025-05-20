import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

torch.set_default_dtype(torch.float64)

# ===================== Helpers =====================

def accel_torch(pos, masses=None):
    B, N, D = pos.size()
    if masses is None:
        masses = torch.ones(B, N, device=pos.device)
    a = torch.zeros_like(pos)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            r = pos[:, j, :] - pos[:, i, :]
            d = torch.norm(r, dim=1, keepdim=True)
            a[:, i, :] += masses[:, j:j+1] * r / (d**3 + 1e-8)
    return a
# ===================== Model =====================

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

# ===================== Dataset =====================

class TrajectoryDataset(Dataset):
    def __init__(self, X, Y):
        # X: (num_traj, steps, 7), Y: (num_traj, steps, 12)
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ===================== Training =====================

def train_model(model, optimizer, X, Y, device='cpu', log_file='training_log.txt',
                steps_per_traj=256, chunk_size=1000, num_epochs=200):
    # Prepare log file
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write('Epoch\tAvgLoss\tAvgDataLoss\tAvgPhysLoss\tWeight\n')

    # Reshape into trajectories
    n_traj = X.size(0) // steps_per_traj
    X = X.view(n_traj, steps_per_traj, -1)  # (-1 resolves to 7)
    Y = Y.view(n_traj, steps_per_traj, -1)  # (-1 resolves to 12)
    dataset = TrajectoryDataset(X, Y)

    # DataLoader: load `chunk_size` trajectories at a time
    loader = DataLoader(dataset, batch_size=chunk_size, shuffle=True, 
                        pin_memory=(device != 'cpu'), num_workers=4)

    # Physics weighting schedule
    lambda_f_schedule = np.linspace(0.01, 0.75, num_epochs-80)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_phys_loss = 0.0

        for xb, yb in loader:
            # Move a chunk to device
            xb = xb.to(device)
            yb = yb.to(device)

            B, S, _ = xb.shape
            xb_flat = xb.view(B * S, -1)  # (batch_time, 7)
            yb_flat = yb.view(B * S, -1)  # (batch_time, 12)

            # Rebuild time input with gradient tracking
            t_in = xb_flat[:, 6:7].clone().detach().requires_grad_(True)
            inp = torch.cat([xb_flat[:, :6], t_in], dim=1)

            # Forward pass
            U = model(inp)  # (batch_time, 12)
            loss_data = F.l1_loss(U, yb_flat)

            # Physics loss
            dr_dt = []
            dv_dt = []
            for i in range(12):
                g_i = torch.autograd.grad(U[:, i].sum(), t_in, create_graph=True)[0].view(-1)
                (dr_dt if i < 6 else dv_dt).append(g_i)
            dr_dt = torch.stack(dr_dt, dim=1)
            dv_dt = torch.stack(dv_dt, dim=1)

            # Compute predicted acceleration from positions
            pos_pred = U[:, :6].view(-1, 3, 2)
            acc_pred = accel_torch(pos_pred).view(-1, 6)
            loss_phys = F.mse_loss(dr_dt, U[:, 6:12]) + F.mse_loss(dv_dt, acc_pred)

            # Combine losses
            weight = 0.1 if epoch >= 80 else 0.0
            loss = loss_data + weight * loss_phys

            # Backward + optimize
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
        current_weight = 0.1 if epoch >= 80 else 0.0
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
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    data = torch.load('/research2/ammartinez20/trainingDatasetSimple3Body.pt')
    X = data['X']  # shape: (num_steps, 7)
    Y = data['Y']  # shape: (num_steps, 12)
    print(X.shape, Y.shape)

    # Updated simulation settings
    duration = 10.0           # total time in seconds
    dt = 0.0390625            # time step
    steps_per_traj = int(duration / dt)  # 256 steps
    n_traj = X.size(0) // steps_per_traj
    print(Y.shape)
    chunk_size = max(1, n_traj // 10)

    train_model(model, optimizer, X, Y, device=device, log_file='trainingLogSimple3Body_1warmup.txt',
                steps_per_traj=steps_per_traj, chunk_size=chunk_size, num_epochs=200)

    torch.save(model.state_dict(), 'finalModelSimple3Body_1warmup.pt')
    print("\nSaved final model to 'finalModelSimple3Body_1warmup.pt'")

if __name__ == '__main__':
    main()
