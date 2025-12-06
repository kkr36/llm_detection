import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ----------------------------
# Top-K Sparse Autoencoder
# ----------------------------
class TopKAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, k):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.k = k  # number of active units per sample

    def forward(self, x):
        z = torch.relu(self.encoder(x))  # pre-sparsity activations

        # --- Apply Top-K sparsity ---
        if self.k < z.shape[1]:
            topk_vals, topk_idx = torch.topk(z, self.k, dim=1)
            mask = torch.zeros_like(z).scatter_(1, topk_idx, 1.0)
            z = z * mask  # zero out non-topk activations

        x_hat = self.decoder(z)
        return x_hat, z

    def loss_fn(self, x, x_hat):
        return nn.functional.mse_loss(x_hat, x)


# ----------------------------
# Training function
# ----------------------------
def train_topk_sae(embeddings, hidden_dim=256, k=10, lr=1e-3, epochs=20, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    dataset = TensorDataset(embeddings)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = embeddings.shape[1]
    model = TopKAutoencoder(input_dim, hidden_dim, k).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(batch)
            loss = model.loss_fn(batch, x_hat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    import numpy as np

    year = 2018
    embeddings = np.load(f"/share/garg/openreview_data/all_embeddings_{year}.npy")

    model = train_topk_sae(
        embeddings,
        hidden_dim=256, k=10, lr=1e-3, epochs=30, batch_size=16
    )

    # Save the model
    torch.save(model.state_dict(), f"/share/garg/openreview_data/sparse_autoencoder_{year}.pt")
    print(f"Model saved to /share/garg/openreview_data/sparse_autoencoder_{year}.pt")

    # ----------------------------
    # Save encoder activations (sparse codes)
    # ----------------------------
    model.eval()
    with torch.no_grad():
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        _, codes = model(embeddings_tensor)
        codes_np = codes.numpy()

    np.save(f"/share/garg/openreview_data/sparse_codes_{year}.npy", codes_np)
    print(f"Sparse codes saved to sparse_codes_{year}.npy with shape {codes_np.shape}")