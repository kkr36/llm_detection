import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class PlattScaler(nn.Module):
    """
    Learns P(y=1 | x) = sigmoid(A * logit + B)
    """
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(1))
        self.B = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        return torch.sigmoid(self.A * logits + self.B)

class PlattCalibratedClassifier(nn.Module):
    """
    Wraps a pretrained binary classifier with Platt scaling.
    """
    def __init__(self, base_model, platt_scaler):
        super().__init__()
        self.base_model = base_model
        self.platt = platt_scaler

    def forward(self, x):
        logits_2d = self.base_model(x)  # shape [B, 2]

        # binary logit
        logits = logits_2d[:, 1] - logits_2d[:, 0]

        # calibrated P(y=1)
        p1 = self.platt(logits)

        # return 2-class probabilities (drop-in replacement)
        return torch.stack([1.0 - p1, p1], dim=1)

def fit_platt_scaler(
    model,
    calib_loader,
    device="cuda"
):
    """
    Fits Platt scaling on a frozen binary classifier.

    Args:
        model: pretrained classifier returning logits [B, 2]
        calib_loader: DataLoader yielding (inputs, labels)
        device: torch device

    Returns:
        trained PlattScaler
    """
    model.eval()
    model.to(device)

    # freeze model
    for p in model.parameters():
        p.requires_grad = False

    logits_list = []
    labels_list = []

    with torch.no_grad():
        print("platt scaling")
        for _, inputs, _, labels in tqdm(calib_loader):
        # for inputs, labels in calib_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            logits = outputs[:, 1] - outputs[:, 0]
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list).float()

    platt = PlattScaler().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.LBFGS(platt.parameters(), max_iter=100)

    def closure():
        optimizer.zero_grad()
        probs = platt(logits.to(device))
        loss = criterion(probs, labels.to(device))
        loss.backward()
        return loss

    optimizer.step(closure)

    return platt
