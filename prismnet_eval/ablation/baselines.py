"""Baseline models for testing data quality hypothesis.

These models are intentionally simple to test whether PrismNet's performance
comes from architectural sophistication or data quality (icSHAPE + sequence).
All models are parameter-matched (~58K params) to the minimal ablation model.
"""

import torch
import torch.nn as nn


class PlainCNN2D1D(nn.Module):
    """Plain CNN with 2D→1D transition, no residual blocks.

    Architecture:
    - 2 Conv2D layers: 1→14→28 channels
    - AvgPool2d to collapse feature dimension
    - 2 Conv1D layers: 28→56→56 channels
    - Global average pooling → FC

    Target: ~57K parameters to match minimal ablation
    """

    def __init__(self):
        super().__init__()

        # 2D convolution stage
        self.conv2d_1 = nn.Conv2d(1, 14, kernel_size=(7, 3), padding=(3, 1))
        self.bn2d_1 = nn.BatchNorm2d(14)

        self.conv2d_2 = nn.Conv2d(14, 28, kernel_size=(7, 3), padding=(3, 1))
        self.bn2d_2 = nn.BatchNorm2d(28)

        # Pool across feature dimension (101×5 → 101×1)
        self.avgpool2d = nn.AvgPool2d(kernel_size=(1, 5))

        # 1D convolution stage
        self.conv1d_1 = nn.Conv1d(28, 56, kernel_size=11, padding=5)
        self.bn1d_1 = nn.BatchNorm1d(56)

        self.conv1d_2 = nn.Conv1d(56, 56, kernel_size=11, padding=5)
        self.bn1d_2 = nn.BatchNorm1d(56)

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(56, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 2D stage
        x = self.relu(self.bn2d_1(self.conv2d_1(x)))
        x = self.relu(self.bn2d_2(self.conv2d_2(x)))

        # Pool feature dimension
        x = self.avgpool2d(x)  # (batch, 28, 101, 1)
        x = x.squeeze(-1)      # (batch, 28, 101)

        # 1D stage
        x = self.relu(self.bn1d_1(self.conv1d_1(x)))
        x = self.relu(self.bn1d_2(self.conv1d_2(x)))

        # Global pooling and classification
        x = self.global_pool(x).squeeze(-1)  # (batch, 56)
        x = self.fc(x)  # (batch, 1)

        return x


class PlainCNN2D(nn.Module):
    """Pure 2D CNN without 1D transition.

    Architecture:
    - 4 Conv2D layers: 1→14→28→42→56 channels
    - Smaller kernels to reduce parameters
    - Global average pooling across both spatial dimensions
    - FC layer

    Target: ~57K parameters
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 14, kernel_size=(7, 3), padding=(3, 1))
        self.bn1 = nn.BatchNorm2d(14)

        self.conv2 = nn.Conv2d(14, 28, kernel_size=(7, 3), padding=(3, 1))
        self.bn2 = nn.BatchNorm2d(28)

        self.conv3 = nn.Conv2d(28, 42, kernel_size=(5, 3), padding=(2, 1))
        self.bn3 = nn.BatchNorm2d(42)

        self.conv4 = nn.Conv2d(42, 56, kernel_size=(5, 3), padding=(2, 1))
        self.bn4 = nn.BatchNorm2d(56)

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(56, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))

        # Global pooling across both dimensions
        x = self.global_pool(x).squeeze(-1).squeeze(-1)  # (batch, 56)
        x = self.fc(x)  # (batch, 1)

        return x


class BiLSTMBaseline(nn.Module):
    """Bidirectional LSTM baseline.

    Tests if CNN spatial processing is necessary or if sequential
    modeling alone is sufficient.

    Architecture:
    - Reshape input to (batch, seq_len, features)
    - 2-layer BiLSTM with hidden_size=41
    - Take final hidden state from both directions
    - FC layer

    Target: ~57K parameters
    """

    def __init__(self, hidden_size=41, num_layers=2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # BiLSTM: input_size=5 (features), hidden_size=41, 2 layers
        self.lstm = nn.LSTM(
            input_size=5,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # FC layer: 2*hidden_size (bidirectional) → 1
        self.fc = nn.Linear(2 * hidden_size, 1)

    def forward(self, x):
        # Input: (batch, 1, 101, 5)
        # Reshape to (batch, 101, 5) for LSTM
        batch_size = x.size(0)
        x = x.squeeze(1)  # (batch, 101, 5)

        # LSTM forward
        # output: (batch, seq_len, 2*hidden_size)
        # h_n: (num_layers*2, batch, hidden_size)
        output, (h_n, c_n) = self.lstm(x)

        # Take final hidden state from both directions
        # h_n[-2]: forward direction last layer
        # h_n[-1]: backward direction last layer
        h_forward = h_n[-2]  # (batch, hidden_size)
        h_backward = h_n[-1]  # (batch, hidden_size)
        h_combined = torch.cat([h_forward, h_backward], dim=1)  # (batch, 2*hidden_size)

        # Classification
        out = self.fc(h_combined)  # (batch, 1)

        return out


class BiGRUBaseline(nn.Module):
    """Bidirectional GRU baseline.

    Similar to BiLSTM but with GRU cells. Adjusted hidden_size=48
    to match parameter count.

    Architecture:
    - Reshape input to (batch, seq_len, features)
    - 2-layer BiGRU with hidden_size=48
    - Take final hidden state from both directions
    - FC layer

    Target: ~57K parameters
    """

    def __init__(self, hidden_size=48, num_layers=2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # BiGRU: input_size=5, hidden_size=48, 2 layers
        self.gru = nn.GRU(
            input_size=5,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # FC layer: 2*hidden_size (bidirectional) → 1
        self.fc = nn.Linear(2 * hidden_size, 1)

    def forward(self, x):
        # Input: (batch, 1, 101, 5)
        # Reshape to (batch, 101, 5) for GRU
        batch_size = x.size(0)
        x = x.squeeze(1)  # (batch, 101, 5)

        # GRU forward
        # output: (batch, seq_len, 2*hidden_size)
        # h_n: (num_layers*2, batch, hidden_size)
        output, h_n = self.gru(x)

        # Take final hidden state from both directions
        h_forward = h_n[-2]  # (batch, hidden_size)
        h_backward = h_n[-1]  # (batch, hidden_size)
        h_combined = torch.cat([h_forward, h_backward], dim=1)  # (batch, 2*hidden_size)

        # Classification
        out = self.fc(h_combined)  # (batch, 1)

        return out


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    """Verification script for baseline models."""
    print("Baseline Model Parameter Counts\n" + "="*50)

    models = {
        "PlainCNN2D1D": PlainCNN2D1D(),
        "PlainCNN2D": PlainCNN2D(),
        "BiLSTM": BiLSTMBaseline(),
        "BiGRU": BiGRUBaseline(),
    }

    # Test input
    x = torch.randn(8, 1, 101, 5)

    for name, model in models.items():
        params = count_parameters(model)
        print(f"\n{name}:")
        print(f"  Parameters: {params:,} ({params/1000:.1f}K)")

        # Test forward pass
        try:
            model.eval()
            with torch.no_grad():
                out = model(x)
            assert out.shape == (8, 1), f"Expected shape (8, 1), got {out.shape}"
            print(f"  Output shape: {out.shape} ✓")
            print(f"  Forward pass: OK ✓")
        except Exception as e:
            print(f"  Forward pass: FAILED - {e}")

    print("\n" + "="*50)
    print("Target: 55K-61K parameters (matching minimal ablation ~58K)")
