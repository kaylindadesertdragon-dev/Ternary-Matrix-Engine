```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# ==================== Ternary369Layer ====================
class Ternary369Layer(nn.Module):
    def __init__(self, in_features: int, hidden_nodes: int = 9, 
                 coupling_strength: float = 1.0, anchor_strength: float = 1.0):
        super().__init__()
        self.hidden_nodes = hidden_nodes
        self.coupling_strength = coupling_strength
        self.anchor_strength = anchor_strength
        
        self.input_proj = nn.Linear(in_features, hidden_nodes)
        self.output_proj = nn.Linear(hidden_nodes, in_features)  # reconstruct target dim
        
        self.coupling = nn.Parameter(torch.randn(hidden_nodes, hidden_nodes) * 0.05)
        
        # Fixed roles
        self.idx3 = 2
        self.idx6 = 5
        self.idx9 = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if len(x.shape) == 3:
            b, s, f = x.shape
            x = x.reshape(b * s, f)
        
        h = self.input_proj(x)
        h = self._apply_ternary_coupling(h)
        h = self._apply_stator_anchor(h)
        out = self.output_proj(h)
        
        if len(orig_shape) == 3:
            out = out.reshape(b, s, -1)
        return out

    def _apply_ternary_coupling(self, h: torch.Tensor) -> torch.Tensor:
        coupled = torch.matmul(h, self.coupling)
        # Per-sample 3-6 kinetic bridge
        for b in range(h.shape[0]):
            diff = h[b, self.idx3] - h[b, self.idx6]
            coupled[b, self.idx3] += self.coupling_strength * torch.sin(diff)
            coupled[b, self.idx6] += self.coupling_strength * torch.sin(-diff)
            # Weak influence on others
            for i in range(self.hidden_nodes):
                if i not in [self.idx3, self.idx6, self.idx9]:
                    coupled[b, i] += 0.3 * self.coupling_strength * (h[b, self.idx3] - h[b, i])
        return h + coupled

    def _apply_stator_anchor(self, h: torch.Tensor) -> torch.Tensor:
        # Global mean centering (anchor)
        mean = h.mean(dim=1, keepdim=True)
        h = h - self.anchor_strength * mean
        # Hard stator on node 9
        h[:, self.idx9] = 0.0
        return h


# ==================== Synthetic Drifting Dataset ====================
def generate_drifting_sequence(seq_len=500, features=8, drift_rate=0.015, noise=0.08, seed=42):
    np.random.seed(seed)
    t = np.arange(seq_len)
    # Base chaotic signal + increasing drift + noise
    base = np.sin(t * 0.1) + 0.5 * np.sin(t * 0.05) + np.cumsum(np.random.randn(seq_len) * drift_rate)
    data = np.zeros((seq_len, features))
    for i in range(features):
        data[:, i] = base + i * 0.2 + np.random.randn(seq_len) * noise
    # Add accelerating concept drift after midpoint
    data[seq_len//2:] += np.linspace(0, 3.0, seq_len//2)[:, None] * np.random.randn(features)
    return torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, features)


# ==================== Models ====================
class TrackingModel(nn.Module):
    def __init__(self, hidden_nodes=9, layer_type='ternary'):
        super().__init__()
        self.hidden_nodes = hidden_nodes
        if layer_type == 'ternary':
            self.layer = Ternary369Layer(in_features=8, hidden_nodes=hidden_nodes)
        else:
            self.layer = nn.Linear(8, hidden_nodes)
            self.out_proj = nn.Linear(hidden_nodes, 8)
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'ternary':
            return self.layer(x)
        else:
            h = self.layer(x)
            return self.out_proj(h)


# ==================== Training Function ====================
def train_model(model, data, epochs=80, lr=0.008):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []
    hidden_states = []  # Track hidden activations
    
    model.train()
    for epoch in tqdm(range(epochs), desc=f"Training {model.layer_type}"):
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, data)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        # Record hidden state (for ternary or linear)
        with torch.no_grad():
            if model.layer_type == 'ternary':
                h = model.layer.input_proj(data.reshape(-1, 8))
                h = model.layer._apply_ternary_coupling(h)
                h = model.layer._apply_stator_anchor(h)
            else:
                h = model.layer(data.reshape(-1, 8))
            hidden_states.append(h.mean(dim=0).numpy())
    
    return losses, np.array(hidden_states)


# ==================== Main Experiment ====================
if __name__ == "__main__":
    data = generate_drifting_sequence(seq_len=400)
    print("Data shape:", data.shape)
    
    # Train both models
    torch.manual_seed(42)
    model_ternary = TrackingModel(layer_type='ternary')
    model_linear = TrackingModel(layer_type='linear')
    
    losses_ternary, hidden_ternary = train_model(model_ternary, data)
    losses_linear, hidden_linear = train_model(model_linear, data)
    
    # ==================== Visualization ====================
    plt.figure(figsize=(15, 10))
    
    # 1. Loss curves
    plt.subplot(2, 2, 1)
    plt.plot(losses_ternary, label='Ternary369', linewidth=2)
    plt.plot(losses_linear, label='Standard Linear', linewidth=2)
    plt.title('Training MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 2. Final Predictions vs Target (last 100 steps)
    with torch.no_grad():
        pred_t = model_ternary(data).squeeze(0)[-100:, 0].numpy()
        pred_l = model_linear(data).squeeze(0)[-100:, 0].numpy()
        target = data.squeeze(0)[-100:, 0].numpy()
    
    plt.subplot(2, 2, 2)
    plt.plot(target, label='Target (drifting)', color='black', linewidth=2)
    plt.plot(pred_t, label='Ternary369', alpha=0.85)
    plt.plot(pred_l, label='Linear', alpha=0.85)
    plt.title('Final 100 Steps Tracking')
    plt.legend()
    plt.grid(True)
    
    # 3. Hidden State Variance Over Training
    plt.subplot(2, 2, 3)
    var_t = np.var(hidden_ternary, axis=1)
    var_l = np.var(hidden_linear, axis=1)
    plt.plot(var_t, label='Ternary369 Hidden Var', linewidth=2)
    plt.plot(var_l, label='Linear Hidden Var', linewidth=2)
    plt.title('Hidden State Variance Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Variance')
    plt.legend()
    plt.grid(True)
    
    # 4. Final Hidden State Heatmap (Ternary)
    plt.subplot(2, 2, 4)
    sns.heatmap(hidden_ternary[-1].reshape(1, -1), annot=True, cmap='viridis', cbar=True)
    plt.title('Final Ternary Hidden State (mean across time)')
    plt.xlabel('Hidden Node Index')
    
    plt.tight_layout()
    plt.show()
    
    # Summary stats
    print("\n=== RESULTS ===")
    print(f"Final Loss - Ternary369 : {losses_ternary[-1]:.6f}")
    print(f"Final Loss - Linear     : {losses_linear[-1]:.6f}")
    print(f"Hidden Variance (last)  - Ternary: {var_t[-1]:.6f}")
    print(f"Hidden Variance (last)  - Linear : {var_l[-1]:.6f}")
```

### How to Run
1. Copy-paste into a `.py` file or Colab/Jupyter notebook.
2. Requires: `torch`, `numpy`, `matplotlib`, `seaborn`, `tqdm`.
3. Run it — it will train both models and display comparison plots.

**What to look for**:
- Lower final loss and more stable tracking = better handling of drift.
- Lower hidden state variance in the Ternary model = successful anchoring by the 9-stator.
- The heatmap shows the strong constraint on the 9-node (index 8).
