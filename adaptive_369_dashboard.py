Import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from math import pi

# =====================================================================
# ADAPTIVE TERNARY 369 LAYER (Deep Learning Architecture)
# =====================================================================
class AdaptiveTernary369Layer(nn.Module):
    def __init__(self, in_features=8, hidden_nodes=9, coupling_strength=1.3, anchor_strength=2.5):
        super().__init__()
        self.hidden_nodes = hidden_nodes
        self.coupling_strength = coupling_strength
        self.anchor_strength = anchor_strength
        
        self.input_proj = nn.Linear(in_features, hidden_nodes)
        self.coupling = nn.Parameter(torch.randn(hidden_nodes, hidden_nodes) * 0.04)
        
        self.idx3, self.idx6, self.idx9 = 2, 5, 8

    def forward_with_tracking(self, x):
        h = self.input_proj(x)
        coupled = torch.matmul(h, self.coupling)
        
        for b in range(h.shape[0]):
            diff = h[b, self.idx3] - h[b, self.idx6]
            coupled[b, self.idx3] += self.coupling_strength * torch.sin(diff)
            coupled[b, self.idx6] += self.coupling_strength * torch.sin(-diff)
            for i in range(self.hidden_nodes):
                if i not in (self.idx3, self.idx6, self.idx9):
                    coupled[b, i] += 0.35 * self.coupling_strength * (h[b, self.idx3] - h[b, i])
        h = h + 0.65 * coupled
        
        # --- ADVANCED ADAPTIVE SATURATION ENGINE ---
        h = torch.tanh(h * 0.6) * 1.8
        mean = h.mean(dim=1, keepdim=True)
        h = h - self.anchor_strength * mean
        h[:, self.idx9] = 0.0
        
        vector_norm = torch.norm(h, dim=1, keepdim=True)
        adaptive_scale = 0.78 / (1.0 + 0.0008 * vector_norm)
        h = h * adaptive_scale
        
        return h


# =====================================================================
# DUAL-PANEL HARMONIC DASHBOARD VISUALIZER
# =====================================================================
class DualPanel369Dashboard:
    def __init__(self):
        self.layer = AdaptiveTernary369Layer()
        
        self.fig = plt.figure(figsize=(16, 9), facecolor='#0d1117')
        gs = gridspec.GridSpec(2, 2, width_ratios=[1.1, 0.9], height_ratios=[1, 1])
        
        self.ax_left = self.fig.add_subplot(gs[:, 0])
        self.ax_left.set_facecolor('#0d1117')
        
        self.ax_norm = self.fig.add_subplot(gs[0, 1])
        self.ax_coherence = self.fig.add_subplot(gs[1, 1])
        for ax in [self.ax_norm, self.ax_coherence]:
            ax.set_facecolor('#161b22')
            ax.tick_params(colors='#c9d1d9')
            ax.grid(True, color='#21262d', linestyle='--')
            
        # Geometry
        self.angles = np.array([i * (2 * pi / 9) + (pi / 2) for i in range(9)])
        self.base_x = np.cos(self.angles).copy()
        self.base_y = np.sin(self.angles).copy()
        self.x_coords = self.base_x.copy()
        self.y_coords = self.base_y.copy()
        
        self.res = 110
        x = np.linspace(-1.68, 1.68, self.res)
        y = np.linspace(-1.68, 1.68, self.res)
        self.X, self.Y = np.meshgrid(x, y)
        
        self.field_img = self.ax_left.imshow(np.zeros((self.res, self.res)), 
                                        extent=[-1.68, 1.68, -1.68, 1.68],
                                        cmap='plasma', alpha=0.55, zorder=0, 
                                        vmin=-3.0, vmax=3.0, origin='lower')
        
        # === ATTRACTOR BOUNDARY RING (R ≈ 3.5) ===
        self.boundary_circle = Circle((0, 0), 3.5, fill=False, linestyle='--', 
                                    color='#88aaff', alpha=0.28, linewidth=1.8, zorder=2)
        self.ax_left.add_artist(self.boundary_circle)
        self.ax_left.text(0, 3.65, 'Contractive Attractor Boundary (R≈3.5)', 
                         ha='center', va='bottom', color='#88aaff', fontsize=10, alpha=0.7)
        
        # Structural connections
        self.lines = []
        self.line_pairs = []
        for i in range(9):
            for j in range(i + 1, 9):
                color = '#ffd700' if {i, j} == {2, 5} else '#1f6feb'
                alpha = 0.8 if {i, j} == {2, 5} else 0.10
                lw = 4.5 if {i, j} == {2, 5} else 1
                line, = self.ax_left.plot([self.x_coords[i], self.x_coords[j]],
                                   [self.y_coords[i], self.y_coords[j]],
                                   color=color, alpha=alpha, linewidth=lw, zorder=1)
                self.lines.append(line)
                self.line_pairs.append((i, j))
        
        self.scatter = self.ax_left.scatter(self.x_coords, self.y_coords, c=np.zeros(9),
                                       cmap='plasma', s=500, edgecolors='white',
                                       linewidths=2.5, zorder=5, vmin=-3.5, vmax=3.5)
        
        # Labels
        self.labels = ['1', '2', '3\n(Kinetic)', '4', '5', '6\n(Kinetic)', '7', '8', '9\n(Stator)']
        self.text_objects = []
        for i, txt in enumerate(self.labels):
            col = '#ffd700' if i in [2, 5] else ('#ff4500' if i == 8 else '#c9d1d9')
            t_obj = self.ax_left.text(self.x_coords[i]*1.28, self.y_coords[i]*1.28, txt,
                        ha='center', va='center', color=col, fontsize=10, fontweight='bold')
            self.text_objects.append(t_obj)
            
        self.ax_left.set_xlim(-4.2, 4.2)
        self.ax_left.set_ylim(-4.2, 4.2)
        self.ax_left.axis('off')
        
        self.title = self.ax_left.text(0, 3.95, "Adaptive Spantelergia Phase Field Dashboard",
                                  ha='center', va='center', color='#c9d1d9', fontsize=12, fontweight='bold')
        
        # Diagnostics
        self.norm_data, self.coherence_data, self.time_steps = [], [], []
        self.line_norm, = self.ax_norm.plot([], [], color='#58a6ff', linewidth=2, label="Total State Norm")
        self.line_coherence, = self.ax_coherence.plot([], [], color='#ff7b72', linewidth=2, label="3-6 Coherence Loss")
        
        self.ax_norm.set_title("System Energy Density Profile", color='#c9d1d9', fontsize=11, fontweight='bold')
        self.ax_coherence.set_title("Kinetic Phase Friction Metric", color='#c9d1d9', fontsize=11, fontweight='bold')
        
        self.time_step = 0
        self.phase3 = 0.0
        self.phase6 = 0.0

    def generate_input(self):
        t = self.time_step * 0.045
        base = np.sin(t * 1.2) + 0.8 * np.sin(t * 3.1)
        drift = np.exp(self.time_step * 0.0045) - 1.0
        noise = np.random.randn(8) * 0.06
        x = np.full(8, base + drift) + np.arange(8) * 0.19 + noise
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    def compute_standing_resonance(self, states):
        pos3 = np.array([self.base_x[2], self.base_y[2]])
        pos6 = np.array([self.base_x[5], self.base_y[5]])
        
        self.phase3 += 0.15 + states[2] * 0.09
        self.phase6 += 0.15 + states[5] * 0.09
        delta_phase = self.phase3 - self.phase6
        
        r3 = np.sqrt((self.X - pos3[0])**2 + (self.Y - pos3[1])**2) + 1e-6
        r6 = np.sqrt((self.X - pos6[0])**2 + (self.Y - pos6[1])**2) + 1e-6
        
        harmonic_condition = np.abs(np.sin(delta_phase / 2.0))
        is_resonant = np.exp(-((harmonic_condition - 1.0) ** 2) / 0.08)
        
        wave3 = np.sin(self.phase3 - r3 * 10.0) / np.sqrt(r3)
        wave6 = np.sin(self.phase6 - r6 * 10.0) / np.sqrt(r6)
        
        standing_wave = np.cos(r3 * 6.0) * np.cos(r6 * 6.0) * np.sin(self.phase3 * 1.5)
        field = (wave3 + wave6) + 2.5 * is_resonant * standing_wave
        
        return field, is_resonant, delta_phase

    def update(self, frame):
        self.time_step = frame
        x_in = self.generate_input()
        
        with torch.no_grad():
            states_tensor = self.layer.forward_with_tracking(x_in)
            states = states_tensor.squeeze(0).numpy()
            total_norm = torch.norm(states_tensor).item()
            
        field, resonance_factor, delta_phase = self.compute_standing_resonance(states)
        self.field_img.set_array(field)
        
        # Metrics
        self.time_steps.append(frame)
        self.norm_data.append(total_norm)
        coherence_loss = np.abs(np.sin(delta_phase)) 
        self.coherence_data.append(coherence_loss)
        
        if len(self.time_steps) > 120:
            self.time_steps.pop(0)
            self.norm_data.pop(0)
            self.coherence_data.pop(0)
            
        self.line_norm.set_data(self.time_steps, self.norm_data)
        self.ax_norm.set_xlim(min(self.time_steps), max(self.time_steps) + 1)
        self.ax_norm.set_ylim(0, max(self.norm_data) * 1.25 + 0.1)
        
        self.line_coherence.set_data(self.time_steps, self.coherence_data)
        self.ax_coherence.set_xlim(min(self.time_steps), max(self.time_steps) + 1)
        self.ax_coherence.set_ylim(-0.1, 1.1)
        
        # Dynamic node positions with anchor lock
        shake_amplitude = resonance_factor * 0.11 * (np.abs(states[:, None]))
        noise_x = np.random.randn(9) * shake_amplitude.flatten()
        noise_y = np.random.randn(9) * shake_amplitude.flatten()
        noise_x[8], noise_y[8] = 0.0, 0.0
        
        self.x_coords = self.base_x + noise_x
        self.y_coords = self.base_y + noise_y
        
        # === DYNAMIC THERMODYNAMIC COLORING + GLOW ===
        node_amplitudes = np.abs(states)
        colors = plt.cm.plasma(np.clip(node_amplitudes / 3.8, 0, 1))  # base plasma
        
        # Red glow near boundary
        edge_colors = ['#ff2222' if amp > 3.0 else '#ffffff' for amp in node_amplitudes]
        edge_widths = [3.5 + 4.0 * (amp - 3.0) if amp > 3.0 else 2.2 for amp in node_amplitudes]
        
        self.scatter.set_offsets(np.stack((self.x_coords, self.y_coords), axis=1))
        self.scatter.set_facecolors(colors)
        self.scatter.set_edgecolors(edge_colors)
        self.scatter.set_linewidths(edge_widths)
        
        # Pulse sizing
        pulse = 1.0 + 0.7 * node_amplitudes * (1.0 + np.sin(self.time_step * 0.4))
        self.scatter.set_sizes(500 + node_amplitudes * 480 * pulse)
        
        # Connections
        for idx, (i, j) in enumerate(self.line_pairs):
            self.lines[idx].set_data([self.x_coords[i], self.x_coords[j]], 
                                   [self.y_coords[i], self.y_coords[j]])
            if {i, j} == {2, 5}:
                kinetic = abs(states[2] - states[5])
                self.lines[idx].set_alpha(min(1.0, 0.3 + kinetic * 1.2))
                self.lines[idx].set_linewidth(4.5 + kinetic * 6.0 + resonance_factor * 5.0)

        for i, t_obj in enumerate(self.text_objects):
            t_obj.set_position((self.x_coords[i] * 1.32, self.y_coords[i] * 1.32))
            
        self.title.set_text(f"Spantelergia Dashboard • Step {self.time_step} | "
                           f"Norm: {total_norm:.3f} | Attractor Pull Active")
        
        return [self.field_img, self.scatter, self.title, self.line_norm, self.line_coherence] + self.lines + self.text_objects

    def animate(self):
        ani = animation.FuncAnimation(self.fig, self.update, frames=None,
                                      interval=42, blit=False, cache_frame_data=False)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    dashboard = DualPanel369Dashboard()
    dashboard.animate()
