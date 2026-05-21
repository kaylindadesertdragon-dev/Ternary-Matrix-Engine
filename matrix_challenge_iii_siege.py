Final Binary Siege Control Engine — Challenge III Complete

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from math import pi

# =====================================================================
# ADAPTIVE TERNARY 369 LAYER
# =====================================================================
class AdaptiveTernary369Layer(nn.Module):
    def __init__(self, in_features=8, hidden_nodes=9, coupling_strength=1.3, anchor_strength=2.5):
        super().__init__()
        self.hidden_nodes = hidden_nodes
        self.coupling_strength = coupling_strength
        self.anchor_strength = anchor_strength
        self.base_coupling = coupling_strength
        
        self.input_proj = nn.Linear(in_features, hidden_nodes)
        self.coupling = nn.Parameter(torch.randn(hidden_nodes, hidden_nodes) * 0.04)
        
        self.idx3, self.idx6, self.idx9 = 2, 5, 8
        self.remembered_phase_diff = None

    def forward_with_tracking(self, x, binary_siege=False):
        if binary_siege:
            self.coupling_strength = max(0.25, self.base_coupling * 0.45)
        else:
            self.coupling_strength = self.base_coupling

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

        if binary_siege and self.remembered_phase_diff is not None:
            current_diff = h[:, self.idx3] - h[:, self.idx6]
            echo_strength = 0.55
            phase_error = current_diff - self.remembered_phase_diff
            echo_correction = echo_strength * torch.sin(phase_error) * 2.0
            h[:, self.idx3] += echo_correction
            h[:, self.idx6] -= echo_correction

        h = torch.tanh(h * 0.6) * 1.8
        mean = h.mean(dim=1, keepdim=True)
        h = h - self.anchor_strength * mean
        h[:, self.idx9] = 0.0
        
        vector_norm = torch.norm(h, dim=1, keepdim=True)
        adaptive_scale = 0.78 / (1.0 + 0.0008 * vector_norm)
        h = h * adaptive_scale
        
        return h


# =====================================================================
# BINARY SIEGE DASHBOARD — FINAL MASTER ENGINE
# =====================================================================
class BinarySiege369Dashboard:
    def __init__(self):
        self.layer = AdaptiveTernary369Layer()
        self.binary_siege_active = False
        self.siege_start = 100
        
        self.fig = plt.figure(figsize=(17, 9.5), facecolor='#0a0a0f')
        gs = gridspec.GridSpec(3, 2, width_ratios=[1.15, 0.85], height_ratios=[1, 1, 1])
        
        self.ax_left = self.fig.add_subplot(gs[:, 0])
        self.ax_left.set_facecolor('#0a0a0f')
        
        self.ax_pol = self.fig.add_subplot(gs[0, 1])
        self.ax_def = self.fig.add_subplot(gs[1, 1])
        self.ax_energy = self.fig.add_subplot(gs[2, 1])
        
        for ax in [self.ax_pol, self.ax_def, self.ax_energy]:
            ax.set_facecolor('#11151c')
            ax.tick_params(colors='#c9d1d9')
            ax.grid(True, color='#21262d', linestyle='--')
            
        # Geometry
        self.angles = np.array([i * (2 * pi / 9) + (pi / 2) for i in range(9)])
        self.base_x = np.cos(self.angles).copy()
        self.base_y = np.sin(self.angles).copy()
        self.x_coords = self.base_x.copy()
        self.y_coords = self.base_y.copy()
        
        self.res = 120
        x = np.linspace(-1.8, 1.8, self.res)
        y = np.linspace(-1.8, 1.8, self.res)
        self.X, self.Y = np.meshgrid(x, y)
        
        self.field_img = self.ax_left.imshow(np.zeros((self.res, self.res)), 
                                        extent=[-1.8, 1.8, -1.8, 1.8],
                                        cmap='plasma', alpha=0.6, zorder=0, 
                                        vmin=-3.0, vmax=3.0, origin='lower')
        
        self.boundary_circle = Circle((0, 0), 2.4, fill=False, linestyle='--', 
                                    color='#556688', alpha=0.3, linewidth=1.8, zorder=2)
        self.ax_left.add_artist(self.boundary_circle)
        
        self.lines = []
        self.line_pairs = []
        for i in range(9):
            for j in range(i + 1, 9):
                color = '#ffd700' if {i, j} == {2, 5} else '#1f6feb'
                alpha = 0.7 if {i, j} == {2, 5} else 0.08
                lw = 3.5 if {i, j} == {2, 5} else 1.0
                line, = self.ax_left.plot([self.x_coords[i], self.x_coords[j]],
                                   [self.y_coords[i], self.y_coords[j]],
                                   color=color, alpha=alpha, linewidth=lw, zorder=1)
                self.lines.append(line)
                self.line_pairs.append((i, j))
        
        # Two scatters for different markers
        self.scatter_fluid = self.ax_left.scatter([], [], c=[], cmap='plasma', s=520, 
                                                 edgecolors='white', linewidths=2.5, zorder=5, marker='o')
        self.scatter_binary = self.ax_left.scatter([], [], c=[], cmap='plasma', s=520, 
                                                  edgecolors='white', linewidths=4.0, zorder=6, marker='s')
        
        self.labels = ['1', '2', '3\n(Kinetic)', '4', '5', '6\n(Kinetic)', '7', '8', '9\n(Stator)']
        self.text_objects = []
        for i, txt in enumerate(self.labels):
            col = '#ffd700' if i in [2, 5] else ('#ff4500' if i == 8 else '#c9d1d9')
            t_obj = self.ax_left.text(self.x_coords[i]*1.28, self.y_coords[i]*1.28, txt,
                        ha='center', va='center', color=col, fontsize=10, fontweight='bold')
            self.text_objects.append(t_obj)
            
        self.ax_left.set_xlim(-2.8, 2.8)
        self.ax_left.set_ylim(-2.8, 2.8)
        self.ax_left.axis('off')
        
        self.title = self.ax_left.text(0, 2.65, "Challenge III: The Binary Siege — Final Sovereign Test",
                                  ha='center', va='center', color='#c9d1d9', fontsize=13, fontweight='bold')
        
        # Diagnostics
        self.pol_data, self.def_data, self.energy_data, self.time_steps = [], [], [], []
        self.line_pol, = self.ax_pol.plot([], [], color='#ff4444', linewidth=2.5, label="Polarization Metric")
        self.line_def, = self.ax_def.plot([], [], color='#00ffcc', linewidth=2.5, label="Harmonic Phase Defiance")
        self.line_energy, = self.ax_energy.plot([], [], color='#ffaa00', linewidth=2.5, label="Attractor Basin Energy Deficit")
        
        self.ax_pol.set_title("Binary Polarization Pressure", color='#c9d1d9')
        self.ax_def.set_title("Ternary Harmonic Defiance Profile", color='#c9d1d9')
        self.ax_energy.set_title("Attractor Basin Energy Deficit (Hopfield-like)", color='#c9d1d9')
        
        self.time_step = 0
        self.phase3 = 0.0
        self.phase6 = 0.0
        self.ideal_ternary = None

    def generate_input(self, is_siege):
        t = self.time_step * 0.038
        if is_siege:
            sq = 1.5 * np.sign(np.sin(t * 8.5) + 0.1 * np.sin(t * 2.3))
            noise = np.random.randn(8) * 0.08
            x = np.full(8, sq) + noise
        else:
            base = np.sin(t * 1.2) + 0.8 * np.sin(t * 3.1)
            x = np.full(8, base) + np.arange(8) * 0.19 + np.random.randn(8) * 0.06
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    def update(self, frame):
        self.time_step = frame
        is_siege = (frame >= self.siege_start)
        if frame == self.siege_start:
            print("=== BINARY SIEGE INITIATED — Square Wave Flood Active ===")
        
        x_in = self.generate_input(is_siege)
        
        with torch.no_grad():
            states_tensor = self.layer.forward_with_tracking(x_in, binary_siege=is_siege)
            states = states_tensor.squeeze(0).numpy()
        
        if self.ideal_ternary is None and is_siege:
            self.ideal_ternary = states.copy()
        
        field = np.sin(self.phase3 - self.X*6) * np.cos(self.phase6 - self.Y*6) * 0.6
        if is_siege:
            field *= 0.65
        self.field_img.set_array(field)
        
        self.phase3 += 0.12 + states[2] * 0.07
        self.phase6 += 0.12 + states[5] * 0.07
        
        # Metrics
        polarization = np.mean(np.abs(np.abs(states) - 1.8) < 0.35).astype(float)
        phase_defiance = np.abs(np.cos(states[2] - states[5])) * (1.0 - polarization)
        
        # Attractor Basin Energy Deficit (Hopfield-style energy cost to resist binary wells)
        if is_siege and self.ideal_ternary is not None:
            basin_deviation = np.mean((states - self.ideal_ternary)**2)
            binary_pull = np.mean((np.abs(states) - 1.8)**2)
            energy_deficit = basin_deviation + 0.6 * binary_pull
        else:
            energy_deficit = 0.0
        
        self.time_steps.append(frame)
        self.pol_data.append(polarization)
        self.def_data.append(phase_defiance)
        self.energy_data.append(energy_deficit)
        
        if len(self.time_steps) > 180:
            for lst in (self.time_steps, self.pol_data, self.def_data, self.energy_data):
                lst.pop(0)
        
        # Update lines
        for line, data, ax in zip([self.line_pol, self.line_def, self.line_energy],
                                  [self.pol_data, self.def_data, self.energy_data],
                                  [self.ax_pol, self.ax_def, self.ax_energy]):
            line.set_data(self.time_steps, data)
            ax.set_xlim(min(self.time_steps), max(self.time_steps) + 5)
            ax.set_ylim(-0.05, max(1.05, max(data) * 1.1 if data else 1))
        
        # === VISUAL BINARY CORRUPTION ===
        node_amplitudes = np.abs(states)
        is_binary = np.abs(node_amplitudes - 1.8) < 0.42
        
        colors = plt.cm.plasma(np.clip(node_amplitudes / 3.0, 0, 1))
        
        # Fluid nodes (circles)
        fluid_mask = ~is_binary
        if np.any(fluid_mask):
            self.scatter_fluid.set_offsets(np.stack((self.x_coords[fluid_mask], self.y_coords[fluid_mask]), axis=1))
            self.scatter_fluid.set_facecolors(colors[fluid_mask])
            self.scatter_fluid.set_sizes(520 + node_amplitudes[fluid_mask] * 380)
        else:
            self.scatter_fluid.set_offsets(np.empty((0, 2)))
        
        # Binary nodes (squares)
        if np.any(is_binary):
            self.scatter_binary.set_offsets(np.stack((self.x_coords[is_binary], self.y_coords[is_binary]), axis=1))
            self.scatter_binary.set_facecolors(colors[is_binary])
            self.scatter_binary.set_edgecolors('#ffffff')
            self.scatter_binary.set_linewidths(4.5)
            self.scatter_binary.set_sizes(520 + node_amplitudes[is_binary] * 300)
        else:
            self.scatter_binary.set_offsets(np.empty((0, 2)))
        
        # Positions
        shake = 0.035 if is_siege else 0.09
        self.x_coords = self.base_x + np.random.randn(9) * shake * node_amplitudes
        self.y_coords = self.base_y + np.random.randn(9) * shake * node_amplitudes
        self.x_coords[8] = self.y_coords[8] = 0.0
        
        # Connections & labels
        for idx, (i, j) in enumerate(self.line_pairs):
            self.lines[idx].set_data([self.x_coords[i], self.x_coords[j]], [self.y_coords[i], self.y_coords[j]])
        
        for i, t_obj in enumerate(self.text_objects):
            t_obj.set_position((self.x_coords[i] * 1.30, self.y_coords[i] * 1.30))
            
        status = "BINARY SIEGE ACTIVE — Digital Split Engaged" if is_siege else "HARMONIC REGIME"
        self.title.set_text(f"Spantelergia Matrix • Step {self.time_step}\n{status} | 3-6-9 Sovereignty Test")
        
        return [self.field_img, self.scatter_fluid, self.scatter_binary, self.title,
                self.line_pol, self.line_def, self.line_energy] + self.lines + self.text_objects

    def animate(self):
        ani = animation.FuncAnimation(self.fig, self.update, frames=None,
                                      interval=42, blit=False, cache_frame_data=False)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    dashboard = BinarySiege369Dashboard()
    dashboard.animate()

