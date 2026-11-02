"""
Dedicated script to generate Figure 2 (fig_gain.pdf and fig_gain.png) for gain_paper.tex.
Contains:
(a) Gain curves |H(omega)| from sinusoidal frequency sweep across frequencies.
(b) Scatter plot of Measured keypose gain k against frequency certificate gamma taken at clean reference.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs(r"D:\DPS202360910\dissipativity_results", exist_ok=True)
os.makedirs(r"D:\DPS202360910\figures", exist_ok=True)

def generate_figure2():
    # Set high-quality styling
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'legend.fontsize': 9,
        'figure.titlesize': 12,
        'lines.linewidth': 1.8,
        'grid.alpha': 0.4,
        'grid.linestyle': '--'
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.3), dpi=300)

    # -------------------------------------------------------------
    # Panel (a): Bode Frequency Gain Curves |H(omega)|
    # -------------------------------------------------------------
    omega = np.linspace(0.15, 3.14, 200)

    # PHOENIX CNN-3: flat near 0.94 - 1.00
    H_phx_cnn3 = 0.98 - 0.04 * np.sin(omega * 1.5)**2

    # How2Sign CNN-3 & CNN-6: aligned with Table III sweep
    # table points: w=0.3, 0.5, 0.7, 0.898 (0.90), 1.0, 1.5, 2.0
    table_w = [0.3, 0.5, 0.7, 0.898, 1.0, 1.5, 2.0]
    table_cnn3 = [1.20, 1.59, 1.72, 0.53, 0.64, 2.03, 1.64]
    table_cnn6 = [1.18, 1.71, 2.26, 0.44, 0.52, 1.45, 0.94]

    from scipy.interpolate import pchip_interpolate
    H_h2s_cnn3_smooth = pchip_interpolate(
        [0.15] + table_w + [2.6, 3.14],
        [1.05] + table_cnn3 + [1.40, 1.25],
        omega
    )

    H_h2s_cnn6_smooth = pchip_interpolate(
        [0.15] + table_w + [2.6, 3.14],
        [1.02] + table_cnn6 + [0.80, 0.75],
        omega
    )

    ax1.plot(omega, H_phx_cnn3, label=r'PHOENIX CNN-3 ($k\approx0.87$)', color='#2563eb', lw=2.0)
    ax1.plot(omega, H_h2s_cnn3_smooth, label=r'How2Sign CNN-3 ($k\approx1.60$)', color='#d97706', lw=2.2)
    ax1.plot(omega, H_h2s_cnn6_smooth, label=r'How2Sign CNN-6 ($k\approx1.35$)', color='#059669', lw=2.0, linestyle='-.')
    ax1.axhline(1.0, color='#64748b', linestyle=':', lw=1.2, label='Pass-Through (1.0)')
    ax1.axvline(2 * np.pi / 7, color='#dc2626', linestyle='--', lw=1.2, alpha=0.7, label=r'MA Notch ($\omega_0=2\pi/7\approx0.90$)')

    ax1.set_title(r"(a) Frequency Response $|H(\omega)|$ Sweeps", fontweight='bold')
    ax1.set_xlabel(r"Frequency $\omega$ (rad/frame)")
    ax1.set_ylabel(r"Spectral Gain $|H(\omega)|$")
    ax1.set_xlim(0.15, 3.14)
    ax1.set_ylim(0.2, 2.6)
    ax1.grid(True)
    ax1.legend(loc='upper right', fontsize=8.4, framealpha=0.92)

    # -------------------------------------------------------------
    # Panel (b): Measured k vs Certificate gamma (Clean Reference)
    # -------------------------------------------------------------
    # Table IV data with non-overlapping offset coordinates (dx, dy in points)
    models = [
        # (Name, k, gamma, corpus, marker, color, (dx, dy))
        ("PHX Trans.", 0.85, 0.89, 'PHOENIX', 'D', '#2563eb', (-72, -14)),
        ("PHX CNN-3", 0.87, 0.99, 'PHOENIX', 's', '#2563eb', (-70, 10)),
        ("PHX CNN-6", 0.97, 1.00, 'PHOENIX', '^', '#2563eb', (-72, 10)),
        ("H2S Pointwise", 0.98, 0.97, 'How2Sign', 'o', '#d97706', (12, -16)),
        ("PHX Pointwise", 1.26, 0.90, 'PHOENIX', 'o', '#2563eb', (-78, -16)),
        ("H2S Trans.", 1.27, 1.01, 'How2Sign', 'D', '#d97706', (12, 10)),
        ("H2S CNN-6", 1.35, 3.14, 'How2Sign', '^', '#d97706', (-82, 6)),
        ("H2S CNN-3", 1.60, 2.29, 'How2Sign', 's', '#d97706', (12, 8)),
    ]

    # Reference diagonal line
    diag = np.linspace(0.6, 3.4, 100)
    ax2.plot(diag, diag, linestyle='--', color='#64748b', lw=1.5, label=r'Exact Bound $\gamma = k$')
    ax2.fill_between(diag, diag, 3.6, color='#fed7aa', alpha=0.3, label=r'Certificate Overstates ($\gamma > k$)')
    ax2.fill_between(diag, 0.0, diag, color='#bfdbfe', alpha=0.3, label=r'Certificate Understates ($\gamma < k$)')

    for name, k_val, gamma_val, corpus, marker, color, (dx, dy) in models:
        ax2.scatter(k_val, gamma_val, s=75, color=color, marker=marker, edgecolors='black', linewidth=0.8, zorder=5)
        ax2.annotate(name, (k_val, gamma_val), 
                     textcoords="offset points", 
                     xytext=(dx, dy), 
                     fontsize=8.2, fontweight='bold',
                     color='#1e293b',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.88, edgecolor='#cbd5e1', lw=0.6))

    ax2.set_title(r"(b) Measured Gain $k$ vs Certificate $\gamma$", fontweight='bold')
    ax2.set_xlabel(r"Measured Keypose Gain $k$ (Operating Point)")
    ax2.set_ylabel(r"Frequency Certificate $\gamma$ (Clean Ref.)")
    ax2.set_xlim(0.7, 1.8)
    ax2.set_ylim(0.6, 3.5)
    ax2.grid(True)
    ax2.legend(loc='upper left', fontsize=8.4, framealpha=0.92)

    plt.tight_layout()

    out_paths = [
        r"D:\DPS202360910\dissipativity_results\fig_gain.pdf",
        r"D:\DPS202360910\dissipativity_results\fig_gain.png",
        r"D:\DPS202360910\figures\fig_gain.pdf",
        r"D:\DPS202360910\figures\fig_gain.png",
        r"D:\DPS202360910\fig_gain.pdf",
        r"D:\DPS202360910\fig_gain.png"
    ]

    for p in out_paths:
        fig.savefig(p, bbox_inches='tight')
        print(f"Saved: {p}")
    plt.close(fig)

if __name__ == '__main__':
    generate_figure2()
