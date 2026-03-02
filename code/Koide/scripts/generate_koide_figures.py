import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import 
from matplotlib.patches import Circle 
from matplotlib.animation import FuncAnimation, PillowWriter


# Real PDG values (from your paper)
m_e = 0.51099895
m_mu = 105.6583755
m_tau = 1776.86

# 120° angles
phi = np.deg2rad([0, 120, 240])

# Geometry for exact Q=2/3 and 45° Foot
y_over_x = np.sqrt(2)
factor_heavy = 1 - 0.5 * y_over_x
x = np.sqrt(np.sqrt(m_mu * m_tau)) / factor_heavy
y = y_over_x * x
sqrt_m_sym = x + y * np.cos(phi)

# 1. Isolated 3D Flavor Space
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter([np.sqrt(m_e)], [np.sqrt(m_mu)], [np.sqrt(m_tau)], c='blue', s=100, label='Real leptons')
ax.scatter([sqrt_m_sym[0]], [sqrt_m_sym[1]], [sqrt_m_sym[2]], c='red', s=200, marker='*', label='Toy symmetric')
ax.plot([0,50],[0,50],[0,50], 'k--', lw=2, label='(1,1,1) direction')
ax.set_xlabel(r'$\sqrt{m_e}$')
ax.set_ylabel(r'$\sqrt{m_\mu}$')
ax.set_zlabel(r'$\sqrt{m_\tau}$')
ax.legend()
ax.view_init(elev=20, azim=135)
plt.savefig('3D_flavor_space.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 2. Isolated 2D Holographic Screen
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
circle = Circle((0,0), 1.05, color='lightblue', fill=False, lw=3)
ax.add_patch(circle)
ax.scatter([1, -0.5, -0.5], [0, 0.866, -0.866], c='red', s=150)
ax.text(1.15, 0, 'e', fontsize=18, color='darkred', fontweight='bold')
ax.text(-0.65, 0.98, 'μ', fontsize=18, color='darkred', fontweight='bold')
ax.text(-0.65, -0.98, 'τ', fontsize=18, color='darkred', fontweight='bold')
ax.set_aspect('equal')
ax.axis('off')
plt.savefig('2D_holographic_screen.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 3. Animated Slider GIF
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter([np.sqrt(m_e)], [np.sqrt(m_mu)], [np.sqrt(m_tau)], c='blue', s=100, label='Real leptons')
ax.scatter([sqrt_m_sym[0]], [sqrt_m_sym[1]], [sqrt_m_sym[2]], c='red', s=200, marker='*', label='Toy symmetric')
ax.plot([0,50],[0,50],[0,50], 'k--', lw=2)
ax.set_xlabel(r'$\sqrt{m_e}$')
ax.set_ylabel(r'$\sqrt{m_\mu}$')
ax.set_zlabel(r'$\sqrt{m_\tau}$')
ax.legend()
ax.view_init(elev=20, azim=135)

moving_pt, = ax.plot([sqrt_m_sym[0]], [sqrt_m_sym[1]], [sqrt_m_sym[2]], 'go', ms=14)

def animate(frame):
    delta = frame / 59.0
    m_b = np.array([m_e, m_mu * (1 - delta*0.6), m_tau * (1 + delta*3.0)])
    s_b = np.sqrt(m_b)
    moving_pt.set_data_3d([s_b[0]], [s_b[1]], [s_b[2]])
    ax.set_title(f'Breaking δ = {delta:.2f}')
    return moving_pt,

ani = FuncAnimation(fig, animate, frames=60, interval=80, blit=False)
ani.save('koide_slider_animation.gif', writer=PillowWriter(fps=12), dpi=120)
plt.close(fig)

print("All files created successfully:")
print(" - 3D_flavor_space.png")
print(" - 2D_holographic_screen.png")
print(" - koide_slider_animation.gif")