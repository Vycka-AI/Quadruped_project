import matplotlib
matplotlib.use('TkAgg') # <-- ADD THIS LINE

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

class HopfOscillator:
    def __init__(self, dt, alpha=50.0):
        self.dt = dt
        self.alpha = alpha
        self.reset_state() # Initialize state

        # Default parameters
        self.mu = 1.0  # Target amplitude squared
        self.omega = 2 * np.pi * 1.0  # Target frequency

    def reset_state(self):
        """Resets the oscillator's internal state."""
        self.x = 0.1 # Start off the limit cycle
        self.y = 0.0

    def set_parameters(self, amplitude, frequency):
        """Set the target amplitude and frequency."""
        self.mu = amplitude**2
        self.omega = 2 * np.pi * frequency

    def step(self):
        """Update the oscillator's state for one time step (dt)."""
        r_sq = self.x**2 + self.y**2
        
        x_dot = self.alpha * (self.mu - r_sq) * self.x - self.omega * self.y
        y_dot = self.alpha * (self.mu - r_sq) * self.y + self.omega * self.x
        
        self.x += x_dot * self.dt
        self.y += y_dot * self.dt
        
        return self.x, self.y

# --- Simulation Parameters ---
dt = 0.01  # Time step for oscillator update
history_len = 500 # Number of points to keep in history for plots (corresponds to 5 seconds if dt=0.01)

# Create the oscillator instance
oscillator = HopfOscillator(dt)

# --- Set up the Plot ---
fig, (ax_phase, ax_time) = plt.subplots(1, 2, figsize=(14, 7))
plt.suptitle('Interactive Hopf Oscillator')

# Phase Portrait Plot (x vs. y)
line_phase, = ax_phase.plot([], [], 'b-', lw=1, alpha=0.8)
start_point_phase, = ax_phase.plot([], [], 'ro', markersize=8) # To show the reset point
limit_circle = plt.Circle((0, 0), 1.0, color='gray', linestyle='--', fill=False) # Initial dummy circle
ax_phase.add_patch(limit_circle)
ax_phase.set_title('Phase Portrait (x vs. y)')
ax_phase.set_xlabel('x')
ax_phase.set_ylabel('y')
ax_phase.grid(True)
ax_phase.axhline(0, color='black', linewidth=0.5)
ax_phase.axvline(0, color='black', linewidth=0.5)
ax_phase.set_aspect('equal', adjustable='box')
ax_phase.set_xlim(-1.5, 1.5) # Max amplitude we expect is 1.0, so +/- 1.5
ax_phase.set_ylim(-1.5, 1.5)

# Time Series Plot (x and y over time)
time_vec = np.linspace(-history_len * dt, 0, history_len) # Time axis relative to current
line_x_time, = ax_time.plot(time_vec, np.zeros(history_len), 'b-', label='x(t)')
line_y_time, = ax_time.plot(time_vec, np.zeros(history_len), 'r--', label='y(t)')
ax_time.set_title('Time Series Output')
ax_time.set_xlabel('Time (s)')
ax_time.set_ylabel('Amplitude')
ax_time.grid(True)
ax_time.set_ylim(-1.5, 1.5) # Max amplitude we expect is 1.0, so +/- 1.5
ax_time.legend()

# --- Sliders ---
axcolor = 'lightgoldenrodyellow'
ax_amp = fig.add_axes([0.15, 0.05, 0.3, 0.03], facecolor=axcolor) # [left, bottom, width, height]
ax_freq = fig.add_axes([0.55, 0.05, 0.3, 0.03], facecolor=axcolor)

s_amp = Slider(ax_amp, 'Amplitude', 0.1, 1.2, valinit=0.8, valstep=0.05)
s_freq = Slider(ax_freq, 'Frequency (Hz)', 0.1, 3.0, valinit=0.75, valstep=0.05)

# --- Reset Button ---
ax_reset = fig.add_axes([0.02, 0.05, 0.08, 0.03])
button_reset = Button(ax_reset, 'Reset', color=axcolor, hovercolor='0.975')

# --- Data storage for animation ---
x_history = [oscillator.x] * history_len
y_history = [oscillator.y] * history_len

# --- Update function for animation ---
def update(frame):
    current_amplitude = s_amp.val
    current_frequency = s_freq.val
    
    oscillator.set_parameters(current_amplitude, current_frequency)
    x, y = oscillator.step()

    # Update history
    x_history.pop(0)
    y_history.pop(0)
    x_history.append(x)
    y_history.append(y)

    # Update Phase Portrait
    line_phase.set_data(x_history, y_history)
    start_point_phase.set_data([x_history[0]], [y_history[0]]) # Keep start point updated
    limit_circle.set_radius(current_amplitude)

    # Update Time Series
    line_x_time.set_ydata(x_history)
    line_y_time.set_ydata(y_history)

    return line_phase, start_point_phase, limit_circle, line_x_time, line_y_time

# --- Reset function for button ---
def reset_oscillator(event):
    oscillator.reset_state()
    global x_history, y_history
    x_history = [oscillator.x] * history_len
    y_history = [oscillator.y] * history_len
    # Need to manually update plots here or let the next animation frame handle it.
    # The animation will pick up the new history on its next frame.
    s_amp.reset() # Reset sliders to initial values
    s_freq.reset()

button_reset.on_clicked(reset_oscillator)

# Create the animation
ani = FuncAnimation(fig, update, interval=dt*1000, blit=True) # interval in ms

plt.show()