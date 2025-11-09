import numpy as np
import matplotlib.pyplot as plt

class HopfOscillator:
    def __init__(self, dt, alpha=50.0):
        self.dt = dt
        self.alpha = alpha
        
        # State variables, initialized off the limit cycle to show convergence
        self.x = 0.1 
        self.y = 0.0
        
        # Target parameters
        self.mu = 1.0  # Target amplitude squared (A=1)
        self.omega = 2 * np.pi * 1.0  # Target frequency (1 Hz)

    def set_parameters(self, amplitude, frequency):
        """
        Set the target amplitude and frequency.
        """
        self.mu = amplitude**2
        self.omega = 2 * np.pi * frequency

    def step(self):
        """
        Update the oscillator's state for one time step (dt)
        using forward Euler integration.
        """
        r_sq = self.x**2 + self.y**2
        
        # Calculate derivatives
        x_dot = self.alpha * (self.mu - r_sq) * self.x - self.omega * self.y
        y_dot = self.alpha * (self.mu - r_sq) * self.y + self.omega * self.x
        
        # Update state
        self.x += x_dot * self.dt
        self.y += y_dot * self.dt
        
        # Return the current state
        return self.x, self.y
    def get_state(self):
        """Returns the current x, y state."""
        return np.array([self.x, self.y])

def main():
    # --- Simulation Parameters ---
    dt = 0.01  # Time step
    simulation_duration = 5.0 # seconds
    num_steps = int(simulation_duration / dt)

    # Create an oscillator instance
    oscillator = HopfOscillator(dt)

    # Set some initial parameters (you can play with these!)
    oscillator.set_parameters(amplitude=0.8, frequency=0.75) # 0.8 amplitude, 0.75 Hz

    # Store the history of x and y for plotting
    x_history = []
    y_history = []
    time_history = []

    # --- Simulate the oscillator ---
    for i in range(num_steps):
        x, y = oscillator.step()
        x_history.append(x)
        y_history.append(y)
        time_history.append(i * dt)

    # --- Plotting ---
    plt.figure(figsize=(12, 6))

    # Subplot 1: Phase Portrait (x vs. y)
    plt.subplot(1, 2, 1)
    plt.plot(x_history, y_history, label='Oscillator Trajectory')
    plt.scatter(x_history[0], y_history[0], color='red', marker='o', s=50, label='Start') # Mark starting point
    plt.scatter(x_history[-1], y_history[-1], color='green', marker='x', s=50, label='End') # Mark ending point

    # Draw the limit cycle circle
    amplitude = np.sqrt(oscillator.mu)
    circle = plt.Circle((0, 0), amplitude, color='gray', linestyle='--', fill=True, label=f'Limit Cycle (Amplitude={amplitude:.2f})')
    plt.gca().add_patch(circle)

    plt.title('Hopf Oscillator Phase Portrait (x vs. y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.axis('equal') # Ensure the circle looks like a circle
    plt.legend()

    # Subplot 2: Time Series Output (x and y over time)
    plt.subplot(1, 2, 2)
    plt.plot(time_history, x_history, label='x(t)')
    plt.plot(time_history, y_history, label='y(t)', linestyle='--')
    plt.title('Hopf Oscillator Time Series')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()