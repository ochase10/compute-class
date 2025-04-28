import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
# Force matplotlib to use Agg backend to avoid display issues
matplotlib.use('Agg')

# Constants in SI units
G = 6.67430e-11  # gravitational constant [m^3 kg^-1 s^-2]
M_sun = 1.989e30  # solar mass [kg]
AU = 1.496e11     # astronomical unit [m]
year = 365.25 * 24 * 3600  # year in seconds

def acceleration(r, M):
    """
    Calculate the gravitational acceleration for a two-body problem.
    
    Args:
        r: position vector [x, y, z]
        M: central mass
    
    Returns:
        acceleration vector
    """
    r_norm = np.linalg.norm(r)
    return -G * M * r / (r_norm**3)

def rk4_step(r, v, dt, M):
    """
    Perform one step of the 4th order Runge-Kutta integration.
    
    Args:
        r: position vector [x, y, z]
        v: velocity vector [vx, vy, vz]
        dt: time step
        M: central mass
    
    Returns:
        new position and velocity vectors
    """
    # First step
    a1 = acceleration(r, M)
    k1r = v
    k1v = a1
    
    # Second step
    a2 = acceleration(r + 0.5 * dt * k1r, M)
    k2r = v + 0.5 * dt * k1v
    k2v = a2
    
    # Third step
    a3 = acceleration(r + 0.5 * dt * k2r, M)
    k3r = v + 0.5 * dt * k2v
    k3v = a3
    
    # Fourth step
    a4 = acceleration(r + dt * k3r, M)
    k4r = v + dt * k3v
    k4v = a4
    
    # Update position and velocity
    r_new = r + dt * (k1r + 2 * k2r + 2 * k3r + k4r) / 6
    v_new = v + dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6
    
    return r_new, v_new

def initialize_orbit(a, e, M):
    """
    Initialize the orbit at perihelion.
    
    Args:
        a: semi-major axis [m]
        e: eccentricity
        M: central mass [kg]
    
    Returns:
        initial position and velocity vectors
    """
    # Perihelion distance
    r_p = a * (1 - e)
    
    # Initial position (perihelion is along the x-axis)
    r0 = np.array([r_p, 0.0, 0.0])
    
    # Velocity at perihelion
    v_p = np.sqrt(G * M * (2/r_p - 1/a))
    v0 = np.array([0.0, v_p, 0.0])
    
    return r0, v0

def integrate_orbit(a, e, M, T, num_steps):
    """
    Integrate the orbit for time T using RK4.
    
    Args:
        a: semi-major axis [m]
        e: eccentricity
        M: central mass [kg]
        T: total integration time [s]
        num_steps: number of integration steps
    
    Returns:
        arrays of time, positions, velocities, energy, and angular momentum
    """
    dt = T / num_steps
    
    # Initialize arrays to store results
    times = np.zeros(num_steps + 1)
    positions = np.zeros((num_steps + 1, 3))
    velocities = np.zeros((num_steps + 1, 3))
    energies = np.zeros(num_steps + 1)
    ang_momenta = np.zeros(num_steps + 1)
    
    # Initialize the orbit
    r, v = initialize_orbit(a, e, M)
    
    # Store initial values
    positions[0] = r
    velocities[0] = v
    
    # Calculate initial energy and angular momentum
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    energies[0] = 0.5 * v_norm**2 - G * M / r_norm
    ang_momenta[0] = np.linalg.norm(np.cross(r, v))
    
    # Integrate the orbit
    for i in range(num_steps):
        times[i+1] = (i+1) * dt
        r, v = rk4_step(r, v, dt, M)
        
        positions[i+1] = r
        velocities[i+1] = v
        
        # Calculate energy and angular momentum
        r_norm = np.linalg.norm(r)
        v_norm = np.linalg.norm(v)
        energies[i+1] = 0.5 * v_norm**2 - G * M / r_norm
        ang_momenta[i+1] = np.linalg.norm(np.cross(r, v))
    
    return times, positions, velocities, energies, ang_momenta

def plot_results(times, positions, energies, ang_momenta):
    """
    Plot the results of the integration.
    
    Args:
        times: array of time points
        positions: array of position vectors
        energies: array of energy values
        ang_momenta: array of angular momentum values
    """
    # Convert to appropriate units
    times_days = times / (24 * 3600)  # convert to days
    positions_au = positions / AU  # convert to AU
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot the orbit
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(positions_au[:, 0], positions_au[:, 1])
    ax1.plot(0, 0, 'yo', markersize=10)  # Plot the Sun
    ax1.set_xlabel('x [AU]')
    ax1.set_ylabel('y [AU]')
    ax1.set_title('Orbit')
    ax1.grid(True)
    ax1.set_aspect('equal')
    
    # Plot the energy
    ax2 = fig.add_subplot(2, 2, 2)
    relative_energy_error = (energies - energies[0]) / abs(energies[0])
    ax2.plot(times_days, relative_energy_error)
    ax2.set_xlabel('Time [days]')
    ax2.set_ylabel('Relative Energy Error')
    ax2.set_title('Energy Conservation')
    ax2.grid(True)
    
    # Plot the angular momentum
    ax3 = fig.add_subplot(2, 2, 3)
    relative_momentum_error = (ang_momenta - ang_momenta[0]) / ang_momenta[0]
    ax3.plot(times_days, relative_momentum_error)
    ax3.set_xlabel('Time [days]')
    ax3.set_ylabel('Relative Angular Momentum Error')
    ax3.set_title('Angular Momentum Conservation')
    ax3.grid(True)
    
    # Plot the distance from the Sun
    ax4 = fig.add_subplot(2, 2, 4)
    r = np.sqrt(np.sum(positions**2, axis=1)) / AU
    ax4.plot(times_days, r)
    ax4.set_xlabel('Time [days]')
    ax4.set_ylabel('Distance [AU]')
    ax4.set_title('Distance from the Sun')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('kepler_orbit.png', dpi=300)
    plt.show()

def animate_orbit(positions):
    """
    Create an animation of the orbit and save as a GIF.
    
    Args:
        positions: array of position vectors
    
    Returns:
        None
    """
    positions_au = positions / AU
    
    # Since animation is causing issues, let's create a simpler version
    # by manually creating a sequence of frames and saving them using PIL
    try:
        from PIL import Image
        import io
        
        print("Creating frames for animation...")
        num_frames = 50
        frame_indices = np.linspace(0, len(positions_au) - 1, num_frames).astype(int)
        images = []
        
        for i, idx in enumerate(frame_indices):
            # Create a new figure for each frame
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(min(positions_au[:, 0]) - 0.1, max(positions_au[:, 0]) + 0.1)
            ax.set_ylim(min(positions_au[:, 1]) - 0.1, max(positions_au[:, 1]) + 0.1)
            ax.set_xlabel('x [AU]')
            ax.set_ylabel('y [AU]')
            ax.set_title('Kepler Orbit Animation')
            ax.grid(True)
            ax.set_aspect('equal')
            
            # Plot the orbit path
            ax.plot(positions_au[:, 0], positions_au[:, 1], 'b-', alpha=0.3)
            
            # Plot the Sun
            ax.plot(0, 0, 'yo', markersize=10)
            
            # Plot the planet at the current position
            ax.plot(positions_au[idx, 0], positions_au[idx, 1], 'ro', markersize=5)
            
            # Save the figure to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            images.append(img.copy())
            plt.close(fig)
            buf.close()
            
            # Show progress
            if i % 10 == 0:
                print(f"Created frame {i+1}/{num_frames}")
        
        # Save the frames as an animated GIF
        print("Saving animation as kepler_orbit_animation.gif...")
        images[0].save('kepler_orbit_animation.gif', save_all=True, append_images=images[1:], 
                        optimize=False, duration=100, loop=0)
        print("Animation saved successfully!")
        
    except ImportError:
        print("Error: Could not import PIL (pillow). Please install with: pip install pillow")
    except Exception as e:
        print(f"Error creating animation: {e}")
        
    # Create a static plot as a fallback
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(min(positions_au[:, 0]) - 0.1, max(positions_au[:, 0]) + 0.1)
    ax.set_ylim(min(positions_au[:, 1]) - 0.1, max(positions_au[:, 1]) + 0.1)
    ax.set_xlabel('x [AU]')
    ax.set_ylabel('y [AU]')
    ax.set_title('Kepler Orbit')
    ax.grid(True)
    ax.set_aspect('equal')
    
    # Plot the orbit
    ax.plot(positions_au[:, 0], positions_au[:, 1], 'b-')
    
    # Plot the Sun
    ax.plot(0, 0, 'yo', markersize=10, label='Sun')
    
    # Mark perihelion and aphelion
    r = np.sum(positions_au**2, axis=1)
    perihelion_idx = np.argmin(r)
    aphelion_idx = np.argmax(r)
    
    ax.plot(positions_au[perihelion_idx, 0], positions_au[perihelion_idx, 1], 'go', markersize=7, label='Perihelion')
    ax.plot(positions_au[aphelion_idx, 0], positions_au[aphelion_idx, 1], 'mo', markersize=7, label='Aphelion')
    
    ax.legend()
    plt.savefig('kepler_orbit_path.png', dpi=300)
    print("Saved static orbit plot as kepler_orbit_path.png")
    plt.close()

def main():
    # Parameters
    a = 1.0 * AU  # semi-major axis in meters
    e = 0.96      # eccentricity
    mass = M_sun   # central mass (solar mass)
    T = year      # integration time (1 year)
    num_steps = 10000  # number of integration steps
    
    print("Integrating orbit...")
    # Integrate the orbit
    times, positions, velocities, energies, ang_momenta = integrate_orbit(a, e, mass, T, num_steps)
    
    print("Plotting and saving results...")
    # Plot the results and save the figure
    plot_results(times, positions, energies, ang_momenta)
    
    # Create and save animation
    print("Creating orbit animation...")
    animate_orbit(positions)
    
    # Calculate the orbital period from Kepler's third law
    orbital_period = 2 * np.pi * np.sqrt(a**3 / (G * mass)) / (24 * 3600)  # in days
    print(f"Theoretical orbital period: {orbital_period:.2f} days")
    
    # Calculate actual period from the simulation data
    # (distance from the central point will be minimum at perihelion)
    r = np.sqrt(np.sum(positions**2, axis=1))
    # Find local minima in distance (perihelion passages)
    perihelion_indices = []
    for i in range(1, len(r) - 1):
        if r[i] < r[i-1] and r[i] < r[i+1]:
            perihelion_indices.append(i)
    
    if len(perihelion_indices) >= 2:
        actual_period = (times[perihelion_indices[1]] - times[perihelion_indices[0]]) / (24 * 3600)
        print(f"Simulated orbital period: {actual_period:.2f} days")
    
    # Calculate error in energy and angular momentum conservation
    max_energy_error = np.max(np.abs((energies - energies[0]) / abs(energies[0])))
    max_momentum_error = np.max(np.abs((ang_momenta - ang_momenta[0]) / ang_momenta[0]))
    
    print(f"Maximum relative energy error: {max_energy_error:.2e}")
    print(f"Maximum relative angular momentum error: {max_momentum_error:.2e}")
    
    print("\nFiles saved:")
    print("- kepler_orbit.png: Orbit plots and analysis")
    print("- kepler_orbit_path.png: Static visualization of the orbit path")
    print("- kepler_orbit_animation.gif: Animated visualization of the orbit (if successful)")

if __name__ == "__main__":
    main()