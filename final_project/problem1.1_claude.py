import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import time


def main():
    """
    Generate a spherical, isotropic Hernquist halo with positions and velocities.
    Parameters:
    - M = 10^12 M_sun (total mass)
    - a = 35 kpc (scale radius)
    - N = 10^6 (number of particles)
    """
    # Start timing
    start_time = time.time()
    
    # Set parameters
    M = 1e12  # Total mass in solar masses
    a = 35    # Scale radius in kpc
    N = int(1e6)  # Number of particles
    G = 4.3e-6    # Gravitational constant in kpc^3 M_sun^-1 Gyr^-2
    
    print(f"Generating Hernquist halo with M={M:.1e} M_sun, a={a} kpc, N={N}")
    
    # Step 1: Generate particle positions
    positions = generate_hernquist_positions(M, a, N)
    r = np.sqrt(np.sum(positions**2, axis=1))
    
    # Step 2: Calculate potential at each position
    potential = hernquist_potential(r, M, a)
    
    # Step 3: Generate velocities using the improved method
    velocities = generate_velocities_improved(r, potential, M, a)
    
    # Step 4: Verify the velocity dispersion profile
    verify_velocity_dispersion(r, positions, velocities, M, a)
    
    # Save the data
    save_data(positions, velocities, M, a)
    
    # Report time taken
    print(f"Time taken: {time.time() - start_time:.2f} seconds")


def hernquist_density(r, M, a):
    """
    Calculate Hernquist density at radius r.
    
    Parameters:
    - r: Radius in kpc
    - M: Total mass in solar masses
    - a: Scale radius in kpc
    
    Returns:
    - Density in M_sun/kpc^3
    """
    return M / (2 * np.pi) * a / (r * (r + a)**3)


def hernquist_mass(r, M, a):
    """
    Calculate enclosed mass at radius r for a Hernquist profile.
    
    Parameters:
    - r: Radius in kpc
    - M: Total mass in solar masses
    - a: Scale radius in kpc
    
    Returns:
    - Enclosed mass in solar masses
    """
    return M * r**2 / (r + a)**2


def hernquist_potential(r, M, a):
    """
    Calculate gravitational potential at radius r for a Hernquist profile.
    
    Parameters:
    - r: Radius in kpc
    - M: Total mass in solar masses
    - a: Scale radius in kpc
    
    Returns:
    - Potential in kpc^2/Gyr^2
    """
    G = 4.3e-6  # Gravitational constant in kpc^3 M_sun^-1 Gyr^-2
    return -G * M / (r + a)


def hernquist_df(E, M, a):
    """
    Hernquist distribution function f(E).
    
    Parameters:
    - E: Energy (not normalized)
    - M: Total mass in solar masses
    - a: Scale radius in kpc
    
    Returns:
    - f(E) value
    """
    G = 4.3e-6  # Gravitational constant in kpc^3 M_sun^-1 Gyr^-2
    # Normalize energy
    E_norm = E * a / (G * M)
    
    # Check if energy is valid (bound particles have E < 0)
    if np.any(E_norm >= 0):
        if isinstance(E_norm, np.ndarray):
            result = np.zeros_like(E_norm, dtype=float)
            mask = E_norm < 0
            valid_E = E_norm[mask]
        else:
            return 0
    else:
        result = np.zeros_like(E_norm, dtype=float)
        mask = np.ones_like(E_norm, dtype=bool)
        valid_E = E_norm
    
    # Hernquist DF (Equation 17 in Hernquist 1990)
    s = np.sqrt(-valid_E)
    term1 = 1 / (1 - s**2)**2.5
    term2 = 3 * np.arcsin(s)
    term3 = s * np.sqrt(1 - s**2) * (1 - 2 * s**2) * (8 * s**4 - 8 * s**2 - 3)
    
    valid_result = M / (8 * np.sqrt(2) * np.pi**3 * a**3 * (G * M / a)**1.5) * term1 * (term2 + term3)
    
    if isinstance(E_norm, np.ndarray):
        result[mask] = valid_result
        return result
    else:
        return valid_result


def distribution_function_times_density(E, phi, M, a):
    """
    Calculate f(E) * p(E|phi) which is proportional to f(E) * sqrt(phi - E).
    This version works with arrays for efficient computation.
    
    Parameters:
    - E: Energy or array of energies
    - phi: Potential energy
    - M: Total mass in solar masses
    - a: Scale radius in kpc
    
    Returns:
    - f(E) * p(E|phi) value or array of values
    """
    # Handle array input
    if isinstance(E, np.ndarray):
        result = np.zeros_like(E, dtype=float)
        # Only bound particles (E < phi)
        mask = E < phi
        if np.any(mask):
            result[mask] = hernquist_df(E[mask], M, a) * np.sqrt(phi - E[mask])
        return result
    else:
        # Handle scalar input
        if E >= phi:
            return 0
        return hernquist_df(E, M, a) * np.sqrt(phi - E)


def generate_hernquist_positions(M, a, N):
    """
    Generate particle positions following a Hernquist profile.
    
    Parameters:
    - M: Total mass in solar masses
    - a: Scale radius in kpc
    - N: Number of particles
    
    Returns:
    - Nx3 array of positions in kpc
    """
    # Generate random radii using the inverse CDF method
    u = np.random.random(N)
    r = a * u**0.5 / (1 - u**0.5)
    
    # Generate random angles
    phi = 2 * np.pi * np.random.random(N)
    cos_theta = 2 * np.random.random(N) - 1
    sin_theta = np.sqrt(1 - cos_theta**2)
    
    # Convert to Cartesian coordinates
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * cos_theta
    
    positions = np.column_stack((x, y, z))
    
    print(f"Positions generated. Radii range: {np.min(r):.2f} - {np.max(r):.2f} kpc")
    
    return positions


def generate_velocities_improved(r, potential, M, a, batch_size=10000):
    """
    Generate velocities using an improved sampling method.
    
    Parameters:
    - r: Array of radii in kpc
    - potential: Array of potential energies
    - M: Total mass in solar masses
    - a: Scale radius in kpc
    - batch_size: Number of particles to process at once
    
    Returns:
    - Nx3 array of velocities in kpc/Gyr
    """
    N = len(r)
    velocities = np.zeros((N, 3))
    G = 4.3e-6  # Gravitational constant in kpc^3 M_sun^-1 Gyr^-2
    
    print("Generating velocities using improved method...")
    
    # Process particles in batches
    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        n_batch = end_idx - start_idx
        
        print(f"Processing particles {start_idx+1}-{end_idx} out of {N}...")
        
        # Get potentials for this batch
        phi_batch = potential[start_idx:end_idx]
        
        # Generate velocity magnitudes for this batch
        v_mag = np.zeros(n_batch)
        
        # Process each particle individually but with optimized sampling approach
        for i in range(n_batch):
            phi = phi_batch[i]
            radius = r[start_idx + i]
            
            # Define energy range for this particle
            # E_min should be well below phi (more negative)
            E_min = phi  # E min is phi
            E_max = 0  # Upper bound is E=0
            
            # Create a grid with more points near E_max where the DF rises sharply
            n_grid = 1000
            # Use a grid that's concentrated near E_max
            q = np.linspace(0, 1, n_grid)**2  # Square to concentrate points near 1
            E_grid = E_min + q * (E_max - E_min)
            
            # Calculate f(E)*p(E|phi) on the grid
            f_values = distribution_function_times_density(E_grid, phi, M, a)
            
            # Create CDF
            cdf = np.cumsum(f_values)
            if cdf[-1] > 0:  # Ensure we have valid values
                cdf = cdf / cdf[-1]  # Normalize
                
                # Generate random number
                u = np.random.random()
                
                # Find corresponding energy through interpolation
                idx = np.searchsorted(cdf, u)
                if idx == 0:
                    sampled_E = E_grid[0]
                elif idx == n_grid:
                    sampled_E = E_grid[-1]
                else:
                    # Linear interpolation
                    alpha = (u - cdf[idx-1]) / (cdf[idx] - cdf[idx-1])
                    sampled_E = E_grid[idx-1] + alpha * (E_grid[idx] - E_grid[idx-1])
                
                # Convert energy to velocity magnitude
                v_sq = 2 * (phi - sampled_E)
                v_mag[i] = np.sqrt(max(0, v_sq))  # Guard against negative values due to numerical issues
            else:
                # Fallback: use the theoretical velocity dispersion at this radius
                sigma = hernquist_velocity_dispersion(radius, M, a)
                v_mag[i] = np.random.normal(0, sigma)
        
        # Generate random directions for this batch
        phi_angle = 2 * np.pi * np.random.random(n_batch)
        cos_theta = 2 * np.random.random(n_batch) - 1
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        # Set velocity components
        vx = v_mag * sin_theta * np.cos(phi_angle)
        vy = v_mag * sin_theta * np.sin(phi_angle)
        vz = v_mag * cos_theta
        
        velocities[start_idx:end_idx, 0] = vx
        velocities[start_idx:end_idx, 1] = vy
        velocities[start_idx:end_idx, 2] = vz
    
    print("Velocities generated.")
    return velocities


def hernquist_velocity_dispersion(r, M, a):
    """
    Calculate the theoretical velocity dispersion at radius r.
    
    Parameters:
    - r: Radius in kpc (scalar or array)
    - M: Total mass in solar masses
    - a: Scale radius in kpc
    
    Returns:
    - Velocity dispersion in kpc/Gyr according to Eq. 10 from Hernquist 1990
    """
    G = 4.3e-6  # Gravitational constant in kpc^3 M_sun^-1 Gyr^-2
    
    # Implement the correct formula from Hernquist 1990 Eq. 10
    # σ²(r) = (GM/2) * [r/(r+a)]
    sigma_squared = G*M / 12 / a * (12*r * (r+a)**3 / a**4 * np.log((r+a)/r) \
             - r / (r+a) * (25 + 52* r/a + 42* (r/a)**2 + 12* (r/a)**3))
    
    # Ensure all values are positive
    if isinstance(sigma_squared, np.ndarray):
        sigma_squared = np.maximum(sigma_squared, 1e-10)
    else:
        sigma_squared = max(sigma_squared, 1e-10)
        
    return np.sqrt(sigma_squared)


def verify_velocity_dispersion(r, positions, velocities, M, a, n_bins=50, max_radius=1e4):
    """
    Verify that the generated velocities match the theoretical velocity dispersion.
    
    Parameters:
    - r: Array of radii in kpc
    - positions: Particle positions
    - velocities: Nx3 array of velocities
    - M: Total mass in solar masses
    - a: Scale radius in kpc
    - n_bins: Number of radial bins for verification
    - max_radius: Maximum radius to include in analysis (kpc)
    """
    print("Verifying velocity dispersion...")
    
    # Filter points within max_radius
    mask_radius = r <= max_radius
    r_filtered = r[mask_radius]
    positions_filtered = positions[mask_radius]
    velocities_filtered = velocities[mask_radius]
    
    # Calculate radial velocities (v_r = v·r̂)
    # Create unit vectors pointing in radial direction
    r_magnitude = np.sqrt(np.sum(positions_filtered**2, axis=1))
    r_hat = positions_filtered / r_magnitude[:, np.newaxis]  # Unit vectors in radial direction
    
    # Calculate radial velocity component (dot product)
    v_r = np.sum(velocities_filtered * r_hat, axis=1)
    
    # Create logarithmically spaced bins from min radius to max_radius
    r_min = np.min(r_filtered)
    r_bins = np.logspace(np.log10(r_min), np.log10(max_radius), n_bins+1)
    r_centers = np.sqrt(r_bins[:-1] * r_bins[1:])
    
    # Calculate velocity dispersion in each bin as sqrt(mean(v_r²))
    v_r_disp = np.zeros(n_bins)
    num_particles = np.zeros(n_bins, dtype=int)
    
    for i in range(n_bins):
        mask = (r_filtered >= r_bins[i]) & (r_filtered < r_bins[i+1])
        num_particles[i] = np.sum(mask)
        if num_particles[i] > 10:  # Only calculate if we have enough particles
            v_r_disp[i] = np.sqrt(np.mean(v_r[mask]**2))
    
    # Calculate theoretical velocity dispersion from Hernquist Eq. 10
    v_theory = hernquist_velocity_dispersion(r_centers, M, a)
    
    # Plot comparison only for bins with enough particles
    valid_bins = num_particles > 10
    
    plt.figure(figsize=(12, 8))
    plt.plot(r_centers[valid_bins], v_r_disp[valid_bins], 'o', label='Measured')
    plt.plot(r_centers, v_theory, '-', label='Theoretical')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Radial Velocity Dispersion (kpc/Gyr)')
    plt.legend()
    plt.title('Verification of Velocity Dispersion')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Add text showing number of particles in each bin
    for i in range(n_bins):
        if valid_bins[i]:
            plt.text(r_centers[i], v_r_disp[i]*1.1, str(num_particles[i]), 
                    fontsize=8, ha='center', va='bottom', alpha=0.7)
    
    plt.savefig('velocity_dispersion.png', dpi=150)
    plt.close()
    
    # Calculate normalized difference
    valid_comparison = valid_bins & (v_r_disp > 0) & (v_theory > 0)
    if np.any(valid_comparison):
        diff = np.abs(v_r_disp[valid_comparison] - v_theory[valid_comparison]) / v_theory[valid_comparison]
        mean_diff = np.mean(diff)
        print(f"Mean fractional difference between measured and theoretical dispersion: {mean_diff:.4f}")
    
    print(f"Verification plot saved as 'velocity_dispersion.png'")


def save_data(positions, velocities, M, a):
    """
    Save the generated positions and velocities to a file.
    
    Parameters:
    - positions: Nx3 array of positions
    - velocities: Nx3 array of velocities
    - M: Total mass in solar masses
    - a: Scale radius in kpc
    """
    # Calculate particle mass
    N = len(positions)
    particle_mass = M / N
    
    # Create output array
    output = np.column_stack((
        np.ones(N) * particle_mass,  # Mass column
        positions,                    # x, y, z
        velocities                    # vx, vy, vz
    ))
    
    # Save to file
    np.savetxt('hernquist_halo.txt', output, 
               header=f'Hernquist Halo: M={M:.1e} M_sun, a={a} kpc, N={N}\n'
                      f'Format: m x y z vx vy vz (units: M_sun, kpc, kpc/Gyr)')
    
    print("Data saved to 'hernquist_halo.txt'")
    
    # Also save as binary (more efficient)
    np.save('hernquist_halo.npy', output)
    print("Data also saved to 'hernquist_halo.npy'")


if __name__ == "__main__":
    main()