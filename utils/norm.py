import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def normalize_radial_intensity_3d(image: np.ndarray, center: tuple = None, plot_fit: bool = False) -> np.ndarray:
    """
    Normalize a 3D image based on the radial intensity profile, ensuring constant mean intensity.

    Parameters:
        image (np.ndarray): Input 3D image (single-channel).
        center (tuple): (y, x) coordinates of the center of the circular region.
                        If None, defaults to the center of the image.
        plot_fit (bool): Whether to display the linear fit of radial intensity.

    Returns:
        np.ndarray: Radially normalized 3D image.
    """
    # Determine the center if not provided
    if center is None:
        center = (image.shape[1] // 2, image.shape[2] // 2)
    
    # Create a grid of radial distances
    y, x = np.indices((image.shape[1], image.shape[2]))
    radii = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

    # Calculate mean intensity per radius
    max_radius = int(np.ceil(radii.max()))
    mean_intensity_per_radius = [
        image[:, (radii >= r) & (radii < r + 1)].mean() if np.any((radii >= r) & (radii < r + 1)) else 0
        for r in range(max_radius)
    ]

    # Define radius values and intensity profile
    radius_values = np.arange(max_radius, dtype=float)
    intensities = np.array(mean_intensity_per_radius)

    # Remove invalid data (NaNs or zeros)
    valid = intensities > 0
    radius_values = radius_values[valid]
    intensities = intensities[valid]

    # Fit a linear decay model
    def linear_decay(r, a, b):
        return a * r + b

    popt, _ = curve_fit(linear_decay, radius_values, intensities, p0=[-1, intensities[0]])
    slope, intercept = popt

    if plot_fit:
        # Plot the observed and fitted intensity profile
        plt.figure(figsize=(8, 6))
        plt.plot(radius_values, intensities, 'b.', label="Observed Intensities")
        plt.plot(radius_values, linear_decay(radius_values, *popt), 'r-', 
                 label=f"Fit: y = {slope:.2f}x + {intercept:.2f}")
        plt.title("Intensity Decay Fit")
        plt.xlabel("Radius")
        plt.ylabel("Mean Intensity")
        plt.legend()
        plt.show()

    # Normalize the intensity for each radius to make the mean the same
    normalized_image = image.copy().astype(float)
    target_mean = intensities.mean()  # The target mean intensity for all radii
    for r in range(max_radius):
        mask = (radii >= r) & (radii < r + 1)
        if np.any(mask):
            observed_mean = image[:, mask].mean()
            scaling_factor = target_mean / observed_mean if observed_mean > 0 else 1
            normalized_image[:, mask] *= scaling_factor

    return normalized_image
