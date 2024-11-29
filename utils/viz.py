import numpy as np
import cv2


def cartesian_to_polar_transform_3d(image: np.ndarray, center: tuple, output_shape: tuple = None) -> np.ndarray:
    """
    Transform a Cartesian 3D image stack to a polar image stack (rectangular image tracing each angle from the center).

    Parameters:
        image (np.ndarray): Input 3D image stack (D, H, W, C), where D is the depth, H and W are height and width, and C is the number of channels.
        center (tuple): Center of the circle (cx, cy).
        output_shape (tuple): Shape of the output image (radius, angle). If None, defaults to the image dimensions.

    Returns:
        np.ndarray: Polar-transformed image stack (D, Radius, Angle, C).
    """
    d, h, w, c = image.shape
    cx, cy = center

    # Determine output shape
    if output_shape is None:
        max_radius = int(np.sqrt((h // 2) ** 2 + (w // 2) ** 2))  # Diagonal distance
        output_shape = (max_radius, 360)  # Default: radius x 360 degrees

    radius, angle = output_shape

    # Create coordinate grid for polar space
    r = np.linspace(0, radius, radius)
    theta = np.linspace(0, 2 * np.pi, angle)
    r_grid, theta_grid = np.meshgrid(r, theta, indexing="ij")

    # Convert polar coordinates back to Cartesian
    x_cartesian = cx + (r_grid * np.cos(theta_grid)).astype(np.float32)
    y_cartesian = cy + (r_grid * np.sin(theta_grid)).astype(np.float32)

    # Initialize the output polar-transformed image stack
    polar_image_stack = np.zeros((d, radius, angle, c), dtype=image.dtype)

    # Apply the transformation for each slice
    for z in range(d):
        for channel in range(c):
            polar_image_stack[z, :, :, channel] = cv2.remap(
                image[z, :, :, channel],
                x_cartesian,
                y_cartesian,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

    return polar_image_stack