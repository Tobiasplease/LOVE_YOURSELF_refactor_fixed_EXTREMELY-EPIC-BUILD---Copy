import numpy as np
import cv2
from scipy.interpolate import griddata

# import matplotlib.pyplot as plt


def correct_fan_distortion(image, pivot_point=None, max_radius=None, interpolation="linear"):
    """
    Correct fan-shaped distortion from a plotter with a pivoting arm.

    Parameters:
    -----------
    image : numpy array
        Input image (grayscale or color)
    pivot_point : tuple (x, y)
        The pivot point of the plotter arm. If None, will estimate from image
    max_radius : float
        Maximum radius of the plotter arm. If None, will estimate from image
    interpolation : str
        Interpolation method ('linear', 'cubic', 'nearest')

    Returns:
    --------
    corrected_image : numpy array
        The corrected image
    """

    height, width = image.shape[:2]

    # Estimate pivot point if not provided (typically off the top of the image)
    if pivot_point is None:
        # Assuming pivot is centered horizontally and above the image
        pivot_x = width // 2
        pivot_y = -height * 0.5  # Negative because it's above the image
        pivot_point = (pivot_x, pivot_y)

    # Create coordinate grids
    y, x = np.ogrid[:height, :width]

    # Calculate polar coordinates relative to pivot
    dx = x - pivot_point[0]
    dy = y - pivot_point[1]
    radius = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dx, dy)  # Note: arctan2(x, y) not (y, x) for fan shape

    # Estimate max radius if not provided
    if max_radius is None:
        max_radius = np.max(radius)

    # Normalize radius
    # normalized_radius = radius / max_radius

    # Create the inverse transformation
    # In the corrected image, we want uniform spacing
    # So we map from distorted polar to regular cartesian

    # For the corrected image, create a regular grid
    corrected_height = int(np.max(radius) - np.min(radius))
    corrected_width = width

    # Create mapping from distorted to corrected coordinates
    # The key insight: in the distorted image, equal angles don't produce equal x-spacing
    # We need to compensate for this

    # Calculate the corrected x-coordinate based on the angle
    # In a fan distortion, x varies with both angle and radius
    corrected_x = pivot_point[0] + radius * np.sin(angle)
    corrected_y = radius - np.min(radius)

    # Create output coordinate grid
    out_y, out_x = np.ogrid[:corrected_height, :corrected_width]

    # Flatten arrays for griddata
    points = np.column_stack((corrected_x.ravel(), corrected_y.ravel()))

    if len(image.shape) == 2:
        # Grayscale
        values = image.ravel()
        corrected = griddata(points, values, (out_x, out_y), method=interpolation)
    else:
        # Color image
        corrected = np.zeros((corrected_height, corrected_width, image.shape[2]))
        for i in range(image.shape[2]):
            values = image[:, :, i].ravel()
            corrected[:, :, i] = griddata(points, values, (out_x, out_y), method=interpolation)

    # Fill NaN values with 255 (white)
    corrected = np.nan_to_num(corrected, nan=255)

    return corrected.astype(np.uint8)


def create_correction_mesh(width, height, pivot_point, grid_size=20):
    """
    Create a mesh transformation for real-time correction.
    This can be applied using cv2.remap() for faster processing.
    """
    # Create destination grid
    dst_x, dst_y = np.meshgrid(np.arange(width), np.arange(height))

    # Calculate source coordinates for each destination pixel
    # This is the inverse of the distortion
    dx = dst_x - pivot_point[0]
    dy = dst_y - pivot_point[1] + height * 0.5  # Adjust for pivot above image

    # Convert to polar
    radius = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dx, dy)

    # Apply inverse fan transformation
    # In the original distorted image, points at the same radius
    # but different angles are spaced according to the arc length
    arc_factor = angle / (np.pi / 3)  # Assuming ~60 degree fan spread

    src_x = pivot_point[0] + radius * np.sin(angle * arc_factor)
    src_y = radius * np.cos(angle * arc_factor) - height * 0.5

    return src_x.astype(np.float32), src_y.astype(np.float32)


# Example usage
if __name__ == "__main__":
    # Load your image
    # img = cv2.imread('plotter_output.jpg', cv2.IMREAD_GRAYSCALE)

    # For demonstration, create a synthetic distorted grid
    img = np.ones((600, 800), dtype=np.uint8) * 255

    # Correct the distortion
    # You may need to adjust the pivot_point based on your plotter
    corrected = correct_fan_distortion(img, pivot_point=(400, -300))

    # For faster processing of multiple images, use remap
    map_x, map_y = create_correction_mesh(800, 600, (400, -300))
    corrected_fast = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    # Save or display results
    # cv2.imwrite('corrected_output.jpg', corrected)

    print("Correction complete!")
