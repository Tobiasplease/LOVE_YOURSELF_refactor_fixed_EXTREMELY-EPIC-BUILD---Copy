import numpy as np
import cv2

# from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# from svgpathtools import Path, Line, QuadraticBezier, CubicBezier, Arc, svg2paths, wsvg


def pre_distort_for_plotter(image, pivot_point=None, max_radius=None, interpolation="linear"):
    """
    Pre-distort an image to compensate for plotter fan distortion.
    When the plotter draws this pre-distorted image, the output should appear correct.

    Parameters:
    -----------
    image : numpy array
        Input image (grayscale or color) - your original design
    pivot_point : tuple (x, y)
        The pivot point of the plotter arm
    max_radius : float
        Maximum radius of the plotter arm
    interpolation : str
        Interpolation method ('linear', 'cubic', 'nearest')

    Returns:
    --------
    pre_distorted_image : numpy array
        The pre-distorted image to send to the plotter
    """

    height, width = image.shape[:2]

    # Estimate pivot point if not provided
    # !!! TWEAK
    # pivot_y (negative value): How far above the paper the arm pivots
    # pivot_x: Usually the center of your paper width
    if pivot_point is None:
        pivot_x = width // 2
        pivot_y = -height * 0.3  # Pivot above the image
        pivot_point = (pivot_x, pivot_y)

    # Create coordinate grids for the output (pre-distorted) image
    # Using meshgrid to ensure proper shape
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # For each pixel in the output, calculate where it should sample from the input
    # This is the inverse of the plotter's distortion

    # Calculate the radius and angle for each output pixel
    dx = x_grid - pivot_point[0]
    dy = y_grid - pivot_point[1]
    radius = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dx, dy)

    # Avoid division by zero
    max_r = np.max(radius)
    if max_r == 0:
        max_r = 1

    # The plotter creates fan distortion where equal angles create unequal spacing
    # So we need to pre-compress the angles to compensate

    # Compensate for the arc distortion
    # When the plotter moves in an arc, horizontal spacing increases with radius
    # So we need to pre-compress horizontally, more at larger radii

    # Calculate where each output pixel should sample from the input
    # The plotter stretches things horizontally, so we pre-compress
    compression_factor = radius / max_r

    # Source coordinates (where to sample from the original image)

    # !!! TWEAK
    src_angle = angle / (1 + compression_factor * 1.2)  # Try 0.3 for less, 0.7 for more
    src_x = pivot_point[0] + radius * np.sin(src_angle)
    src_y = y_grid.astype(np.float32)  # Keep y coordinates the same

    # Ensure coordinates are float32 and properly shaped
    map_x = src_x.astype(np.float32)
    map_y = src_y.astype(np.float32)

    # Apply remapping
    pre_distorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    return pre_distorted


def pre_distort_coordinates(points, image_shape, pivot_point=None):
    """
    Pre-distort a set of coordinates (useful for vector/SVG data).

    Parameters:
    -----------
    points : array of shape (N, 2)
        Array of (x, y) coordinates
    image_shape : tuple (height, width)
        The dimensions of the drawing area
    pivot_point : tuple (x, y)
        The pivot point of the plotter arm

    Returns:
    --------
    distorted_points : array of shape (N, 2)
        Pre-distorted coordinates
    """
    height, width = image_shape

    if pivot_point is None:
        pivot_point = (width // 2, -height * 0.3)

    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    # Convert to polar coordinates relative to pivot
    dx = x - pivot_point[0]
    dy = y - pivot_point[1]
    radius = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dx, dy)

    # Apply pre-distortion
    # The key is to expand angles to compensate for the plotter's compression
    max_radius = np.sqrt(width**2 + (height + abs(pivot_point[1])) ** 2)
    compression_factor = radius / max_radius

    # Pre-expand angles (opposite of what the plotter does)
    distorted_angle = angle * (1 + compression_factor * 0.5)

    # Convert back to Cartesian
    distorted_x = pivot_point[0] + radius * np.sin(distorted_angle)
    distorted_y = y  # Keep y relatively unchanged

    return np.column_stack([distorted_x, distorted_y])


def create_test_grid(width=800, height=600, grid_size=20):
    """
    Create a test grid pattern for calibration.
    """
    img = np.ones((height, width), dtype=np.uint8) * 255

    # Draw vertical lines
    for x in range(0, width, grid_size):
        cv2.line(img, (x, 0), (x, height), 0, 1)

    # Draw horizontal lines
    for y in range(0, height, grid_size):
        cv2.line(img, (0, y), (width, y), 0, 1)

    return img


def visualize_distortion(original, pre_distorted):
    """
    Visualize the original and pre-distorted images side by side.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(original, cmap="gray")
    ax1.set_title("Original Grid")
    ax1.axis("off")

    ax2.imshow(pre_distorted, cmap="gray")
    ax2.set_title("Pre-distorted Grid (send this to plotter)")
    ax2.axis("off")

    plt.tight_layout()
    plt.show()


def calibrate_plotter_distortion(test_outputs, expected_grid):
    """
    Calibrate the distortion parameters by comparing plotter output with expected output.
    This is an advanced function for fine-tuning.
    """
    # This would analyze the actual plotter output to determine optimal parameters
    # Implementation depends on your specific needs
    pass


# Example usage
if __name__ == "__main__":
    # Create a test grid
    test_grid = create_test_grid(800, 600, 30)

    # Pre-distort the grid
    # Adjust pivot_point based on your plotter's geometry
    pre_distorted = pre_distort_for_plotter(test_grid, pivot_point=(400, -200))

    # Save the pre-distorted image
    cv2.imwrite("pre_distorted_grid.png", pre_distorted)

    # Visualize the transformation
    visualize_distortion(test_grid, pre_distorted)

    # For vector/SVG workflow
    # Example: pre-distort a set of line endpoints
    line_points = np.array(
        [
            [100, 100],
            [700, 100],  # Horizontal line
            [100, 200],
            [700, 200],  # Another horizontal line
            [200, 50],
            [200, 550],  # Vertical line
        ]
    )

    distorted_points = pre_distort_coordinates(line_points, (600, 800), (400, -200))

    print("Original points:")
    print(line_points)
    print("\nPre-distorted points:")
    print(distorted_points)

    # Create an SVG with pre-distorted paths
    # This is what you'd send to your plotter
    from svgwrite import Drawing

    dwg = Drawing("pre_distorted_drawing.svg", size=("800px", "600px"))

    # Add pre-distorted lines
    for i in range(0, len(distorted_points), 2):
        start = distorted_points[i]
        end = distorted_points[i + 1]
        dwg.add(dwg.line(start=tuple(start), end=tuple(end), stroke="black", stroke_width=1))

    dwg.save()
    print("\nPre-distorted SVG saved!")
