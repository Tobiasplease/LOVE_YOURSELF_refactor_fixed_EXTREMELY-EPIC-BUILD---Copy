import numpy as np
import cv2
import matplotlib.pyplot as plt


class TwoJointPlotter:
    """
    Handles distortion correction for a two-joint plotter system.

    A two-joint plotter has two arms connected in series:
    - Base joint at fixed position (shoulder)
    - Second joint connecting two arm segments (elbow)
    - End effector (pen) position determined by both joint angles

    This creates more complex distortion than a single-joint fan pattern.
    """

    def __init__(self, shoulder_pos=(0, 0), upper_arm_length=200, lower_arm_length=150):
        """
        Initialize two-joint plotter geometry.

        Parameters:
        -----------
        shoulder_pos : tuple (x, y)
            Fixed position of the base/shoulder joint
        upper_arm_length : float
            Length of upper arm segment (shoulder to elbow)
        lower_arm_length : float
            Length of lower arm segment (elbow to pen)
        """
        self.shoulder_pos = np.array(shoulder_pos)
        self.L1 = upper_arm_length
        self.L2 = lower_arm_length
        self.max_reach = self.L1 + self.L2
        self.min_reach = abs(self.L1 - self.L2)

    def forward_kinematics(self, theta1, theta2):
        """
        Calculate end effector position from joint angles.

        Parameters:
        -----------
        theta1 : float or array
            Shoulder joint angle (radians)
        theta2 : float or array
            Elbow joint angle (radians)

        Returns:
        --------
        x, y : float or array
            End effector position(s)
        """
        # Elbow position
        elbow_x = self.shoulder_pos[0] + self.L1 * np.cos(theta1)
        elbow_y = self.shoulder_pos[1] + self.L1 * np.sin(theta1)

        # End effector position
        x = elbow_x + self.L2 * np.cos(theta1 + theta2)
        y = elbow_y + self.L2 * np.sin(theta1 + theta2)

        return x, y

    def inverse_kinematics(self, x, y):
        """
        Calculate joint angles from end effector position.

        Parameters:
        -----------
        x, y : float or array
            Target end effector position(s)

        Returns:
        --------
        theta1, theta2 : float or array
            Joint angles (radians), or NaN if unreachable
        """
        # Relative position from shoulder
        dx = x - self.shoulder_pos[0]
        dy = y - self.shoulder_pos[1]

        # Distance to target
        r = np.sqrt(dx**2 + dy**2)

        # Check reachability
        theta1 = np.full_like(r, np.nan)
        theta2 = np.full_like(r, np.nan)

        reachable = (r >= self.min_reach) & (r <= self.max_reach)

        if np.any(reachable):
            r_reach = r[reachable] if hasattr(r, "__len__") else r
            dx_reach = dx[reachable] if hasattr(dx, "__len__") else dx
            dy_reach = dy[reachable] if hasattr(dy, "__len__") else dy

            # Law of cosines for elbow angle
            cos_theta2 = (r_reach**2 - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
            cos_theta2 = np.clip(cos_theta2, -1, 1)
            theta2_reach = np.arccos(cos_theta2)

            # Shoulder angle
            alpha = np.arctan2(dy_reach, dx_reach)
            beta = np.arctan2(self.L2 * np.sin(theta2_reach), self.L1 + self.L2 * np.cos(theta2_reach))
            theta1_reach = alpha - beta

            if hasattr(theta1, "__len__"):
                theta1[reachable] = theta1_reach
                theta2[reachable] = theta2_reach
            else:
                theta1 = theta1_reach
                theta2 = theta2_reach

        return theta1, theta2

    def calculate_jacobian(self, theta1, theta2):
        """
        Calculate Jacobian matrix for velocity/distortion analysis.

        Returns:
        --------
        J : array (2, 2)
            Jacobian matrix [dx/dθ1, dx/dθ2; dy/dθ1, dy/dθ2]
        """
        # Partial derivatives of forward kinematics
        dx_dtheta1 = -self.L1 * np.sin(theta1) - self.L2 * np.sin(theta1 + theta2)
        dx_dtheta2 = -self.L2 * np.sin(theta1 + theta2)
        dy_dtheta1 = self.L1 * np.cos(theta1) + self.L2 * np.cos(theta1 + theta2)
        dy_dtheta2 = self.L2 * np.cos(theta1 + theta2)

        J = np.array([[dx_dtheta1, dx_dtheta2], [dy_dtheta1, dy_dtheta2]])

        return J


def pre_distort_for_two_joint_plotter(image, plotter, workspace_bounds=None):
    """
    Pre-distort an image to compensate for two-joint plotter distortion.

    Parameters:
    -----------
    image : numpy array
        Input image to pre-distort
    plotter : TwoJointPlotter
        Plotter configuration
    workspace_bounds : tuple (x_min, x_max, y_min, y_max)
        Physical workspace bounds, auto-calculated if None

    Returns:
    --------
    pre_distorted_image : numpy array
        Pre-distorted image for plotter
    """
    height, width = image.shape[:2]

    if workspace_bounds is None:
        # Default workspace: reachable area below shoulder
        margin = 50
        x_min = plotter.shoulder_pos[0] - plotter.max_reach + margin
        x_max = plotter.shoulder_pos[0] + plotter.max_reach - margin
        y_min = plotter.shoulder_pos[1] - plotter.max_reach + margin
        y_max = plotter.shoulder_pos[1] + plotter.max_reach - margin
        workspace_bounds = (x_min, x_max, y_min, y_max)

    x_min, x_max, y_min, y_max = workspace_bounds

    # Create output coordinate grid (what the plotter will draw)
    x_out = np.linspace(x_min, x_max, width)
    y_out = np.linspace(y_min, y_max, height)
    X_out, Y_out = np.meshgrid(x_out, y_out)

    # Calculate joint angles for each output position
    theta1, theta2 = plotter.inverse_kinematics(X_out, Y_out)

    # Calculate distortion based on joint space non-linearities
    # The key insight: equal changes in joint angles don't produce equal Cartesian movements

    # Calculate Jacobian determinant (measure of local area distortion)
    J_det = np.zeros_like(theta1)
    valid = ~np.isnan(theta1) & ~np.isnan(theta2)

    for i in range(height):
        for j in range(width):
            if valid[i, j]:
                J = plotter.calculate_jacobian(theta1[i, j], theta2[i, j])
                J_det[i, j] = np.abs(np.linalg.det(J))

    # Normalize Jacobian determinant
    max_det = np.nanmax(J_det)
    if max_det > 0:
        J_det_norm = J_det / max_det
    else:
        J_det_norm = np.ones_like(J_det)

    # Create mapping from output space to input space
    # Areas with higher distortion (lower Jacobian) need more compression in input
    distortion_factor = np.where(valid, 1.0 / np.maximum(J_det_norm, 0.1), 1.0)

    # Calculate source coordinates with distortion compensation
    center_x, center_y = width // 2, height // 2

    # Distance from center
    dx_norm = (np.arange(width) - center_x) / center_x
    dy_norm = (np.arange(height) - center_y) / center_y
    DX_norm, DY_norm = np.meshgrid(dx_norm, dy_norm)

    # Apply distortion compensation
    src_x = center_x + DX_norm * center_x / distortion_factor
    src_y = center_y + DY_norm * center_y / distortion_factor

    # Clip to valid range
    src_x = np.clip(src_x, 0, width - 1)
    src_y = np.clip(src_y, 0, height - 1)

    # Apply remapping
    map_x = src_x.astype(np.float32)
    map_y = src_y.astype(np.float32)

    pre_distorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    return pre_distorted


def create_distortion_map(plotter, image_shape, workspace_bounds=None):
    """
    Create a distortion map showing how the two-joint system affects drawing.

    Returns:
    --------
    distortion_map : numpy array
        Visual representation of distortion (brighter = more distorted)
    """
    height, width = image_shape

    if workspace_bounds is None:
        margin = 50
        x_min = plotter.shoulder_pos[0] - plotter.max_reach + margin
        x_max = plotter.shoulder_pos[0] + plotter.max_reach - margin
        y_min = plotter.shoulder_pos[1] - plotter.max_reach + margin
        y_max = plotter.shoulder_pos[1] + plotter.max_reach - margin
        workspace_bounds = (x_min, x_max, y_min, y_max)

    x_min, x_max, y_min, y_max = workspace_bounds

    # Create workspace grid
    x_coords = np.linspace(x_min, x_max, width)
    y_coords = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x_coords, y_coords)

    # Calculate joint angles
    theta1, theta2 = plotter.inverse_kinematics(X, Y)

    # Calculate Jacobian determinant at each point
    distortion_map = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            if not np.isnan(theta1[i, j]) and not np.isnan(theta2[i, j]):
                J = plotter.calculate_jacobian(theta1[i, j], theta2[i, j])
                det_J = np.abs(np.linalg.det(J))
                # Invert so high distortion = bright
                distortion_map[i, j] = 1.0 / (det_J + 0.1)
            else:
                distortion_map[i, j] = 0

    # Normalize to 0-255
    if np.max(distortion_map) > 0:
        distortion_map = (distortion_map / np.max(distortion_map) * 255).astype(np.uint8)

    return distortion_map


def create_test_pattern(width=800, height=600, pattern_type="grid"):
    """
    Create test patterns for calibration.
    """
    img = np.ones((height, width), dtype=np.uint8) * 255

    if pattern_type == "grid":
        grid_size = 40
        # Vertical lines
        for x in range(0, width, grid_size):
            cv2.line(img, (x, 0), (x, height), 0, 2)
        # Horizontal lines
        for y in range(0, height, grid_size):
            cv2.line(img, (0, y), (width, y), 0, 2)

    elif pattern_type == "circles":
        # Concentric circles centered on image
        center = (width // 2, height // 2)
        for r in range(50, min(width, height) // 2, 50):
            cv2.circle(img, center, r, 0, 2)

    elif pattern_type == "radial":
        # Radial lines from center
        center = (width // 2, height // 2)
        for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            x2 = int(center[0] + 300 * np.cos(angle))
            y2 = int(center[1] + 300 * np.sin(angle))
            cv2.line(img, center, (x2, y2), 0, 2)

    return img


def visualize_two_joint_correction(original, pre_distorted, plotter, workspace_bounds=None):
    """
    Visualize the correction process for two-joint plotter.
    """
    distortion_map = create_distortion_map(plotter, original.shape, workspace_bounds)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Original pattern
    ax1.imshow(original, cmap="gray")
    ax1.set_title("Original Pattern")
    ax1.axis("off")

    # Pre-distorted pattern
    ax2.imshow(pre_distorted, cmap="gray")
    ax2.set_title("Pre-distorted (send to plotter)")
    ax2.axis("off")

    # Distortion map
    im3 = ax3.imshow(distortion_map, cmap="hot")
    ax3.set_title("Distortion Map (bright = high distortion)")
    plt.colorbar(im3, ax=ax3)

    # Plotter workspace
    if workspace_bounds:
        x_min, x_max, y_min, y_max = workspace_bounds
        # Show reachable workspace
        theta_range = np.linspace(0, 2 * np.pi, 100)

        # Max reach circle
        max_x = plotter.shoulder_pos[0] + plotter.max_reach * np.cos(theta_range)
        max_y = plotter.shoulder_pos[1] + plotter.max_reach * np.sin(theta_range)

        # Min reach circle
        min_x = plotter.shoulder_pos[0] + plotter.min_reach * np.cos(theta_range)
        min_y = plotter.shoulder_pos[1] + plotter.min_reach * np.sin(theta_range)

        ax4.plot(max_x, max_y, "r-", label="Max reach")
        ax4.plot(min_x, min_y, "b-", label="Min reach")
        ax4.plot(plotter.shoulder_pos[0], plotter.shoulder_pos[1], "ko", markersize=10, label="Shoulder")

        # Show workspace bounds
        ax4.axhline(y_min, color="g", linestyle="--", alpha=0.7)
        ax4.axhline(y_max, color="g", linestyle="--", alpha=0.7)
        ax4.axvline(x_min, color="g", linestyle="--", alpha=0.7)
        ax4.axvline(x_max, color="g", linestyle="--", alpha=0.7)

        ax4.set_xlim(x_min - 100, x_max + 100)
        ax4.set_ylim(y_min - 100, y_max + 100)
        ax4.set_aspect("equal")
        ax4.legend()
        ax4.set_title("Plotter Workspace")
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Create two-joint plotter configuration
    # Shoulder at origin, arm lengths typical for desktop plotter
    plotter = TwoJointPlotter(shoulder_pos=(400, 100), upper_arm_length=250, lower_arm_length=200)  # Above the drawing area

    # Define workspace (drawing area)
    workspace_bounds = (200, 600, 200, 500)  # x_min, x_max, y_min, y_max

    # Create test patterns
    grid_pattern = create_test_pattern(800, 600, "grid")
    circle_pattern = create_test_pattern(800, 600, "circles")
    radial_pattern = create_test_pattern(800, 600, "radial")

    # Pre-distort for two-joint plotter
    pre_distorted_grid = pre_distort_for_two_joint_plotter(grid_pattern, plotter, workspace_bounds)

    pre_distorted_circles = pre_distort_for_two_joint_plotter(circle_pattern, plotter, workspace_bounds)

    # Save results
    cv2.imwrite("two_joint_pre_distorted_grid.png", pre_distorted_grid)
    cv2.imwrite("two_joint_pre_distorted_circles.png", pre_distorted_circles)

    # Visualize the process
    visualize_two_joint_correction(grid_pattern, pre_distorted_grid, plotter, workspace_bounds)

    print("Two-joint plotter correction complete!")
    print(f"Plotter reach: {plotter.min_reach:.1f} to {plotter.max_reach:.1f}")
    print(f"Workspace: {workspace_bounds}")

    # Test inverse kinematics
    test_points = np.array([[300, 300], [500, 400], [400, 350]])
    theta1, theta2 = plotter.inverse_kinematics(test_points[:, 0], test_points[:, 1])

    print("\nInverse kinematics test:")
    for i, (x, y) in enumerate(test_points):
        if not np.isnan(theta1[i]):
            # Verify with forward kinematics
            x_check, y_check = plotter.forward_kinematics(theta1[i], theta2[i])
            print(f"Point ({x}, {y}) -> angles ({theta1[i]:.3f}, {theta2[i]:.3f}) " f"-> check ({x_check:.1f}, {y_check:.1f})")
        else:
            print(f"Point ({x}, {y}) -> unreachable")
