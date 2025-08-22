import cv2
import numpy as np
from skimage.morphology import skeletonize

import torch
import torchvision.transforms.functional as F

# from skimage.util import invert
import svgwrite


def raster_to_centerline_svg(
    input_path,
    output_path,
    threshold_value=180,
    blur_kernel=(1, 1),  # (1,1) = ingen blur, (3,3) = mild
    do_dilate=False,
    dilation_iterations=1,
    scale=1.0,
    contrast_alpha=2.0,  # Contrast control (1.0 = no change, >1.0 = more contrast, above 2.0...bad?
    save_steps=False,  # When True, saves intermediate images with "_step1", "_step2", etc.
):

    print("[INFO] Läser in bild...")
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image from {input_path}")

    if save_steps:
        base_path = output_path.rsplit(".", 1)[0]
        cv2.imwrite(f"{base_path}_step0_original.png", img)

    # === Förbehandling ===
    # Step 1: Adjust contrast
    if contrast_alpha != 1.0:
        print(f"[INFO] Justerar kontrast med PyTorch (factor={contrast_alpha})...")
        # Convert to tensor, normalize to [0,1], apply contrast, convert back
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension
        img_tensor = F.adjust_contrast(img_tensor, contrast_alpha)
        img = (torch.clamp(img_tensor.squeeze(0), 0, 1) * 255).byte().numpy()

        if save_steps:
            cv2.imwrite(f"{base_path}_step1_contrast.png", img)

    # Step 2: Gaussian blur
    print("[INFO] Kör Gaussian blur...")
    img = cv2.GaussianBlur(img, blur_kernel, 0)  # type: ignore
    if save_steps:
        cv2.imwrite(f"{base_path}_step2_blur.png", img)

    print(f"[INFO] Trösklar med värde {threshold_value}...")
    _, binary = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    if save_steps:
        cv2.imwrite(f"{base_path}_step3_threshold.png", binary)

    binary = binary == 0  # Gör om till bool för skeletonize

    if do_dilate:
        #  Dilation expands/thickens the black regions in the binary image using a 2x2 kernel. It fills small
        #  gaps and makes thin lines thicker before skeletonization. This can help connect broken line segments
        #  but may also merge separate features that are close together.

        print(f"[INFO] Dilar {dilation_iterations} gång(er)...")
        binary = binary.astype(np.uint8)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=dilation_iterations)
        if save_steps:
            cv2.imwrite(f"{base_path}_step4_dilated.png", binary * 255)
        binary = binary == 1  # tillbaka till bool

    print("[INFO] Skeletonizing...")
    skeleton = skeletonize(binary)
    if save_steps:
        cv2.imwrite(f"{base_path}_step5_skeleton.png", (skeleton * 255).astype(np.uint8))

    # === Konvertera till SVG ===
    print("[INFO] Konverterar till SVG...")
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    contours, _ = cv2.findContours(skeleton_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    height, width = skeleton.shape
    dwg = svgwrite.Drawing(output_path, size=(f"{width*scale}px", f"{height*scale}px"))

    for cnt in contours:
        points = [(p[0][0] * scale, p[0][1] * scale) for p in cnt]  # type: ignore
        if len(points) > 1:
            dwg.add(dwg.polyline(points=points, stroke="black", fill="none", stroke_width=1))

    dwg.save()
    print(f"[KLART] Sparade centerline-SVG till {output_path}")


if __name__ == "__main__":
    import sys

    input_path = sys.argv[1] if len(sys.argv) > 1 else "input.png"
    output_path = input_path.rsplit(".", 1)[0] + ".svg"

    raster_to_centerline_svg(
        input_path=input_path,
        output_path=output_path,
        threshold_value=180,  # Testa 160–200 beroende på bild
        blur_kernel=(1, 1),  # (1,1) = ingen blur, (3,3) = mild
        do_dilate=True,  # Sätt till False om det tar med för mycket
        dilation_iterations=1,  # Testa 0–2
        scale=1.0,  # SVG-skalning
        save_steps=True,  # Spara mellanliggande steg
    )
