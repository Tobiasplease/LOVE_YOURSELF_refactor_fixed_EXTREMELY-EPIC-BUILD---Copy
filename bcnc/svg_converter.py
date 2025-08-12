import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import invert
import svgwrite


def raster_to_centerline_svg(
    input_path,
    output_path,
    threshold_value=180,
    blur_kernel=(3, 3),
    do_dilate=True,
    dilation_iterations=1,
    scale=1.0,
):

    print("[INFO] Läser in bild...")
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # === Förbehandling ===
    print("[INFO] Kör Gaussian blur...")
    img = cv2.GaussianBlur(img, blur_kernel, 0)

    print(f"[INFO] Trösklar med värde {threshold_value}...")
    _, binary = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

    binary = binary == 0  # Gör om till bool för skeletonize

    if do_dilate:
        print(f"[INFO] Dilar {dilation_iterations} gång(er)...")
        binary = binary.astype(np.uint8)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=dilation_iterations)
        binary = binary == 1  # tillbaka till bool

    print("[INFO] Skeletonizing...")
    skeleton = skeletonize(binary)

    # === Konvertera till SVG ===
    print("[INFO] Konverterar till SVG...")
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        skeleton_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    height, width = skeleton.shape
    dwg = svgwrite.Drawing(output_path, size=(f"{width*scale}px", f"{height*scale}px"))

    for cnt in contours:
        points = [(p[0][0] * scale, p[0][1] * scale) for p in cnt]
        if len(points) > 1:
            dwg.add(
                dwg.polyline(points=points, stroke="black", fill="none", stroke_width=1)
            )

    dwg.save()
    print(f"[KLART] Sparade centerline-SVG till {output_path}")


# === Exempelanrop ===
if __name__ == "__main__":
    raster_to_centerline_svg(
        input_path="input.png",
        output_path="output.svg",
        threshold_value=180,  # Testa 160–200 beroende på bild
        blur_kernel=(3, 3),  # (1,1) = ingen blur, (3,3) = mild
        do_dilate=True,  # Sätt till False om det tar med för mycket
        dilation_iterations=1,  # Testa 0–2
        scale=1.0,  # SVG-skalning
    )
