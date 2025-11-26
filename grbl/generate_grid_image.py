import numpy as np
from PIL import Image, ImageDraw

# Grid parameters matching setup_grbl_grid.py
GRID_SIZE_MM = 40  # 40mm x 40mm
GRID_SPACING_MM = 10  # 10mm spacing
PIXELS_PER_MM = 10  # Scale factor for image resolution

# Calculate image dimensions
img_size = GRID_SIZE_MM * PIXELS_PER_MM
grid_spacing_px = GRID_SPACING_MM * PIXELS_PER_MM

# Create white background
img = Image.new('RGB', (img_size, img_size), 'white')
draw = ImageDraw.Draw(img)

# Draw vertical lines
for x_mm in range(0, GRID_SIZE_MM + 1, GRID_SPACING_MM):
    x_px = x_mm * PIXELS_PER_MM
    draw.line([(x_px, 0), (x_px, img_size)], fill='black', width=2)

# Draw horizontal lines  
for y_mm in range(0, GRID_SIZE_MM + 1, GRID_SPACING_MM):
    y_px = y_mm * PIXELS_PER_MM
    draw.line([(0, y_px), (img_size, y_px)], fill='black', width=2)

# Save the image
output_path = 'grbl/grid_40x40mm.png'
img.save(output_path)
print(f"Grid image saved to {output_path}")
print(f"Image size: {img_size}x{img_size} pixels")
print(f"Grid: {GRID_SIZE_MM}x{GRID_SIZE_MM}mm with {GRID_SPACING_MM}mm spacing")