# BodyMeasurementAI
This project leverages the <a href='https://github.com/LeCongThuong/U2Net'>U2Net model</a> to segment the human body from images and accurately extract its dimensions. 
By calibrating the camera to determine the body's distance, the system ensures precise measurements. 
Combining advanced AI-powered segmentation with distance-based calibration, this tool provides a reliable solution for estimating human body dimensions from simple images.

# Calibration Process
The system requires initial calibration using a reference subject of known height standing in a fixed position relative to the camera. This reference measurement establishes a pixel-to-real-world ratio by analyzing the segmented body dimensions from the U2Net model output. 
The fixed position ensures consistent camera perspective and depth, allowing calculation of scaling factors that account for distance-related size variations. 
Once calibrated, any subsequent subject standing in the same position can have their height and body proportions accurately determined through proportional pixel analysis of the AI-segmented silhouette, maintaining measurement consistency across different individuals without requiring additional distance sensors.

Alternatively, if the scaling factor has already been calculated, it can be set during the initialization phase of the measurer, allowing to bypass the calibration process and proceed directly to the calculations.

Citations:
[1] https://github.com/LeCongThuong/U2Net

---
Answer from Perplexity: pplx.ai/share

# Usage Example
```py
import U2Net

# INITALIZATION (the resize factor bypasses the calibration process, no need to set it if you're going to calibrate)
measurer = BodyMeasurer(resize_factor=0.0951, out_dir='out/')

# CALIBRATION
measurer.calibrate('data/test.jpg', body_height=180)

# SET THE BODY TO MEASURE
measurer.set_body('data/test.jpg')

# GET DIMENSIONS
print(f'height: {measurer.get_body_height()}cm')
print(f'waist: {measurer.get_body_width(level=0.45)}cm')

# ADDON: DRAW ELLIPSE
measurer.draw_markers(level=0.45)

```