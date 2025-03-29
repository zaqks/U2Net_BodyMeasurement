<table style="width:100%">
  <tr>    
    <th>Input Image</th>
    <th>Result</th>
    <th>Height, Waist_Width</th>
  </tr>
  <tr>    
    <td><img src='data/test.jpg' width='100%'></td>    
    <td><img src='out/body_lines.png' width='100%'></td>
    <td>180cm, 32.72cm</td>    
  </tr>
</table>



# U2Net_BodyMeasurement
This project leverages the <a href='https://github.com/LeCongThuong/U2Net'>U2Net model</a> to segment the human body from images and accurately extract its dimensions. 
By calibrating the camera to determine the body's distance, the system ensures precise measurements. 
Combining advanced AI-powered segmentation with distance-based calibration, this tool provides a reliable solution for estimating human body dimensions from simple images.

# Calibration Process
The system requires initial calibration using a reference subject of known height standing in a fixed position relative to the camera. This reference measurement establishes a pixel-to-real-world ratio by analyzing the segmented body dimensions from the U2Net model output. 
The fixed position ensures consistent camera perspective and depth, allowing calculation of scaling factors that account for distance-related size variations. 
Once calibrated, any subsequent subject standing in the same position can have their height and body proportions accurately determined through proportional pixel analysis of the AI-segmented silhouette, maintaining measurement consistency across different individuals without requiring additional distance sensors.

Alternatively, if the scaling factor has already been calculated, it can be set during the initialization phase of the measurer, allowing to bypass the calibration process and proceed directly to the calculations.

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


# Results
<table style="width:100%">
    <tbody>
      <tr>
        <td>Input Image</td>
        <td><img src='data/test.jpg' width='100%'></td>
      </tr>
      <tr>
        <td>Subject Mask</td>
        <td><img src='out/body_mask.png' width='100%'></td>
      </tr>
      <tr>        
        <td>Result</td>
        <td><img src='out/body_lines.png' width='100%'></td>
      </tr>
      <tr>
        <td>Height</td>
        <td>180cm</td>
      </tr>
      <tr>
        <td>Waist Width</td>
        <td>32.72cm</td>
      </tr>
    </tbody>
  
</table>

# Calculation of Height and Width

The dimensions of the human body, specifically height and width, are calculated using the following process:
- Height Calculation

    - Bounding Box Generation: After segmenting the body using the U2Net model, a bounding box is drawn around the mask that represents the segmented body.

    - Height Measurement: The height of the bounding box (in pixels) is directly used as the body height in the image. This measurement is then scaled to real-world dimensions using a pre-calculated scaling factor obtained during calibration.

- Width Calculation

    - Level Selection: A level parameter is defined between 0.1 and 1 (representing a percentage of the total height). This level determines which horizontal slice of the mask will be analyzed for width.

    - Pixel Analysis: At the selected level, corresponding to `level×bb_height`, the row of pixels is extracted from the mask.

    - Width Measurement:

        The maximum and minimum x-coordinates of non-zero pixels in this row are identified.

        The width is calculated as `max_x − min_x`, representing the widest part of the body at that specific level.

    - Scaling: Similar to height, this pixel-based width is converted to real-world dimensions using the scaling factor.


# Citation
```
@InProceedings{Qin_2020_PR,
author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar and Jagersand, Martin},
title = {U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection},
booktitle = {Pattern Recognition},
year = {2020}
}
```


