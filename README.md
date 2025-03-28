# BodyMeasurementAI
This project leverages the <a href='https://github.com/LeCongThuong/U2Net'>U2Net model</a> to segment the human body from images and accurately extract its dimensions. 
By calibrating the camera to determine the body's distance, the system ensures precise measurements. 
Combining advanced AI-powered segmentation with distance-based calibration, this tool provides a reliable solution for estimating human body dimensions from simple images.

# Usage Example
```py
import U2Net

# LOAD THE MODEL
model = U2Net.BodySegmentationModel()

# SEGMENT
if __name__ == '__main__':
  model.predict('image_path', 'output_directory')

```
