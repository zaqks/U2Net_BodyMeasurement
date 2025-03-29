from U2Net_BodyMeasurement import BodyMeasurer

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
