from PIL import Image, ImageDraw
from U2Net import BodySegmentationModel


class BodyMeasurer:
    def __init__(self, resize_factor=None):
        self.mdl = BodySegmentationModel()
        # CALIBRATION
        self.resize_factor = resize_factor  # 1px = ? CM

    def calibrate(self, calibration_img: str, body_height: int):
        # GET THE MASK
        self.mdl.predict(calibration_img, 'out/calibration_mask.png')
        # GET THE BOUNDING BOX
        img = Image.open('out/calibration_mask.png').convert("L")
        bbox = img.getbbox()  # Returns (left, upper, right, lower) coordinates of the bounding box

        # Convert the image back to RGB mode for drawing
        image_rgb = img.convert("RGB")
        draw = ImageDraw.Draw(image_rgb)

        if bbox:
            # width = bbox[2] - bbox[0]  # right - left
            height = bbox[3] - bbox[1]  # lower - upper

            self.resize_factor = round(body_height/height, 4)

            # print(f'{height}px = {calibration_height}cm')
            # print(f'1px = {round(calibration_height/height, 4)}cm')

            # Draw a red rectangle around the white shape
            # Width specifies thickness of the rectangle
            # draw.rectangle(bbox, outline="red", width=3)

            # Save the output image
            # image_rgb.save("data/bounding_box.png")
        else:
            print('No Bounding Box Detected')


if __name__ == '__main__':
    measurer = BodyMeasurer(resize_factor=0.0951)
    # measurer.calibrate('data/front.jpg', 180)
