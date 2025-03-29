from PIL import Image, ImageDraw
from U2Net import BodySegmentationModel
from os.path import join
import numpy as np


class BodyMeasurer:
    def __init__(self, resize_factor=None, out_dir='.'):
        self.mdl = BodySegmentationModel()
        self.out_dir = out_dir
        # CALIBRATION
        self.resize_factor = resize_factor  # 1px = ? CM
        #
        self.img_path = None
        self.mask = None
        self.bbox = None

    def calibrate(self, calibration_img: str, body_height: int):
        # GET THE MASK
        self.mdl.predict(calibration_img, join(
            self.out_dir, 'calibration_mask.png'))
        # GET THE BOUNDING BOX
        img = Image.open(
            join(self.out_dir, 'calibration_mask.png')).convert("L")
        bbox = img.getbbox()  # Returns (left, upper, right, lower) coordinates of the bounding box

        # Convert the image back to RGB mode for drawing
        image_rgb = img.convert("RGB")
        draw = ImageDraw.Draw(image_rgb)

        if bbox:
            # width = bbox[2] - bbox[0]  # right - left
            height = bbox[3] - bbox[1]  # lower - upper

            self.resize_factor = body_height/height

            # print(f'{height}px = {calibration_height}cm')
            # print(f'1px = {round(calibration_height/height, 4)}cm')

            # Draw a red rectangle around the white shape
            # Width specifies thickness of the rectangle
            # draw.rectangle(bbox, outline="red", width=3)

            # Save the output image
            # image_rgb.save("data/bounding_box.png")
        else:
            print('No Bounding Box Detected')

    def set_body(self, img_path: str):
        if not self.resize_factor:
            print('resize_factor not set for auto-calibration')
            return None

        # SEGMENTATION
        mask_path = join(self.out_dir, 'body_mask.png')
        self.img_path = img_path
        self.mdl.predict(self.img_path, mask_path)

        # GET THE BOUNDING BOX
        self.mask = Image.open(mask_path).convert('L')
        self.bbox = self. mask.getbbox()

    def get_body_height(self):
        return (self.bbox[3] - self.bbox[1])*self.resize_factor

    def get_body_width(self,  level=0.5):
        crop = self.mask.crop(self.bbox)

        j = int(crop.height*level)
        row = [_ for _ in range(crop.width) if crop.getpixel(
            (_, j))]

        return (row[-1] - row[0])*self.resize_factor

    def draw_markers(self, level=0.5):
        # COORDS
        width = self.bbox[2] - self.bbox[0]

        x1, y1, x2, y2 = self.bbox
        x1, x2 = x1 + width/2, x2-width/2
        #
        img = Image.open(self.img_path)
        draw = ImageDraw.Draw(img)

        draw.line([(x1, y1), (x2, y2)], fill='green', width=5)
        img.save(join(self.out_dir, 'body_lines.png'))


if __name__ == '__main__':
    measurer = BodyMeasurer(resize_factor=0.0951, out_dir='out/')
    # measurer.calibrate('data/front.jpg', body_height=180)
    measurer.set_body('data/front.jpg')

    print(f'height: {measurer.get_body_height()}cm')
    print(f'waist: {measurer.get_body_width()}cm')

    measurer.draw_markers()
