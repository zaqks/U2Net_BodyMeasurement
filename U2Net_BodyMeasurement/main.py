from PIL import Image, ImageDraw
from .U2Net import BodySegmentationModel
from os.path import join, basename, splitext
import numpy as np


class BodyMeasurer:
    def __init__(self, resize_factor=None, out_dir='.'):
        self.mdl = BodySegmentationModel()
        self.out_dir = out_dir
        # CALIBRATION
        self.resize_factor = resize_factor  # 1px = ? CM
        #
        self.img_path = None
        self.mask_path = None

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
        name, extension = splitext(basename(img_path))
        self.mask_path = join(self.out_dir, f"{name}_mask.png")
        self.img_path = img_path
        self.mdl.predict(self.img_path, self.mask_path)

        # GET THE BOUNDING BOX
        self.mask = Image.open(self.mask_path).convert('L')
        self.bbox = self. mask.getbbox()

    # NO RESIZE

    def _get_body_height(self):
        return (self.bbox[3] - self.bbox[1])

    def _get_body_width(self, level=0.5):
        crop = self.mask.crop(self.bbox)
        crop_array = np.array(crop)

        j = int(crop.height * level)
        row = crop_array[j, :]

        # Find indices of non-zero pixels
        non_zero_indices = np.where(row)[0]

        if len(non_zero_indices) > 0:
            return non_zero_indices[0], non_zero_indices[-1]

        return non_zero_indices

    # RESIZE
    def get_body_height(self):
        return self._get_body_height() * self.resize_factor

    def get_body_width(self, level=0.5):
        _, __ = self._get_body_width(level=level)
        return (__-_) * self.resize_factor

    # DRAW
    def draw_markers(self, level=0.5):
        # DRAW
        img = Image.open(self.img_path)
        draw = ImageDraw.Draw(img)

        # COORDS
        crop_width = self.bbox[2] - self.bbox[0]
        crop_height = self.bbox[3] - self.bbox[1]

        # HEIGHT
        x1, y1, x2, y2 = self.bbox
        x1 += crop_width/2
        x2 = x1

        draw.line([(x1, y1), (x2, y2)], fill='green', width=5)

        # WIDTH
        y3 = self.bbox[1] + crop_height*level
        y4 = y3

        x3, x4 = self._get_body_width(level=level)

        x3 += self.bbox[0]
        x4 += self.bbox[0]

        draw.line([(x3, y3), (x4, y4)], fill='red', width=5)

        draw.ellipse((x3, y1, x4, y2), outline="blue", width=5)

        # SAVE
        name, extension = splitext(basename(self.img_path))
        img.save(join(self.out_dir, f'{name}_lines.png'))
