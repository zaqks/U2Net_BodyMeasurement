import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob
import argparse

from .data_loader import *

from .model import *  # full size version 173.6 MB / small version u2net 4.7 MB
import time


class BodySegmentationModel:
    def __init__(self, model_name='u2netp'):
        self.model_name = model_name
        self.model = None
        #
        self.load_model()

    # normalize the predicted SOD probability map

    def my_collate(self, batch):
        batch = list(filter(lambda img: img is not None, batch))
        return torch.utils.data.dataloader.default_collate(list(batch))

    def normPRED(self, d):
        ma = torch.max(d)
        mi = torch.min(d)

        dn = (d - mi) / (ma - mi)

        return dn

    # ADAPTED SAVING FUNCTION

    def save_output(self, image_name: str, pred, output_name: str):
        try:
            # Convert prediction tensor to PIL Image
            predict = pred.squeeze().cpu().data.numpy()
            im = Image.fromarray(predict * 255).convert('RGB')

            # Get original image dimensions
            image = io.imread(image_name)
            imo = im.resize((image.shape[1], image.shape[0]), Image.BILINEAR)

            # Apply thresholding to create a binary mask
            pb_np = np.array(imo)
            thresholded_np = np.where(pb_np > 200, 255, 0)  # Threshold at 200

            # Convert thresholded array back to PIL Image
            thresholded_im = Image.fromarray(thresholded_np.astype(np.uint8))

            # Handle filenames with multiple dots (e.g., "image.1.jpg")
            img_name = os.path.basename(image_name)
            parts = img_name.split(".")
            imidx = ".".join(parts[:-1])  # Remove file extension

            # Save final thresholded image as PNG
            thresholded_im.save(output_name)

        except Exception as error:
            raise Exception(f"Error saving image: {error}")

    def get_parameters(self):
        parser = argparse.ArgumentParser(
            description="Identifying Salient Object Detection")
        parser.add_argument("-i",
                            "--input",
                            help="Path to the file that lists all path to images",
                            type=str)
        parser.add_argument("-o",
                            "--output_dir",
                            help="Path to the output dir", type=str)

        parser.add_argument("-e",
                            "--errorFile",
                            help="Path to the log error file", type=str)
        args = parser.parse_args()

        return args

    def load_model(self):

        model_dir = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), 'saved_models/' + self.model_name + '.pth')

        # --------- 3. model define ---------
        if self.model_name == 'u2net':
            print("...load U2NET---173.6 MB")
            self.model = U2NET(3, 1)
        elif self.model_name == 'u2netp':
            print("...load U2NEP---4.7 MB")
            self.model = U2NETP(3, 1)
        self.model.load_state_dict(torch.load(
            # CPU MODE
            model_dir, map_location=None if torch.cuda.is_available() else torch.device('cpu')))
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def predict(self, src: str, out: str):
        img_name_list = [src]

        # --------- 2. dataloader ---------
        # 1. dataloader
        test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                            lbl_name_list=[],
                                            transform=transforms.Compose([RescaleT(320),
                                                                          ToTensorLab(flag=0)])
                                            )

        test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1,
                                            collate_fn=self.my_collate)

        # --------- 4. inference for each image ---------

        for i_test, data_test in enumerate(test_salobj_dataloader):
            try:
                print("\r------In processing file {} with name {}--------".format(i_test +
                                                                                  1, img_name_list[i_test].split("/")[-1]), end='\n')

                inputs_test = data_test['image']
                inputs_test = inputs_test.type(torch.FloatTensor)

                if torch.cuda.is_available():
                    inputs_test = Variable(inputs_test.cuda())
                else:
                    inputs_test = Variable(inputs_test)

                d1, d2, d3, d4, d5, d6, d7 = self.model(inputs_test)

                # normalization
                pred = d1[:, 0, :, :]
                pred = self.normPRED(pred)

                # save results to test_results folder
                self.save_output(img_name_list[i_test], pred, out)

                del d1, d2, d3, d4, d5, d6, d7
            except Exception as error:
                print(error)
                # with open(error_file_link, 'a+') as err_file:
                #     error_mess = img_name_list[i_test] + \
                #         '*' + str(error) + '\n'
                #     err_file.write(error_mess)
                # continue
