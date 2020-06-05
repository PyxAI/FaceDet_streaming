import os
from configparser import ConfigParser
from argparse import ArgumentParser
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from input import create_cropped_images, get_train_samples, vid2img, verify_img_folder

# From file settings
settings = ConfigParser()
settings.read("settings.ini")
img_base_path = settings.get('Preperation','img_base_path')
video_path = settings.get('Preperation','video_path')
video_label = settings.get('Preperation','video_label')

# Take arugments and parse them
parser = ArgumentParser(description="Creating a training dataset")
parser.add_argument("-input", choices = ["webcam", "image_folder", "video"], help="Input path for training", required = True)
args = parser.parse_args()
in_type = args.input

# Create folder
if in_type == "video":
	try:
		new_label_folder = os.path.join(img_base_path,'data',video_label)
		os.mkdir(new_label_folder)
	except FileExistsError:
		pass
	vid2img(video_path, new_label_folder, device)

# Check for validity of image folder
verify_img_folder(img_base_path)

# Creating cropped images folder
create_cropped_images(img_base_path, device = device)

