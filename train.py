import torch
import os
from joblib import dump
from models import *
from utils import getEmbedings
from input import create_cropped_images, get_train_samples, vid2img, verify_img_folder
import pickle
from configparser import ConfigParser
from argparse import ArgumentParser

workers = 0 if os.name == 'nt' else 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# From file settings
settings = ConfigParser()
settings.read("settings.ini")
img_base_path = settings.get('Training','img_base_path')
video_path = settings.get('Training','video_path')
video_label = settings.get('Training','video_label')
class_model_name = settings.get('Training','class_model_name')


# Take arugments and parse them
parser = ArgumentParser(description="Training face regconition module")
parser.add_argument("--input", choices = ["webcam", "image_folder", "video"], help="Input path for training") # not implemented
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

# Get cropped faces
trainX, trainY = get_train_samples(img_base_path, device=device)

resnet = get_resnet(device, classify = True)
# Create a list of embeddings and labels for trainning images
trainX, trainY = getEmbedings(trainX, trainY, resnet)

norm = get_norm('l2')
trainX = norm.transform(trainX.squeeze())

# Prepare labels
convert_labels = get_encoder()
convert_labels.fit(trainY)
trainY = convert_labels.transform(trainY)

clf = get_model(class_model_name)

# Fit classifier!
print ("Fitting model:{}".format(clf))

clf.fit(trainX.squeeze(),trainY)
print("Finished!")
dump(clf, 'trained_models/' + class_model_name + '.joblib')
pickle.dump(convert_labels,open('trained_models/label_enc.pkl','wb'))

