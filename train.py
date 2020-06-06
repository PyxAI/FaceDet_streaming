import torch
import os
from joblib import dump
from models import *
from utils import getEmbedings
from input import get_train_samples, vid2img, verify_img_folder
import pickle
from configparser import ConfigParser

workers = 0 if os.name == 'nt' else 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# From file settings
settings = ConfigParser()
settings.read("settings.ini")
img_base_path = settings.get('Preperation','img_base_path')
class_model_name = settings.get('Training','class_model_name')

# Get cropped images
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

