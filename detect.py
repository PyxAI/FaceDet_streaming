import cv2
import time
import numpy as np
from utils import get_face_crop, clear_buffer
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from models import *
import argparse
from joblib import load
import pickle
from configparser import ConfigParser
from argparse import ArgumentParser

parser = ArgumentParser(description = "Detect faces trained from dataset")
parser.add_argument("-input", choices = ["webcam", "ipcam"], help = "Choose either connected webcam or external ipcam", required = True)
args = parser.parse_args()

settings = ConfigParser()
settings.read("settings.ini")
class_model_name = settings.get('Training','class_model_name').lower()
frame_rate = settings.get('Detection','frame_rate').lower()
font  = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255,0,0)
lineType = 2

# TODO

# improve speed by only using the mtcnn once for bboxes and crops

# imrpove speed by using fast mtcnn


#Get from webcam or ipcam
clf = load('trained_models/' + class_model_name + '.joblib') 
mtcnn = face_extractor(device)
resnet = get_resnet(device, classify = True)
norm = get_norm('l2')
convert_labels = pickle.load(open('trained_models/label_enc.pkl','rb'))

if args.input == "webcam":
	in_src = 0
else:
	in_src = settings.get('Detection','ipcam')
cap = cv2.VideoCapture(in_src)
while(True):
	# Get frame
	ret, frame = cap.read()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	
	# Run through mtcnn to get a cropped face
	crop = get_face_crop(frame, device)
	if crop is not None:
		
		# Get draw bboxes on and draw them
		bbox, probs = mtcnn.detect(frame)

		# To verify we got a bbox, (comes as an array, or None) we run throught two "if"s
		if isinstance(probs,np.ndarray):
			if probs.any():
				start_point = (bbox[0][0],bbox[0][1])
				end_point = (bbox[0][2],bbox[0][3])
				frame = cv2.rectangle(frame, start_point, end_point, (255,0,0), 2) 

		# Get faceID only if we have a crop
		embeddings = resnet(crop.unsqueeze(0)).detach().numpy()
		embeddings = (embeddings- embeddings.mean()) / embeddings.std()
		# Predict!
		prob = clf.predict_proba(embeddings)
		pred = np.argmax(prob)
		face = convert_labels.inverse_transform([pred])[0] + "\nprobability: {:.4f}".format(prob[0][pred])
		print("pred = {}, prob = {}".format(pred,prob))
		# Draw text
		bottomLeftCornerOfText = (int(start_point[0]),int(start_point[1])-10)
		frame = cv2.putText(frame, face+': ', bottomLeftCornerOfText, font, fontScale, color=fontColor, lineType=lineType)
		
		# For debugging
		#display(Image.open(base+'who_crop.jpg'))
		print(face)

	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break
	if args.input == "ipcam":
		clear_buffer(cap, frame_rate=frame_rate)

