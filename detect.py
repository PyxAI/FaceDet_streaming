import cv2
import time
import numpy as np
import utils
from utils import get_face_crop, clear_buffer
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from models import *
import argparse
from joblib import load
import pickle
from configparser import ConfigParser
from argparse import ArgumentParser


# From settings file
settings = ConfigParser()
settings.read("settings.ini")
class_model_name = settings.get('Training','class_model_name').lower()
try:
	frame_rate =int(settings.get('Detection','frame_rate'))
except ValueError:
	print ("Frame_rate in settings file cannot be converted to int")


# From command line args
parser = ArgumentParser(description = "Detect faces trained from dataset")
parser.add_argument("-input", choices = ["webcam", "ipcam"], \
	help = "Choose either connected webcam or external ipcam", required = True)
parser.add_argument("-user", help = "enter username for ipcam ", required = False)
parser.add_argument("-password", help = "enter username for ipcam", required = False)
args = parser.parse_args()
# Set input source
if args.input =="ipcam":
	in_src = settings.get('Detection','ipcam')
	if args.user and args.password:
		import re
		user = args.user
		password = args.password
		user_pos = re.search('//',in_src).start() + 2
		in_src = in_src[:user_pos]+ user+':'+password +'@'+in_src[user_pos:] #construct user pass
		print (in_src)
elif args.input == "webcam":
	in_src = 0
elif args.input == "video":
	print ("Not implemented yet")


font  = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255,0,0)
lineType = 2
# TODO

# imrpove speed by using fast mtcnn

#Get from webcam or ipcam
clf = load('trained_models/' + class_model_name + '.joblib') 
mtcnn = face_extractor(device, scale_factor = 0.5, margin = 10)
resnet = get_resnet(device, classify = True)
norm = get_norm('l2')
convert_labels = pickle.load(open('trained_models/label_enc.pkl','rb'))

cap = cv2.VideoCapture(in_src)

while(True):
	# Get frame
	ret, frame = cap.read()
	try:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	except cv2.error as err:
		print ("Check input stream, user-name and password")
		exit(2)
	# Run through mtcnn to get a cropped face
	bbox_list, probs = mtcnn.detect(frame)
	if bbox_list is not None:
		for bbox in bbox_list:
			bbox = [int(x) for x in bbox]
			x1, y1, x2, y2 = (bbox[i] for i in range(4))
			crop = frame[x1:x2,y1:y2,:]
			crop = np.moveaxis(crop, -1, 0)
			crop = torch.Tensor(crop)

			# Get draw bboxes on and draw them
			
			# To verify we got a bbox, (comes as an array, or None) we run throught two "if"s
			if isinstance(probs,np.ndarray):
				if probs.any():
					start_point = (x1, y1)
					end_point = (x2, y2)
					print (start_point, end_point)
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

