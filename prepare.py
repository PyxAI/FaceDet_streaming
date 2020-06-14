import os
from configparser import ConfigParser
from argparse import ArgumentParser
import torch
import cv2
import time
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from input import ask_usr, create_cropped_images, get_train_samples, vid2img, verify_img_folder

# From file settings
settings = ConfigParser()
settings.read("settings.ini")
video_path = settings.get('Preperation','video_path')
video_label = settings.get('Preperation','video_label')
img_base_path = settings.get('Preperation','img_base_path')
max_frames = int(settings.get('Preperation','max_frames'))
data_dir = os.path.join(img_base_path, 'data')

# Take arugments and parse them
parser = ArgumentParser(description="Creating a training dataset")
parser.add_argument("-input", choices = ["webcam", "image_folder", "video"], help="Input path for training", required = True)
args = parser.parse_args()
in_type = args.input

# Create folder
if in_type == "video":
	try:
		new_label_folder = os.path.join(data_dir, video_label)
		os.mkdir(new_label_folder)
	except FileExistsError:
		pass
	vid2img(video_path, new_label_folder, device)

if in_type == "webcam":
	label = input("\n Please enter a name for the person in this webcam session:\n\
		This will be the folder name as well.\n\
		You can use (ctrl / command)+c to stop the session at any time\n \t")
	dst_dir = os.path.join(data_dir, label)
	try:
		os.mkdir(dst_dir)
	except FileExistsError as err:
		question = f"destination label {label} exists (folder exists), do you want to overwrite it? "
		if not ask_usr(question):
			exit()

	from PIL import Image
	cap = cv2.VideoCapture(0)
	time.sleep(0.2) # Give the camera time to adjust to environment
	frames = 0
	try:
		while frames < max_frames:
			frames += 1
			ret, frame = cap.read()
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			img = Image.fromarray(frame)
			img.save(os.path.join(dst_dir, "img{}.jpg".format(frames)))
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
			frame = cv2.putText(frame, f'frame {frames+1}/{max_frames}', (50,50),\
			cv2.FONT_HERSHEY_DUPLEX, 1.5, color=(255,0,0), lineType=2)	
			cv2.imshow('frame',frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break
		cv2.destroyAllWindows()
	except KeyboardInterrupt as stop:
		cv2.destroyAllWindows()


# Check for validity of image folder
verify_img_folder(data_dir)

# Creating cropped images folder
create_cropped_images(data_dir, device = device)

