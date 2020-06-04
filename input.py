from facenet_pytorch import training
from models import face_extractor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from utils import collate_fn, get_face_crop
import cv2
from PIL import Image
from tqdm import tqdm

batch_size = 32
workers = 0 if os.name == 'nt' else 8
suffix_list = ['jpg','jpeg','png','tiff']
yes = ['Y', 'y', 'Yes', 'YES', 'yes']
no = ['N', 'n', 'No', 'NO', 'no']
w, h = (512, 768)
#If it's a video: verify and convert into images

#If it's an image folder(s), verify files and proceed

def vid2img(vid_path, save_to, device):
	import skvideo.io.ffprobe as video_descriptor
	desc = video_descriptor(vid_path)
	rotate = True if desc['video']['@width'] > desc['video']['@height'] else False
	print ("Converting video {}\nto: images in {}".format(vid_path, save_to))
	frame_num=0
	cap = cv2.VideoCapture(vid_path)
	
	pbar = tqdm(total = cap.get(cv2.CAP_PROP_FRAME_COUNT))
	# Do while
	ret, frame = cap.read()

	# Extract frame by frame, save if face is found
	while ret:
		pbar.update(1)
		frame_num+=1
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		crop = get_face_crop(frame, device)
		if crop is not None:
			img = Image.fromarray(frame)
			#if rotate:
			#	img = img.rotate(90)
			img.save(os.path.join(save_to,"frame{}.jpg".format(frame_num)))
		ret, frame = cap.read()
	pbar.close()
			

def verify_img_folder(base_path, rotate = False):
	data_dir = os.path.join(base_path, 'data')
	if not os.path.exists(data_dir):
		print ("Path does not exist or not a path")

	print('Verifying image folder')
	for root, dirs, files in os.walk(data_dir):
		for file in files:
			if not (any([file.endswith(img_suffix) for img_suffix in suffix_list]) or file.startswith('.')):
				print ("Some files in the path or it's subfolders contains a file that is not an image:\n")
				print (os.path.join(root,file))
				exit(1)
	if rotate:
		rotate_images(data_dir)


def create_cropped_images(base_path, device = 'cpu', resize = False):
	mtcnn = face_extractor(device)
	data_dir = os.path.join(base_path, 'data')

	# Creating cropped images dataset.
	dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((w, h)) if resize else None)
	dataset.samples = [(p, p.replace(data_dir, data_dir + '_cropped')) for p, _ in dataset.samples]

	# Load the dataset with the new cropped addreses
	loader = DataLoader(dataset, num_workers=workers, batch_size=1, collate_fn=training.collate_pil)

	# Create ID to class name
	dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}

	# Generate cropped faces (with user verification)
	user_q = '?'
	while user_q not in yes + no:
		user_q = input("This will create a new folder [if needed] in: {}_cropped\n Continue?\n".format(data_dir))
		if user_q in ['n','N']:
			break
		elif user_q in ['y','Y']:
			for i, (x, y) in tqdm(enumerate(loader), total = len(loader)):
				if not os.path.exists(y[0]):
					mtcnn(x, save_path=y)
					print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
		print ("Please decide yes or no and type your answer\n")


def get_train_samples(img_base_path, device='cpu'):
	# Setup the new data loader for cropped images
	dataset = datasets.ImageFolder(os.path.join(img_base_path ,'data_cropped'))
	dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
	loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

	# Verify faces and get the cropped images into memory
	trainX = []
	trainY = []
	mtcnn = face_extractor(device)
	print ("Getting faces...")
	for x, y in tqdm(loader):
		x_aligned, prob = mtcnn(x, return_prob=True)
		if x_aligned is not None:
			if float(prob)>0.99:
				trainX.append(x_aligned)
				trainY.append(dataset.idx_to_class[y])
		else:
			print('Face not detected in {}'.format(y))			
	return trainX, trainY



def rotate_images(data_dir):
	# Just rotates them
	for root, dirs, files in os.walk(data_dir):
		for img in files:
			img_path = os.path.join(img, root)
			img = Image.open(os.path.join(path,img_path))
			img = img.rotate(-90)
			img.save(img_path)
