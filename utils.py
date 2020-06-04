from tqdm import tqdm
from PIL import Image
from models import face_extractor, get_resnet
import numpy as np
import time

def getEmbedings(aligned, names, resnet):
	embbedX = list()
	embbedY = list()
	print ("Creating embeddings for all training images")
	for im, name in tqdm(zip(aligned, names), total = len(names)):
		std = im.std()
		mean = im.mean()
		im = (im - mean) / std
		emb = resnet(im.unsqueeze(0)).detach().numpy()
		embbedX.append(emb)
		embbedY.append(name)
	return np.array(embbedX), np.array(embbedY)


def collate_fn(x):
    return x[0]

def get_face_crop(image_src, device):
	mtcnn = face_extractor(device)
	if isinstance(image_src, np.ndarray): # When we get it from cv2
		img = Image.fromarray(image_src)
	elif isinstance(image_src, torch.Tensor):
		img = Image.fromarray(image_src)
	else:
		img = Image.open(image_src)
	img = mtcnn(img)
	return img

def clear_buffer(cap, frame_rate = 30):
    ret = True
    while ret:
        t1 = time.time()
        ret, _ = cap.read()
        if (time.time()-t1)> 1/frame_rate:
            break
