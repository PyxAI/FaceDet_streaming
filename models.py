#models

#set parameters from settings file

def get_resnet(device, classify = True):
	from facenet_pytorch import InceptionResnetV1
	resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
	resnet.classify = classify
	return resnet


def face_extractor(device, margin = 10):
	from facenet_pytorch import MTCNN
	mtcnn = MTCNN(\
	image_size=160, margin=margin, min_face_size=20,\
	thresholds=[0.7, 0.8, 0.8], factor=0.709, post_process=True,device=device)
	return mtcnn

def get_model(model, kernel=None):
	if model == 'svm':
		from sklearn import svm
		clf = svm.SVC(probability=True, decision_function_shape='ovr')
	elif model == 'ridge':
		from sklearn.linear_model import Ridge
		clf = Ridge(alpha=1.0)	
	elif model.lower() =='logisticregression':
		from sklearn.linear_model import LogisticRegression
		clf = LogisticRegression(random_state=0, multi_class='ovr')

	return clf

def get_norm(norm_type = 'l2'):
	from sklearn.preprocessing import Normalizer
	return Normalizer(norm_type)

def get_encoder():
	from sklearn.preprocessing import LabelEncoder
	return LabelEncoder()




