
def get_resnet(device, classify = True):
	from facenet_pytorch import InceptionResnetV1
	resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
	resnet.classify = classify
	return resnet

def face_extractor(device, margin = 15, scale_factor = 0.709):
	from facenet_pytorch import MTCNN
	mtcnn = MTCNN(\
	image_size=160, margin=margin, min_face_size=20,\
	thresholds=[0.7, 0.8, 0.8], factor=scale_factor, post_process=True,device=device)
	return mtcnn

def get_model(model, kernel=None):
	if 'svm' in model:
		from sklearn import svm # Pretty good, sometimes is wrong
		clf = svm.SVC(kernel = 'linear', probability=True,\
		gamma = 0.0001, C = 1) #decision_function_shape='ovr')
	elif model.lower() =='logisticregression':  # Too wrong
		from sklearn.linear_model import LogisticRegression
		clf = LogisticRegression(random_state=0, multi_class='ovr')
	elif model.lower() == 'bayes':
		from sklearn.naive_bayes import BernoulliNB # Almost always returns probabilities 0 or 1
		clf = BernoulliNB()

	return clf

def get_norm(norm_type = 'l2'):
	from sklearn.preprocessing import Normalizer
	return Normalizer(norm_type)

def get_encoder():
	from sklearn.preprocessing import LabelEncoder
	return LabelEncoder()

