# FaceDet streaming
Train and detect faces live from IP cameras or input video

Uses code adapted from facedet-pytorch repo:
https://github.com/timesler/facenet-pytorch

<i>The following demo was done on a live feed from my webcam on i5 cpu only inference</i><br>
<img src = 'https://github.com/PyxAI/FaceDet_streaming/blob/master/ezgif.com-optimize.gif?raw=true'>
Using <b>MTCNN</b> to locate faces in images, a pretrained resnet to extract face embeddings and a Linear SVM to classify the images.

To speed things up, I am using the mtcnn's <i>detect</i> method to return only the bounding boxes, and then extract the faces from the original frame.

To train, you can provide a video of a person (with no other people in the video as to not affect detection)
You can also just add a folder with the images of the person to the data folder.
you can add as many people as you'd like.

In the future, adding a clustering algorythem to extract only the most seen person in a video can be added.


installation:
cd to the desired installation folder.

Clone the repo:
```
git clone https://github.com/PyxAI/FaceDet_streaming
cd FaceDet_streaming
```
Install the dependencies :
`pip install -r requirements.txt`

<h3>Preperation:</h3>
Training images should be in separate folders with the label name as folder name.
The format is as follows:
<img_base_path>/data/person1/img1.jpg
<img_base_path>/data/person1/img2.jpeg
<img_base_path>/data/person1/some_name.jpeg
...
<img_base_path>/data/person2/img.jpg
...

Provide the <img_base_path> to the training image folder in the settings.ini file under img_base_path 

  - To extract images from a video, provide the video_path and video_label in the settings.ini file and set `--input video` as argument.

provide arguments where to prepare the data from:
 ["webcam", "image_folder", "video"]
 
Run:
`python prepare.py -input image_folder`

<h3>Training</h3>
Select the classifier in settings.ini
the options are: ["svm", "logisticregression", "bayes"], it's recommended to stay with the default SVM.

`python train.py`

<h3>Run inference</h3>
Arguments for detection:<br>
input: ["webcam", "ipcam"]<br>
If using IP cam, make sure to include the address of the RTSP stream with user:pass included (can see example in the script)<br>
Alternativly, insert username and password in the terminal, but provied the address of the stream in the settings file nonetheless.

`python detect.py -input webcam`

