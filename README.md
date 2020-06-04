# FaceDet_streaming
Train and detect faces live from IP cameras or input video

Uses code adapted from facedet-pytorch repo:
https://github.com/timesler/facenet-pytorch


installation:
cd to the desired installation folder.

Clone the repo:
```
git clone https://github.com/PyxAI/FaceDet_streaming
cd FaceDet_streaming
```
Install the dependencies :
`pip install -r requirements.txt`

Place the training images in the data directory in the format:
data/person1/img1.jpg
data/person1/img2.jpeg
...
data/person2/img.jpg
...

Usage: 
train the model:
`python train.py`

start live camera stream:
`python detect.py`
