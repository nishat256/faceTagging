# faceTagging
You ever wondered how Amazon/Netflix shows characters name with details at left side as soon as character comes into scene. I tried to
replicate same thing using Face Recognition technique with the help of pre-trained **VGG-face** model to create embeddings for faces that helps
in recognition process.

## Installation Prerequisites

1) Dowload VGG-face mode
   - pip install keras-vggface

2) Python Packages
   - opencv
   - numpy
  
## On Boarding Process
 
 1) Place images of face of different characters in movie/video in **faces/** folder with character name as filename.
     - You could use face_detection.py file to crop faces out of image
     - You can also use two or more images of the same character in order to improve recognition capabilities but files name must be
        name of character plus number **ex:-** nishat1.jpeg, nishat2.jpeg etc.
   
 2) Place images of character that you want to show at left side of video everytime the character appears in the scene in **character_photo/** 
    folder with same name as you placed face in **faces/** folder.
  
 3) Run command python **face_onboarding.py**
 
## Face Tagging(Prediction)

You can tag faces in an image using **tag_faces_in_photo** function or a complete video using **tag_faces_in_video** from 
**process_video.py** file.


I tried this for the bollywood movie **KICK**.

Sample Input:
![alt text](https://github.com/nishat256/faceTagging/blob/master/input/image1.jpg)

Sample Output:
![alt text](https://github.com/nishat256/faceTagging/blob/master/output/sample1.jpg)
        
