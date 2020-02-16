from keras.models import load_model
from train_siamese_network import euclidean_distance ,contrastive_loss
import cv2 
model = load_model('model/siamese_nn.h5',
	                custom_objects = {'contrastive_loss':contrastive_loss,'euclidean_distance':euclidean_distance})
true_img = cv2.imread('image0.jpg',0)
true_img = true_img.astype('float32')/255
true_img = cv2.resize(true_img,(92,112)) 
true_img = true_img.reshape(1,true_img.shape[0],true_img.shape[1],1)

face_img = cv2.imread('image2.jpg',0)
face_img = face_img.astype('float32')/255
face_img = cv2.resize(face_img,(92,112)) 
face_img = face_img.reshape(true_img.shape)
result = model.predict([true_img,face_img])[0][0]
print(result)
similarity_score = 1 - result
print(similarity_score)