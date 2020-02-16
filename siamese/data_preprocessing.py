from keras.preprocessing.image import load_img , img_to_array
import numpy as np 
import os
import random

def load_face_images():
	faces_dir = 'data/'
	X_train ,Y_train = [] ,[]
	X_test, Y_test = [], []
	sub_folders = sorted([file.path for file in os.scandir(faces_dir) if file.is_dir()])
	for index, folder_name in enumerate(sub_folders):
		for file in os.listdir(folder_name):
			full_path = folder_name+'/'+file
			img = load_img(full_path,color_mode='grayscale')
			img = img_to_array(img).astype('float32')/255
			if index < 35:
				X_train.append(img)
				Y_train.append(index)
			else:
				X_test.append(img)
				Y_test.append(index-35)
	X_train = np.array(X_train)
	print(X_train.shape)
	Y_train = np.array(Y_train)
	X_test = np.array(X_test)
	Y_test = np.array(Y_test)
	num_classes = len(np.unique(Y_train))
	train_pairs,train_labels = create_pairs(X_train,Y_train,len(np.unique(Y_train)))
	test_pairs,test_labels = create_pairs(X_test,Y_test,len(np.unique(Y_test)))
	return {'training_data':(train_pairs,train_labels),'testing_data':(test_pairs,test_labels)}

def create_pairs(X,Y,num_classes):
	pairs , labels = [] , []
	class_idx = [np.where(Y==i)[0]for i in range(num_classes)]
	min_images = min([len(class_idx[i]) for i in range(num_classes)]) -1
	for c in range(num_classes):
		for n in range(min_images):
			# create positive pairs
			img1 = X[class_idx[c][n]]
			img2 = X[class_idx[c][n+1]]
			pairs.append((img1,img2))
			labels.append(1)

			# create negative pair
			neg_list = list(range(num_classes))
			neg_list.remove(c)

			#select random class from a list
			neg_c = random.sample(neg_list,1)[0]
			img2 = X[class_idx[neg_c][n]]
			pairs.append((img1,img2))
			labels.append(0)
	return np.array(pairs),np.array(labels)

if __name__ == "__main__":
	labels , pairs = load_face_images()
	print(len(labels))
	print(len(pairs))