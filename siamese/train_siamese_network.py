from keras.models import Sequential , Input,Model
from keras.layers import Conv2D, MaxPooling2D , Flatten ,Dense,Lambda
from keras import backend as K
from data_preprocessing import load_face_images

def euclidean_distance(vectors):
	vect1 , vect2 = vectors
	sum_square = K.sum(K.square(vect1-vect2),axis=-1,keepdims=True)
	return K.sqrt(K.maximum(sum_square,K.epsilon()))

def contrastive_loss(Y_true,D):
	margin = 1
	return K.mean(Y_true*K.square(D)+(1-Y_true)*K.maximum((margin-D),0))

def accuracy(y_true,y_pred):
	return K.mean(K.equal(y_true,K.cast(y_pred < .5 ,y_true.dtype)))

def create_shared_network(input_shape):
	model = Sequential()
	model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu',input_shape=input_shape))
	model.add(MaxPooling2D())
	model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
	model.add(Flatten())
	model.add(Dense(units=128,activation='sigmoid'))
	return model

def train_network():
	data_set = load_face_images()
	X_train,Y_train = data_set['training_data']
	input_shape = X_train[0][0].shape
	shared_network = create_shared_network(input_shape)
	input_top = Input(shape=input_shape)
	input_bottom = Input(shape=input_shape)
	output_top = shared_network(input_top)
	output_bottom = shared_network(input_bottom)
	distance = Lambda(euclidean_distance,output_shape=(1,))([output_top,output_bottom])
	model = Model(inputs=[input_top,input_bottom],outputs = distance)
	model.compile(loss=contrastive_loss,optimizer='adam',metrics=[accuracy])
	model.fit([X_train[:,0],X_train[:,1]],Y_train,batch_size=64,epochs=10)
	model.save('model/siamese_nn.h5')

if __name__ == "__main__":
	train_network()

