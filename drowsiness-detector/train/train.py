import csv
import numpy as np 
import keras 
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,Activation,Dropout,MaxPooling2D
from keras.activations import relu
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
#we will use images 26x34x1 (1 is for grayscale images)
height = 26
width = 34
dims = 1

def readCsv(path):

	with open(path,'r') as f:
		#read the scv file with the dictionary format 
		reader = csv.DictReader(f)
		rows = list(reader)

	#imgs is a numpy array with all the images
	#tgs is a numpy array with the tags of the images
	imgs = np.empty((len(list(rows)),height,width, dims),dtype=np.uint8)
	tgs = np.empty((len(list(rows)),1))
		
	for row,i in zip(rows,range(len(rows))):
			
		#convert the list back to the image format
		img = row['image']
		img = img.strip('[').strip(']').split(', ')
		im = np.array(img,dtype=np.uint8)
		im = im.reshape((26,34))
		im = np.expand_dims(im, axis=2)
		imgs[i] = im

		#the tag for open is 1 and for close is 0
		tag = row['state']
		if tag == 'open':
			tgs[i] = 1
		else:
			tgs[i] = 0
	
	#shuffle the dataset
	index = np.random.permutation(imgs.shape[0])
	imgs = imgs[index]
	tgs = tgs[index]

	#return images and their respective tags
	return imgs,tgs	

#make the convolution neural network
def makeModel():
	model = Sequential()

	model.add(Conv2D(32, (3,3), padding = 'same', input_shape=(height,width,dims)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64, (2,2), padding= 'same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (2,2), padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	
	model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy',metrics=['accuracy'])

	return model

def main():

	xTrain ,yTrain = readCsv('/Users/phuc1403/projects/simple-blink-detector/drowsiness-detector/train/dataset.csv')
	plt.imshow(xTrain[0], cmap='gray')  # Đặt cmap='gray' nếu hình ảnh là ảnh xám
	plt.title('Example Image')
	plt.axis('off')  # Tắt trục
	plt.show()
	# xTrain ,yTrain = readCsv('/Users/phuc1403/projects/Driver-Drowsiness-Detection-using-Deep-Learning/data/dataset3.csv')

	def randomize_data(x_data, y_data):
		# Kết hợp dữ liệu vào một mảng 2D
		combined = list(zip(x_data, y_data))
		# Ngẫu nhiên hoán đổi dữ liệu
		np.random.shuffle(combined)
		# Tách dữ liệu đã hoán đổi trở lại thành hai mảng riêng biệt
		x_data[:], y_data[:] = zip(*combined)
  
  
	randomize_data(xTrain, yTrain)

	#scale the values of the images between 0 and 1
	xTrain = xTrain.astype('float32')
	xTrain /= 255

	model = makeModel()

	#do some data augmentation
	datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        )
	datagen.fit(xTrain)


	x_train, x_test, y_train, y_test = train_test_split(xTrain, yTrain, test_size=0.2, random_state=42)

	#train the model
	model.fit_generator(datagen.flow(x_train,y_train,batch_size=32),
						steps_per_epoch=len(x_train) / 32, epochs=5)
 
	def test_model(model, x_test, y_test):
		# Đánh giá mô hình trên tập dữ liệu kiểm tra
		loss, accuracy = model.evaluate(x_test, y_test)
		print("Test Loss:", loss)
		print("Test Accuracy:", accuracy)

	test_model(model, x_test, y_test)
 
	#save the model
	model.save('blinkModel.hdf5')

if __name__ == '__main__':
	main()
