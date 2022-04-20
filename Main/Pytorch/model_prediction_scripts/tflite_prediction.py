# https://www.tensorflow.org/lite/guide/inference

import numpy as np
import tensorflow as tf

import os
import torchvision.transforms as transforms
from PIL import Image


image_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224))
])


interpreter = tf.lite.Interpreter(model_path="../weights/cat_vs_dog.tflite")
interpreter.allocate_tensors()

cat = []
dog = []

for path in os.listdir("../TestData/Cat/"):
	
	path = "../TestData/Cat/{}".format(path)
	image = Image.open(path)
	image = image_transformation(image).numpy()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	
	input_shape = input_details[0]["shape"]
	img = image.reshape(input_shape)

	interpreter.set_tensor(input_details[0]["index"],img)

	interpreter.invoke()

	output_data = interpreter.get_tensor(output_details[0]["index"])
	output_data = np.argmax(output_data,axis=1).item()
	
	cat.append(output_data)

for path in os.listdir("../TestData/Dog/"):
	path = "../TestData/Dog/{}".format(path)
	image = Image.open(path)
	image = image_transformation(image).numpy()
	
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	input_shape = input_details[0]["shape"]
	img = image.reshape(input_shape)

	interpreter.set_tensor(input_details[0]["index"],img)

	interpreter.invoke()

	output_data = interpreter.get_tensor(output_details[0]["index"])
	output_data = np.argmax(output_data,axis=1).item()
	
	dog.append(output_data)


print(cat)
print(dog)


