import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import argparse

# get path to the downloaded model and image to predict as arguments 
parser = argparse.ArgumentParser(description='Argument to collect info')
parser.add_argument('--model_path', help='path to the downloaded model')
parser.add_argument('--class_path', help='path to the downloaded classnames file')
parser.add_argument('--image_path', help='path to the test image')

args = parser.parse_args()
model_path = args.model_path
class_path = args.class_path
image_path = args.image_path

#load model
model = tf.keras.models.load_model(model_path,custom_objects={'KerasLayer':hub.KerasLayer})
input_shape = model.input_shape[1:]

# image preprocessing
img = tf.keras.preprocessing.image.load_img(image_path,target_size=input_shape)
img = tf.keras.preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img/255

# prediction
result = model.predict(img)
lines = [line.replace("\n","") for line in open(class_path,"r").readlines()]
classes = {i:lines[i] for i in range(len(lines))}
prediction_probability = {i:j for i,j in zip(classes.values(),result.tolist()[0])}

print("\n\nPrediction: \n")
print(prediction_probability)