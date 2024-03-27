# image_prediction_app/views.py
from django.shortcuts import render
from django.http import HttpResponse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

def predict_image(request):
    if request.method == 'POST':
        model_path = request.POST.get('model_path')
        class_path = request.POST.get('class_path')
        image_path = request.POST.get('image_path')

        # Load model
        model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
        input_shape = model.input_shape[1:]

        # Image preprocessing
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=input_shape)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255

        # Prediction
        result = model.predict(img)
        show=""
        lines = [line.replace("\n", "") for line in open(class_path, "r").readlines()]
        classes = {i: lines[i] for i in range(len(lines))}
        prediction_probability = {i: j for i, j in zip(classes.values(), result.tolist()[0])}
        if prediction_probability["yes"]> prediction_probability["no"]:
            show="cancer detected"
        else:
            show="cancer not detected"



        return render(request, 'index.html', {'prediction': show})
    else:
        return render(request, 'index.html')
