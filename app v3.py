import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from flask import Flask, request
import html

app = Flask(__name__)

html_form = """
<!DOCTYPE html>
<html>
<head>
    <title>Medical Image (X-ray, CT scan, MRI) -Based Disease Detection SystemU</title>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <script>
        $(document).ready(function(){
            $('#submit_btn').click(function(){
                var model_path = $('#model_path').val();
                var class_path = $('#class_path').val();
                var image_path = $('#image_path').val();

                $.ajax({
                    type: 'POST',
                    url: '/predict',
                    data: {'model_path': model_path, 'class_path': class_path, 'image_path': image_path},
                    success: function(response){
                        $('#prediction_result').html(response);
                    }
                });
            });
        });
    </script>
</head>
<body>
    <h2>Medical Image (X-ray, CT scan, MRI) -Based Disease Detection SystemU</h2>
    <form id="prediction_form">
        <label for="model_path">Model Path:</label><br>
        <input type="text" id="model_path" name="model_path"><br><br>
        
        <label for="class_path">Class Path:</label><br>
        <input type="text" id="class_path" name="class_path"><br><br>
        
        <label for="image_path">Image Path:</label><br>
        <input type="text" id="image_path" name="image_path"><br><br>
        
        <input type="button" id="submit_btn" value="Predict">
    </form>
    <div id="prediction_result"></div>
</body>
</html>
"""

@app.route('/')
def index():
    return html_form

@app.route('/predict', methods=['POST'])
def predict():
    model_path = html.escape(request.form['model_path'])
    class_path = html.escape(request.form['class_path'])
    image_path = html.escape(request.form['image_path'])

    print("Model Path:", model_path)  # Print model path for debugging

    # Check if model file exists
    if not os.path.exists(model_path):
        return "Error: Model file not found"

    # Load model
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    except Exception as e:
        return "Error: " + str(e)

    input_shape = model.input_shape[1:]

    # Image preprocessing
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=input_shape)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255

    # Prediction
    result = model.predict(img)
    lines = [line.replace("\n","") for line in open(class_path,"r").readlines()]
    classes = {i:lines[i] for i in range(len(lines))}
    prediction_probability = {i:j for i,j in zip(classes.values(),result.tolist()[0])}

    return str(prediction_probability)

if __name__ == '__main__':
    app.run(debug=True)
