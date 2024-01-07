# How to load and use the model file :

- When you click Download 
  1. Model file in .h5 format &rarr; {model_name}{project_id}.h5
  2. A file with list of classes &rarr; classes.names 
- Once you've downloaded the model file you can test it in your local system with the code snippet. Please use the Github repo with sample code with the link provided in export model page. 
- You need to have python installed in your system. python 2.7+ and 3.4+ comes with pip and virtualenv along with installation. Create an environment and install the required packages.
- Run the following command to install requirements using pip

        python -m pip install -r requirements.txt
- If you're using conda use the following command to install requirements
        
        conda install --file requirements.txt
- Copy the code into a file and save it as prediction.py. Then run the following command and pass as arguments the path to the downloaded model file and image file you want to predict with the model 

        python prediction.py --model_path path/to/downloaded/{model_name}{project_id}.h5 --class_path path/to/downloaded/classes.names --image_path path/to/image_file
-  Please note if you're using python3.4 or above, try replacing **python** to **python3** if you're facing error while running the above command
