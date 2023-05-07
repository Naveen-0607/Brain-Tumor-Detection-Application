# Import necessary libraries
import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename


# Set allowed extensions for image upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_path =r'C:\Users\rohit\Downloads\Brain Tumor Detection App\model\Finalmodel05.h5'
model = tf.keras.models.load_model(model_path)

# Define route for home page
@app.route('/')
def home():
    return render_template('home.html')
# Define function to check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define route for brain tumor detection page
# @app.route('/detect', methods=['GET', 'POST'])
# def detect():
#     if request.method == 'POST':
#         # Check if file was uploaded
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files['file']
#         # Check if file has an allowed extension
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             app.config['UPLOAD_FOLDER'] = r"C:\Users\rohit\Downloads\Brain Tumor Detection App\UPLOAD_FOLDER"
#             # Save file to upload folder
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             # Call function to detect brain tumor
#             result = detect_tumor(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             # Render template with result
#             return render_template('result.html', result=result)
#     return render_template('detect.html')

# Define function to detect brain tumor
# Define route for brain tumor detection page
@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # Check if file has an allowed extension
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            app.config['UPLOAD_FOLDER'] = r"C:\Users\rohit\Downloads\Brain Tumor Detection App\UPLOAD_FOLDER"
            # Save file to upload folder
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Call function to detect brain tumor
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img= tf.keras.preprocessing.image.load_img(file_path, target_size=(128,128))
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
            x = np.array(img.resize((128,128)))
            x = x.reshape(1,128,128,3)   
            result= model.predict(x)
            classification = np.where(result == np.amax(result))[1][0]
            print(classification)
            if classification == 0:
                res = 'Its a Brain Tumor'
            else: 
                res = "Its not a Brain Tuumor"
            prediction=str(result[0][classification]*100)
            # Render template with result
            return render_template('result.html', res=res)
    return render_template('detect.html')

# def names(number):
#     if number==0:
#         return 'Its a brain tumor '
#     else:
#         return 'Its  not a brain tumor'
# def detect_tumor(filename):
#     # Load and preprocess the image
#     img= tf.keras.preprocessing.image.load_img(filename, target_size=(128,128))
#     x = tf.keras.preprocessing.image.img_to_array(img)
#     x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
#     x = np.array(img.resize((128,128)))
#     x = x.reshape(1,128,128,3)   
#     result= model.predict(x)
#     classification = np.where(result == np.amax(result))[1][0]
#     prediction=str(result[0][classification]*100) +  names(classification)
#     # Return the result as a dictionary
#     return render_template('result.html',result=result,prediction=prediction)
    

# Run the app
if __name__ == '__main__':
    app.run(debug=True)     