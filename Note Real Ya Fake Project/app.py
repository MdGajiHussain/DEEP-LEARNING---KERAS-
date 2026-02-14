# # from flask import Flask, render_template, request
# # import pickle
# # import numpy as np
# # from tensorflow.keras.models import load_model



# # # Now you can use this model to make predictions

# # # Initialize the Flask app
# # app = Flask(__name__)

# # # Load the model and scaler using pickle
# # # Load the model
# # model = load_model('models/model.h5')
# # with open('models/scaler.pkl', 'rb') as scaler_file:
# #     scaler = pickle.load(scaler_file)


# # # Define a function for making predictions
# # def make_prediction(input_data):
# #     # Preprocess input data (apply scaling)
# #     input_data_scaled = scaler.transform(input_data)  # Use transform instead of fit_transform

# #     # Use the trained model to predict the class
# #     predictions = model.predict(input_data_scaled)

# #     # Convert prediction to binary (0 or 1)
# #     predicted_classes = (predictions > 0.5).astype(int)

# #     return predicted_classes


# # # Define the route for the home page
# # @app.route('/')
# # def index():
# #     return render_template('index.html')


# # # Define the route for prediction
# # @app.route('/predict', methods=['POST', 'GET'])
# # def predict():
# #     if request.method == 'POST':
# #         # Get form data
# #         VWTI = float(request.form['VWTI'])
# #         SWTI = float(request.form['SWTI'])
# #         CWTI = float(request.form['CWTI'])
# #         EI = float(request.form['EI'])

# #         # Prepare input data for prediction
# #         input_data = np.array([[VWTI, SWTI, CWTI, EI]])

# #         # Get the prediction
# #         result = make_prediction(input_data)
# #         print(result)
# #         if result[0] == 1:
# #             output = "real"
# #         else:
# #             output = "fake"
# #         print(output)
# #         # Pass the result to the template
# #         return render_template('index.html',prediction=output)  # result[0] gives the first element in array (0 or 1)

# #     return render_template('index.html', prediction=None)


# # if __name__ == '__main__':
# #     app.run(debug=True)



# from flask import Flask, render_template, request
# import pickle
# import numpy as np
# import os
# from tensorflow.keras.models import load_model

# # ----------------------------
# # Suppress TensorFlow INFO/WARN messages
# # ----------------------------
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # ----------------------------
# # Initialize the Flask app
# # ----------------------------
# app = Flask(__name__)

# # ----------------------------
# # Load the model and scaler
# # ----------------------------
# MODEL_PATH = 'models/model.h5'
# SCALER_PATH = 'models/scaler.pkl'

# try:
#     model = load_model(MODEL_PATH)
# except Exception as e:
#     print(f"Error loading model: {e}")
#     model = None

# try:
#     with open(SCALER_PATH, 'rb') as f:
#         scaler = pickle.load(f)
# except Exception as e:
#     print(f"Error loading scaler: {e}")
#     scaler = None

# # ----------------------------
# # Prediction function
# # ----------------------------
# def make_prediction(input_data):
#     """
#     input_data: 2D array [[VWTI, SWTI, CWTI, EI]]
#     Returns: 0 or 1
#     """
#     if scaler is None or model is None:
#         raise Exception("Model or scaler not loaded properly.")

#     # Convert input to 2D NumPy array
#     input_array = np.array(input_data).reshape(1, -1)

#     # Scale input
#     input_scaled = scaler.transform(input_array)

#     # Predict probability
#     pred_prob = model.predict(input_scaled)

#     # Convert probability to binary
#     pred_class = (pred_prob > 0.5).astype(int)

#     return int(pred_class[0][0])

# # ----------------------------
# # Home route
# # ----------------------------
# @app.route('/')
# def index():
#     return render_template('index.html', prediction=None)

# # ----------------------------
# # Prediction route
# # ----------------------------
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Collect form inputs safely
#         VWTI = float(request.form.get('VWTI', 0))
#         SWTI = float(request.form.get('SWTI', 0))
#         CWTI = float(request.form.get('CWTI', 0))
#         EI = float(request.form.get('EI', 0))

#         input_data = [[VWTI, SWTI, CWTI, EI]]

#         # Make prediction
#         result = make_prediction(input_data)

#         # Map to readable output
#         output = "real" if result == 1 else "fake"

#         return render_template('index.html', prediction=output)

#     except ValueError:
#         return render_template('index.html', prediction="Error: Invalid input. Please enter numeric values.")
#     except Exception as e:
#         return render_template('index.html', prediction=f"Error: {str(e)}")

# # ----------------------------
# if __name__ == '__main__':
#     app.run(debug=True)







# import streamlit as st
# import pickle
# import numpy as np
# from tensorflow.keras.models import load_model

# # -------------------------------
# # Load the model and scaler
# # -------------------------------
# model = load_model('models/model.h5')

# with open('models/scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)

# # -------------------------------
# # Prediction function
# # -------------------------------
# def make_prediction(input_data):
#     """
#     input_data: 2D array-like [[VWTI, SWTI, CWTI, EI]]
#     """
#     # Scale input
#     input_scaled = scaler.transform(input_data)

#     # Predict using Keras model
#     pred_prob = model.predict(input_scaled)

#     # Convert probability to binary
#     pred_class = (pred_prob > 0.5).astype(int)

#     return pred_class[0][0]  # return single value (0 or 1)

# # -------------------------------
# # Streamlit UI
# # -------------------------------
# st.title("Fake vs Real Prediction App")

# st.write("Enter the input values below:")

# # Get user input
# VWTI = st.number_input("VWTI", value=0.0)
# SWTI = st.number_input("SWTI", value=0.0)
# CWTI = st.number_input("CWTI", value=0.0)
# EI = st.number_input("EI", value=0.0)

# # Predict button
# if st.button("Predict"):
#     input_data = np.array([[VWTI, SWTI, CWTI, EI]])
#     result = make_prediction(input_data)

#     if result == 1:
#         output = "Real Note"
#     else:
#         output = "Fake Note"

#     st.success(f"Prediction: {output}")





import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image  # For image display

# -------------------------------
# Load the model and scaler
# -------------------------------
model = load_model('models/model.h5')

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# -------------------------------
# Prediction function
# -------------------------------
def make_prediction(input_data):
    """
    input_data: 2D array-like [[VWTI, SWTI, CWTI, EI]]
    """
    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict using Keras model
    pred_prob = model.predict(input_scaled)

    # Convert probability to binary
    pred_class = (pred_prob > 0.5).astype(int)

    return pred_class[0][0]  # return single value (0 or 1)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Fake vs Real Prediction App")

st.write("Enter the input values below:")

# Get user input
VWTI = st.number_input("VWTI", value=0.0)
SWTI = st.number_input("SWTI", value=0.0)
CWTI = st.number_input("CWTI", value=0.0)
EI = st.number_input("EI", value=0.0)

# Predict button
if st.button("Predict"):
    input_data = np.array([[VWTI, SWTI, CWTI, EI]])
    result = make_prediction(input_data)

    if result == 1:
        output = "Real Note"
        img = Image.open("static/real_note.png")
    else:
        output = "Fake Note"
        img = Image.open("static/fake_note.png")

    st.success(f"Prediction: {output}")
    st.image(img, caption=output, width=400)
