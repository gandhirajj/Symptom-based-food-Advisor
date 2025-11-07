from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load model and preprocessing objects
model = load_model(r"C:\Users\GANDHIRAJ J\Downloads\dis\disease_food_model.h5")

print("Model loaded successfully!")
with open(r"C:\Users\GANDHIRAJ J\Downloads\dis\tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open(r"C:\Users\GANDHIRAJ J\Downloads\dis\label_encoders.pkl", "rb") as f:
    label_encoder_disease, label_encoder_prescription = pickle.load(f)
with open(r"C:\Users\GANDHIRAJ J\Downloads\dis\maxlen.pkl", "rb") as f:
    max_length = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    patient_problem = request.form['problem']
    sequence = tokenizer.texts_to_sequences([patient_problem])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    
    prediction = model.predict(padded_sequence)
    disease_index = np.argmax(prediction[0], axis=1)[0]
    prescription_index = np.argmax(prediction[1], axis=1)[0]
    
    disease_predicted = label_encoder_disease.inverse_transform([disease_index])[0]
    prescription_predicted = label_encoder_prescription.inverse_transform([prescription_index])[0]
    
    return render_template('result.html', problem=patient_problem,
                           food=disease_predicted, nutrient=prescription_predicted)

if __name__ == '__main__':
    app.run(debug=True)
