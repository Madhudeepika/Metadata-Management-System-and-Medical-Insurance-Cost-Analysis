# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import shap
import numpy as np
import openai

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open(model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Add your OpenAI API key here
openai.api_key = 'your_openai_api_key'

@app.route('/')
def home():
    return render_template('/content/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data, index=[0])
    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)
    prediction = prediction[0]

    # Explain prediction using SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(df_scaled)
    explanation_data = shap.Explanation(shap_values[0], base_values=explainer.expected_value, data=df_scaled[0]).data.tolist()

    # Generate explanation using GPT-3.5
    explanation_text = generate_explanation(explanation_data)

    return jsonify({
        'prediction': prediction,
        'explanation': explanation_text
    })

def generate_explanation(explanation_data):
    explanation_prompt = f"Given the SHAP values {explanation_data}, explain the prediction of medical insurance charges."
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=explanation_prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

if __name__ == '__main__':
    app.run(debug=True)
