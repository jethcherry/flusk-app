from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the pickle model
with open('business_growth_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the data
    Market_Demand = request.form.get('Market Demand')
    Competitive_Landscape = request.form.get('Competitive Landscape')
    Economic_Conditions = request.form.get('Economic Conditions')
    Technology_Advancements = request.form.get('Technology Advancements')
    Financial_Resources = request.form.get('Financial Resources')
    Human_Resources = request.form.get('Human Resources')
    Marketing_Sales = request.form.get("Marketing and Sales Effectiveness")
    Product_Quality = request.form.get("Product/Service Quality")
    Regulatory_Environment = request.form.get("Regulatory Environment")
    Customer_Satisfaction = request.form.get("Customer Satisfaction")
    Supply_Chain_Management = request.form.get("Supply Chain Management")
    Geopolitical_Factors = request.form.get("Geopolitical Factors")

    # Create a NumPy array with the input features
    input_data = np.array([[Market_Demand, Competitive_Landscape, Economic_Conditions,
                            Technology_Advancements, Financial_Resources, Human_Resources,
                            Marketing_Sales, Product_Quality, Regulatory_Environment,
                            Customer_Satisfaction, Supply_Chain_Management, Geopolitical_Factors]])

    # Make prediction
    result = model.predict(input_data)[0]

    return jsonify({'Target': str(result)})

if __name__ == '__main__':
    
    app.run(debug=True)
