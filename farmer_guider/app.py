from flask import Flask, request, render_template
import numpy as np
import joblib
import pandas as pd
import mysql.connector

# Load the KMeans model, scaler, and data
kmeans = joblib.load('models/kmeans_model.lb')
std = joblib.load('models/standardscaler.lb')
df = pd.read_csv('models/filteringdata.csv')

# Mapping clusters to crop details
item_images = {
    0: [{'name': 'Pigeonpeas', 'image': 'pigeonpeas.jpg'}, {'name': 'Moth Beans', 'image': 'mothbeans.jpg'},
        {'name': 'Mung Bean', 'image': 'mungbean.jpg'}, {'name': 'Black Gram', 'image': 'blackgram.jpg'},
        {'name': 'Lentil', 'image': 'lentil.jpg'}, {'name': 'Mango', 'image': 'mango.jpg'},
        {'name': 'Orange', 'image': 'orange.jpg'}, {'name': 'Papaya', 'image': 'papaya.jpg'}],
    1: [{'name': 'Maize', 'image': 'maize.jpg'}, {'name': 'Lentil', 'image': 'lentil.jpg'},
        {'name': 'Banana', 'image': 'banana.jpg'}, {'name': 'Papaya', 'image': 'papaya.jpg'},
        {'name': 'Coconut', 'image': 'coconut.jpg'}, {'name': 'Cotton', 'image': 'cotton.jpg'},
        {'name': 'Jute', 'image': 'jute.jpg'}, {'name': 'Coffee', 'image': 'coffee.jpg'}],
    2: [{'name': 'Grapes', 'image': 'grapes.jpg'}, {'name': 'Apple', 'image': 'apple.jpg'}],
    3: [{'name': 'Pigeonpeas', 'image': 'pigeonpeas.jpg'}, {'name': 'Pomegranate', 'image': 'pomegranate.jpg'},
        {'name': 'Orange', 'image': 'orange.jpg'}, {'name': 'Papaya', 'image': 'papaya.jpg'},
        {'name': 'Coconut', 'image': 'coconut.jpg'}],
    4: [{'name': 'Rice', 'image': 'rice.jpg'}, {'name': 'Pigeonpeas', 'image': 'pigeonpeas.jpg'},
        {'name': 'Papaya', 'image': 'papaya.jpg'}, {'name': 'Coconut', 'image': 'coconut.jpg'},
        {'name': 'Jute', 'image': 'jute.jpg'}, {'name': 'Coffee', 'image': 'coffee.jpg'}],
    5: [{'name': 'Pigeonpeas', 'image': 'pigeonpeas.jpg'}, {'name': 'Moth Beans', 'image': 'mothbeans.jpg'},
        {'name': 'Lentil', 'image': 'lentil.jpg'}, {'name': 'Mango', 'image': 'mango.jpg'}],
    6: [{'name': 'Watermelon', 'image': 'watermelon.jpg'}, {'name': 'Muskmelon', 'image': 'muskmelon.jpg'}],
    7: [{'name': 'Chickpea', 'image': 'chickpea.jpg'}, {'name': 'Kidney Beans', 'image': 'kidneybeans.jpg'},
        {'name': 'Pigeonpeas', 'image': 'pigeonpeas.jpg'}, {'name': 'Lentil', 'image': 'lentil.jpg'}]
}

app = Flask(__name__)

# Database connection
def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='@Shubham2003',
        database='farmer_guidance'
    )

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/project')
def project():
    return render_template('project.html')

@app.route('/output')
def output():
    return render_template('output.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Retrieve form data
    n = float(request.form.get('n'))
    p = float(request.form.get('p'))
    k = float(request.form.get('k'))
    temperature = float(request.form.get('temperature'))
    humidity = float(request.form.get('humidity'))
    ph = float(request.form.get('ph'))
    rainfall = float(request.form.get('rainfall'))

    # Prepare input data for prediction
    input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])

    # Transform the input data
    transformed_input_data = std.transform(input_data)

    # Make prediction using the KMeans model
    cluster = kmeans.predict(transformed_input_data)[0]
    cluster = int(cluster)  # Convert numpy.int32 to int

    # Get crop options for the predicted cluster
    crops = item_images.get(cluster, [{'name': 'Unknown', 'image': 'default.jpg'}])
    crop_names = ', '.join([crop['name'] for crop in crops])

    # Save the prediction result to the database
    connection = get_db_connection()
    cursor = connection.cursor()
    sql = """
    INSERT INTO predictions (N, P, K, temperature, humidity, ph, rainfall, crop)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(sql, (n, p, k, temperature, humidity, ph, rainfall, crop_names))
    connection.commit()
    cursor.close()
    connection.close()

    # Pass the predicted cluster to the output page
    return render_template('output.html', crops=crops)


if __name__ == '__main__':
    app.run(debug=True)
