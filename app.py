from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained Pipeline model
try:
    model = joblib.load("athlete_model.pkl")
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Error: athlete_model.pkl not found. Please train and save the model first.")
    model = None

# Serve the HTML page
@app.route('/')
def index():
    """Serve the main dashboard HTML page"""
    return send_from_directory('.', 'index.html')

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    """Check if the API and model are ready"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the number of top 10 finishes for an athlete.
    
    Expected JSON body:
    {
        "age": float,
        "weight": float,
        "height": float,
        "gender": str ("MALE" or "FEMALE"),
        "division": str ("Individual" or "Team"),
        "avg_points": float
    }
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please ensure athlete_model.pkl exists.'
        }), 500
    
    try:
        # Get JSON data from request
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['age', 'weight', 'height', 'gender', 'division', 'avg_points']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Convert and validate numeric fields
        try:
            data['age'] = float(data['age'])
            data['weight'] = float(data['weight'])
            data['height'] = float(data['height'])
            data['avg_points'] = float(data['avg_points'])
        except ValueError as e:
            return jsonify({
                'error': f'Invalid numeric value: {str(e)}'
            }), 400
        
        # Validate ranges
        if not (0 < data['age'] < 120):
            return jsonify({'error': 'Age must be between 0 and 120'}), 400
        if not (30 < data['weight'] < 200):
            return jsonify({'error': 'Weight must be between 30 and 200 kg'}), 400
        if not (1.0 < data['height'] < 2.5):
            return jsonify({'error': 'Height must be between 1.0 and 2.5 m'}), 400
        if not (0 <= data['avg_points'] <= 500):
            return jsonify({'error': 'Average points must be between 0 and 500'}), 400
        
        # Validate categorical fields
        data['gender'] = data['gender'].upper().strip()
        if data['gender'] not in ['MALE', 'FEMALE']:
            return jsonify({'error': 'Gender must be MALE or FEMALE'}), 400
        
        data['division'] = data['division'].strip()
        if data['division'] not in ['Individual', 'Team']:
            return jsonify({'error': 'Division must be Individual or Team'}), 400
        
        # Create DataFrame for prediction
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Ensure non-negative integer result
        predicted_top10 = max(int(round(prediction)), 0)
        
        # Return result with additional info
        return jsonify({
            'success': True,
            'predicted_top10_finishes': predicted_top10,
            'input_data': data,
            'message': f'Cet athlète devrait réaliser environ {predicted_top10} finitions dans le top 10.'
        })
        
    except Exception as e:
        # Log the error for debugging
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': f'Erreur lors de la prédiction: {str(e)}'
        }), 500


# Additional endpoint for batch predictions
@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict for multiple athletes at once.
    
    Expected JSON body:
    {
        "athletes": [
            {"age": 28, "weight": 70, ...},
            {"age": 32, "weight": 60, ...}
        ]
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        athletes = data.get('athletes', [])
        
        if not athletes:
            return jsonify({'error': 'No athletes provided'}), 400
        
        # Create DataFrame
        df = pd.DataFrame(athletes)
        
        # Make predictions
        predictions = model.predict(df)
        predictions_int = [max(int(round(p)), 0) for p in predictions]
        
        # Combine results
        results = []
        for athlete, pred in zip(athletes, predictions_int):
            results.append({
                'athlete': athlete,
                'predicted_top10_finishes': pred
            })
        
        return jsonify({
            'success': True,
            'predictions': results,
            'count': len(results)
        })
        
    except Exception as e:
        print(f"Error during batch prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Check if model file exists
    if not os.path.exists('athlete_model.pkl'):
        print("\n⚠️  WARNING: athlete_model.pkl not found!")
        print("Please run the training script first to generate the model.\n")
    
    # Run Flask app
    print("\n🚀 Starting Flask server...")
    print("📍 Dashboard available at: http://127.0.0.1:5000")
    print("📍 API endpoint: http://127.0.0.1:5000/predict")
    print("\nPress CTRL+C to stop the server.\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)