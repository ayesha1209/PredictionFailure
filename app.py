from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import traceback
import os
import logging
from logging.handlers import RotatingFileHandler
import sys


# Configure logging
def setup_logging(app):
    if not app.debug:
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.mkdir('logs')

        # Set up file handler with rotation
        file_handler = RotatingFileHandler('logs/tttf_api.log',
                                           maxBytes=10240, backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)

        app.logger.setLevel(logging.INFO)
        app.logger.info('TTTF API startup')


app = Flask(__name__)
CORS(app)

# Rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Global variable to store model
model_package = None


def load_model():
    """Load the enhanced model package"""
    global model_package
    model_paths = [
        'models/tttf_xgb_model_enhanced.pkl',
        'models/tttf_xgb_model_enhanced_backup.pkl'
    ]

    for path in model_paths:
        try:
            model_package = joblib.load(path)
            app.logger.info(f'Model loaded successfully from {path}')
            return True
        except FileNotFoundError:
            app.logger.warning(f'Model file not found: {path}')
            continue
        except Exception as e:
            app.logger.error(f'Error loading model from {path}: {str(e)}')
            continue

    app.logger.error('Failed to load model from any path')
    return False


def validate_input_data(input_data):
    """Validate input data structure and types"""
    if not model_package:
        return False, "Model not loaded"

    required_features = model_package['feature_names']

    # Check for missing features
    missing_features = [f for f in required_features if f not in input_data]
    if missing_features:
        return False, f"Missing required features: {missing_features}"

    # Validate data types
    label_encoders = model_package['label_encoders']

    for feature, value in input_data.items():
        if feature not in required_features:
            continue

        # Check categorical features
        if feature in label_encoders:
            if value not in label_encoders[feature].classes_ and str(value) not in label_encoders[feature].classes_:
                valid_options = list(label_encoders[feature].classes_)
                return False, f"Invalid value '{value}' for {feature}. Valid options: {valid_options}"

        # Check numerical features
        else:
            if value is not None and value != '':
                try:
                    float(value)
                except (ValueError, TypeError):
                    return False, f"Invalid numerical value '{value}' for {feature}"

    return True, "Valid"


def get_feature_info():
    """Get feature information for the frontend"""
    if not model_package:
        return None

    feature_names = model_package['feature_names']
    label_encoders = model_package['label_encoders']

    features = []
    for feature in feature_names:
        if feature in label_encoders:
            features.append({
                'name': feature,
                'type': 'categorical',
                'options': list(label_encoders[feature].classes_)
            })
        else:
            # Add guidance for numerical features
            guidance = ""
            if feature == 'volt':
                guidance = "Voltage reading (V)"
            elif feature == 'rotate':
                guidance = "Rotation speed (RPM)"
            elif feature == 'pressure':
                guidance = "Pressure reading (PSI)"
            elif feature == 'vibration':
                guidance = "Vibration level"
            elif feature == 'age':
                guidance = "Machine age in days"
            elif feature == 'error_count':
                guidance = "Recent error count"
            elif 'days_since' in feature:
                component = feature.split('_')[2]
                guidance = f"Days since {component} maintenance"

            features.append({
                'name': feature,
                'type': 'numerical',
                'guidance': guidance
            })

    return features


def create_example_data():
    """Create example data for quick testing"""
    if not model_package:
        return None

    feature_names = model_package['feature_names']
    label_encoders = model_package['label_encoders']

    example_data = {}

    # Get valid model if available
    if 'model' in label_encoders:
        valid_models = list(label_encoders['model'].classes_)
        example_data['model'] = valid_models[0] if valid_models else 'model1'

    # Set example values for other features
    example_values = {
        'volt': 168.5,
        'rotate': 415.2,
        'pressure': 98.7,
        'vibration': 45.3,
        'age': 150,
        'error_count': 2,
        'days_since_comp1_maint': 15,
        'days_since_comp2_maint': 8,
        'days_since_comp3_maint': 22,
        'days_since_comp4_maint': 12
    }

    for feature in feature_names:
        if feature not in example_data:
            example_data[feature] = example_values.get(feature, 100.0)

    return example_data


def make_prediction(input_data):
    """Make prediction using the loaded model"""
    try:
        # Extract model components
        model = model_package['model']
        scaler = model_package['scaler']
        imputer = model_package['imputer']
        label_encoders = model_package['label_encoders']
        feature_names = model_package['feature_names']
        target_names = model_package['target_names']

        # Create DataFrame from input
        sample_df = pd.DataFrame([input_data])
        X_sample = sample_df[feature_names].copy()

        # Process categorical features
        categorical_features = ['model']
        for cat_feat in categorical_features:
            if cat_feat in feature_names and cat_feat in label_encoders:
                le = label_encoders[cat_feat]
                original_value = X_sample[cat_feat].iloc[0]

                # Handle unknown categories
                if pd.isna(original_value) or str(original_value) not in le.classes_:
                    X_sample[cat_feat] = 'unknown'

                # Encode
                encoded_value = le.transform([str(X_sample[cat_feat].iloc[0])])[0]
                X_sample[cat_feat] = encoded_value

        # Process numerical features
        numerical_features = [f for f in feature_names if f not in categorical_features]
        if numerical_features:
            # Handle missing values and convert to float
            for feat in numerical_features:
                value = X_sample[feat].iloc[0]
                if pd.isna(value) or value == '' or str(value).lower() in ['nan', 'null', 'none']:
                    X_sample[feat] = np.nan
                else:
                    X_sample[feat] = float(value)

            X_sample[numerical_features] = imputer.transform(X_sample[numerical_features])

        # Scale features
        X_scaled = scaler.transform(X_sample)

        # Make prediction
        raw_prediction = model.predict(X_scaled)[0]
        prediction_floor = np.floor(raw_prediction)

        # Prepare results
        results = []
        for i, target in enumerate(target_names):
            component = target.replace('ttf_', '').replace('_weeks', '').upper()
            exact_weeks = float(raw_prediction[i])
            floor_weeks = int(prediction_floor[i])

            # Determine status
            if floor_weeks == 0:
                status = "CRITICAL"
                emoji = "🚨"
            elif floor_weeks <= 2:
                status = "URGENT"
                emoji = "🔴"
            elif floor_weeks <= 4:
                status = "SOON"
                emoji = "🟡"
            elif floor_weeks <= 8:
                status = "NORMAL"
                emoji = "🟢"
            else:
                status = "GOOD"
                emoji = "✅"

            results.append({
                'component': component,
                'exact_weeks': exact_weeks,
                'floor_weeks': floor_weeks,
                'status': status,
                'emoji': emoji
            })

        # Calculate summary
        min_weeks = int(min(prediction_floor))
        avg_weeks = float(np.mean(prediction_floor))
        critical_components = sum(1 for weeks in prediction_floor if weeks <= 2)

        # Maintenance recommendation
        if min_weeks == 0:
            recommendation = "IMMEDIATE maintenance required!"
            rec_detail = "Stop operations and inspect immediately"
        elif min_weeks <= 2:
            recommendation = "URGENT: Schedule maintenance within 2 weeks"
            rec_detail = "Prioritize this machine in maintenance schedule"
        elif min_weeks <= 4:
            recommendation = "Schedule maintenance within 1 month"
            rec_detail = "Monitor closely and prepare maintenance"
        elif min_weeks <= 8:
            recommendation = "Normal maintenance schedule (within 2 months)"
            rec_detail = "Continue regular monitoring"
        else:
            recommendation = "Machine in good condition"
            rec_detail = "Follow standard maintenance schedule"

        return {
            'success': True,
            'results': results,
            'summary': {
                'min_weeks': min_weeks,
                'avg_weeks': avg_weeks,
                'critical_components': critical_components,
                'total_components': len(target_names)
            },
            'recommendation': {
                'title': recommendation,
                'detail': rec_detail
            }
        }

    except Exception as e:
        app.logger.error(f'Prediction error: {str(e)}')
        if app.debug:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        else:
            return {
                'success': False,
                'error': 'Prediction failed. Please check your input data.'
            }


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/model-info')
@limiter.limit("10 per minute")
def model_info():
    """Get model information"""
    if not model_package:
        return jsonify({'success': False, 'error': 'Model not loaded'})

    features = get_feature_info()
    target_names = model_package['target_names']
    excluded_features = model_package.get('excluded_features', [])

    return jsonify({
        'success': True,
        'features': features,
        'targets': target_names,
        'excluded_features': excluded_features,
        'total_features': len(features)
    })


@app.route('/api/example-data')
@limiter.limit("10 per minute")
def example_data():
    """Get example data for testing"""
    if not model_package:
        return jsonify({'success': False, 'error': 'Model not loaded'})

    example = create_example_data()
    return jsonify({'success': True, 'data': example})


@app.route('/api/predict', methods=['POST'])
@limiter.limit("30 per minute")
def predict():
    """Make prediction endpoint"""
    if not model_package:
        return jsonify({'success': False, 'error': 'Model not loaded'})

    try:
        input_data = request.json
        if not input_data:
            return jsonify({'success': False, 'error': 'No input data provided'})

        # Validate input data
        is_valid, validation_message = validate_input_data(input_data)
        if not is_valid:
            return jsonify({'success': False, 'error': validation_message})

        result = make_prediction(input_data)

        # Log successful predictions (without sensitive data)
        if result['success']:
            app.logger.info('Successful prediction made')

        return jsonify(result)

    except Exception as e:
        app.logger.error(f'Prediction endpoint error: {str(e)}')
        if app.debug:
            return jsonify({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Internal server error'
            })


@app.route('/api/performance')
@limiter.limit("5 per minute")
def performance():
    """Get model performance metrics"""
    if not model_package:
        return jsonify({'success': False, 'error': 'Model not loaded'})

    if 'performance_metrics' in model_package:
        metrics = model_package['performance_metrics']

        # Calculate averages
        avg_r2 = np.mean([m['Test_R2'] for m in metrics])
        avg_mape = np.mean([m['Test_MAPE'] for m in metrics])

        return jsonify({
            'success': True,
            'metrics': metrics,
            'summary': {
                'avg_r2': avg_r2,
                'avg_mape': avg_mape
            }
        })
    else:
        return jsonify({'success': False, 'error': 'Performance metrics not available'})


@app.route('/api/status')
def status():
    """Get API status"""
    model_loaded = model_package is not None
    return jsonify({
        'status': 'running',
        'model_loaded': model_loaded,
        'version': '2.0_enhanced' if model_loaded else None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_package is not None
    })


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({'error': 'Rate limit exceeded', 'message': str(e.description)}), 429


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Server Error: {error}')
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("🚀 Starting Enhanced TTTF Prediction Web API")
    print("=" * 50)

    # Set up logging
    setup_logging(app)

    # Load model on startup
    if load_model():
        print("✅ Enhanced model loaded successfully!")
        feature_count = len(model_package['feature_names'])
        target_count = len(model_package['target_names'])
        print(f"📊 Features: {feature_count}, Targets: {target_count}")
    else:
        print("❌ Failed to load model!")
        print("   Please ensure model file exists in the models directory.")
        sys.exit(1)  # Exit if model loading fails

    # Get port from environment variable
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    print(f"🌐 Starting Flask server on port {port}...")
    print(f"🔧 Debug mode: {debug_mode}")

    # Use production-ready settings
    app.run(host='0.0.0.0', port=port, debug=debug_mode)