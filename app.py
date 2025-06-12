from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import traceback
import os
import sys

app = Flask(__name__)
CORS(app)

# Global variable to store model
model_package = None


def load_model():
    global model_package
    try:
        # Get the absolute path of the current script
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Print debugging information
        print(f"🔍 Script directory: {base_dir}")
        print(f"🔍 Current working directory: {os.getcwd()}")
        print(f"🔍 Python path: {sys.path}")

        # List all files in the directory
        try:
            files = os.listdir(base_dir)
            print(f"🔍 Files in script directory: {files}")
            pkl_files = [f for f in files if f.endswith('.pkl')]
            print(f"🔍 PKL files found: {pkl_files}")
        except Exception as e:
            print(f"❌ Error listing files: {e}")

        # Try multiple possible model file names and locations
        possible_paths = [
            os.path.join(base_dir, 'tttf_xgb_model_enhanced.pkl'),
            os.path.join(base_dir, 'tttf_xgb_model_enhanced_backup.pkl'),
            os.path.join(os.getcwd(), 'tttf_xgb_model_enhanced.pkl'),
            os.path.join(os.getcwd(), 'tttf_xgb_model_enhanced_backup.pkl'),
            'tttf_xgb_model_enhanced.pkl',  # Relative path
            'tttf_xgb_model_enhanced_backup.pkl'  # Relative path
        ]

        for i, model_path in enumerate(possible_paths):
            try:
                print(f"📦 Attempt {i + 1}: Trying model path: {model_path}")
                print(f"📦 Path exists: {os.path.exists(model_path)}")

                if os.path.exists(model_path):
                    print(f"📦 File size: {os.path.getsize(model_path)} bytes")
                    model_package = joblib.load(model_path)
                    print("✅ Model loaded successfully!")

                    # Validate model package
                    if validate_model_package():
                        return True
                    else:
                        print("❌ Model validation failed, trying next path...")
                        model_package = None
                        continue
                else:
                    print(f"❌ File does not exist: {model_path}")

            except FileNotFoundError:
                print(f"❌ File not found: {model_path}")
                continue
            except Exception as e:
                print(f"❌ Error loading model from {model_path}: {e}")
                continue

        print("❌ All model loading attempts failed")
        return False

    except Exception as e:
        print(f"🔥 Unexpected error in load_model: {e}")
        print(f"🔥 Traceback: {traceback.format_exc()}")
        return False


def validate_model_package():
    """Validate that the loaded model package has required components"""
    if not model_package:
        print("❌ Model package is None")
        return False

    required_keys = ['model', 'feature_names', 'target_names']
    missing_keys = [key for key in required_keys if key not in model_package]

    if missing_keys:
        print(f"❌ Model package missing required keys: {missing_keys}")
        print(f"🔍 Available keys: {list(model_package.keys())}")
        return False

    print(f"✅ Model package validated with keys: {list(model_package.keys())}")
    return True


def get_feature_info():
    """Get feature information for the frontend"""
    if not model_package:
        print("❌ Model package not available for feature info")
        return None

    try:
        feature_names = model_package.get('feature_names', [])
        label_encoders = model_package.get('label_encoders', {})

        print(f"🔍 Getting feature info for {len(feature_names)} features")

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
                    guidance = "Voltage reading"
                elif feature == 'rotate':
                    guidance = "Rotation speed"
                elif feature == 'pressure':
                    guidance = "Pressure reading"
                elif feature == 'vibration':
                    guidance = "Vibration level"
                elif feature == 'age':
                    guidance = "Machine age in years"
                elif feature == 'error_count':
                    guidance = "Recent error count"
                elif 'days_since' in feature:
                    component = feature.split('_')[2] if len(feature.split('_')) > 2 else 'component'
                    guidance = f"Days since {component} maintenance"

                features.append({
                    'name': feature,
                    'type': 'numerical',
                    'guidance': guidance
                })

        print(f"✅ Feature info generated for {len(features)} features")
        return features

    except Exception as e:
        print(f"❌ Error getting feature info: {e}")
        return None


def create_example_data():
    """Create example data for quick testing"""
    if not model_package:
        print("❌ Model package not available for example data")
        return None

    try:
        feature_names = model_package.get('feature_names', [])
        label_encoders = model_package.get('label_encoders', {})

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

        print(f"✅ Example data created with {len(example_data)} features")
        return example_data

    except Exception as e:
        print(f"❌ Error creating example data: {e}")
        return None


def make_prediction(input_data):
    """Make prediction using the loaded model"""
    try:
        print(f"🔍 Making prediction with input: {list(input_data.keys())}")

        # Extract model components
        model = model_package['model']
        scaler = model_package.get('scaler')
        imputer = model_package.get('imputer')
        label_encoders = model_package.get('label_encoders', {})
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
        if numerical_features and imputer is not None:
            # Handle missing values and convert to float
            for feat in numerical_features:
                value = X_sample[feat].iloc[0]
                if pd.isna(value) or value == '' or str(value).lower() in ['nan', 'null', 'none']:
                    X_sample[feat] = np.nan
                else:
                    X_sample[feat] = float(value)

            X_sample[numerical_features] = imputer.transform(X_sample[numerical_features])

        # Scale features
        if scaler is not None:
            X_scaled = scaler.transform(X_sample)
        else:
            X_scaled = X_sample.values

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

        print(f"✅ Prediction completed successfully")
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
        print(f"❌ Error making prediction: {e}")
        print(f"🔍 Prediction error traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/model-info')
def model_info():
    """Get model information"""
    print("🔍 Model info endpoint called")

    if not model_package:
        print("❌ Model not loaded for model-info endpoint")
        return jsonify({'success': False, 'error': 'Model not loaded'})

    try:
        features = get_feature_info()
        target_names = model_package.get('target_names', [])
        excluded_features = model_package.get('excluded_features', [])

        print(f"✅ Model info generated: {len(features)} features, {len(target_names)} targets")

        return jsonify({
            'success': True,
            'features': features,
            'targets': target_names,
            'excluded_features': excluded_features,
            'total_features': len(features) if features else 0
        })
    except Exception as e:
        print(f"❌ Error in model-info endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/example-data')
def example_data():
    """Get example data for testing"""
    print("🔍 Example data endpoint called")

    if not model_package:
        print("❌ Model not loaded for example-data endpoint")
        return jsonify({'success': False, 'error': 'Model not loaded'})

    try:
        example = create_example_data()
        if example:
            print(f"✅ Example data generated with {len(example)} features")
            return jsonify({'success': True, 'data': example})
        else:
            return jsonify({'success': False, 'error': 'Failed to generate example data'})
    except Exception as e:
        print(f"❌ Error in example-data endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction endpoint"""
    print("🔍 Predict endpoint called")

    if not model_package:
        print("❌ Model not loaded for predict endpoint")
        return jsonify({'success': False, 'error': 'Model not loaded'})

    try:
        input_data = request.json
        if not input_data:
            print("❌ No input data provided")
            return jsonify({'success': False, 'error': 'No input data provided'})

        result = make_prediction(input_data)
        return jsonify(result)

    except Exception as e:
        print(f"❌ Error in predict endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


@app.route('/api/performance')
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
    """Check application status"""
    model_loaded = model_package is not None
    status_info = {
        'status': 'running',
        'model_loaded': model_loaded,
        'version': '2.0_enhanced' if model_loaded else None
    }

    if model_loaded:
        status_info['feature_count'] = len(model_package.get('feature_names', []))
        status_info['target_count'] = len(model_package.get('target_names', []))

    print(f"🔍 Status check: {status_info}")
    return jsonify(status_info)


@app.route('/api/debug')
def debug():
    """Debug endpoint to check file system and model status"""
    debug_info = {
        'model_loaded': model_package is not None,
        'current_directory': os.getcwd(),
        'script_directory': os.path.dirname(os.path.abspath(__file__)),
        'python_version': sys.version,
        'environment_variables': dict(os.environ)
    }

    try:
        # List files in current directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        files = os.listdir(script_dir)
        debug_info['files_in_script_directory'] = files
        debug_info['pkl_files'] = [f for f in files if f.endswith('.pkl')]

        # Also check current working directory
        cwd_files = os.listdir(os.getcwd())
        debug_info['files_in_cwd'] = cwd_files
        debug_info['pkl_files_in_cwd'] = [f for f in cwd_files if f.endswith('.pkl')]

    except Exception as e:
        debug_info['directory_error'] = str(e)

    if model_package:
        debug_info['model_keys'] = list(model_package.keys())
        debug_info['feature_names'] = model_package.get('feature_names', [])
        debug_info['target_names'] = model_package.get('target_names', [])

    return jsonify(debug_info)


# Load model when the module is imported (for production servers)
print("🚀 Initializing Enhanced TTTF Prediction Web API")
print("=" * 50)

# Try to load model immediately
if load_model():
    print("✅ Enhanced model loaded successfully during initialization!")
    if model_package:
        feature_count = len(model_package.get('feature_names', []))
        target_count = len(model_package.get('target_names', []))
        print(f"📊 Features: {feature_count}, Targets: {target_count}")
else:
    print("❌ Failed to load model during initialization!")
    print("   Check the /api/debug endpoint for more information.")

if __name__ == '__main__':
    print("🌐 Starting Flask development server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
else:
    print("🌐 Running in production mode (WSGI server)")