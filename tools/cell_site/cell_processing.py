from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import tempfile
import json
from werkzeug.utils import secure_filename
import traceback

# Import from your site.py file (NOT your_script_name)
from . import cell_site_processing as site  # This imports the site.py file you created

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Use a fixed directory for uploads instead of temp
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Cell Site Locator API is running!'}), 200

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV, XLSX, XLS allowed'}), 400
        
        # Get parameters from form data
        method = request.form.get('method', 'noml')
        min_samples = int(request.form.get('min_samples', 30))
        bin_size = int(request.form.get('bin_size', 5))
        soft_spacing = request.form.get('soft_spacing', 'false').lower() == 'true'
        use_ta = request.form.get('use_ta', 'false').lower() == 'true'
        make_map = request.form.get('make_map', 'false').lower() == 'true'
        
        print(f"Processing with method: {method}")
        print(f"Parameters: min_samples={min_samples}, bin_size={bin_size}, soft_spacing={soft_spacing}")
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"File saved to: {filepath}")
        
        # Create unique output directory for this request
        import time
        timestamp = str(int(time.time()))
        outdir = os.path.join(app.config['OUTPUT_FOLDER'], f'output_{timestamp}')
        os.makedirs(outdir, exist_ok=True)
        print(f"Output directory: {outdir}")
        
        # Setup logger from site.py
        site.setup_logger(outdir, tag=method)
        
        # Process file based on method
        results = {}
        if method == 'noml':
            print("Running NO-ML method...")
            results = site.run_noml(
                input_path=filepath,
                outdir=outdir,
                min_samples=min_samples,
                bin_size=bin_size,
                soft_spacing=soft_spacing,
                use_ta=use_ta,
                make_map=make_map,
                merge_sites=soft_spacing  # Add this parameter
            )
        else:  # ml method
            print("Running ML method...")
            # For ML, we need either a model or training data
            # This is a simplified example - you might need to adjust based on your needs
            model_path = request.form.get('model_path', None)
            train_path = request.form.get('train_path', None)
            
            results = site.run_ml(
                train_path=train_path,
                model_path=model_path,
                input_path=filepath,
                outdir=outdir,
                min_samples=min_samples,
                bin_size=bin_size,
                soft_spacing=soft_spacing,
                make_map=make_map
            )
        
        # Convert file paths to relative paths for downloading
        relative_results = {}
        for key, path in results.items():
            if path and os.path.exists(path):
                # Store just the filename for downloading
                relative_results[key] = os.path.basename(path)
        
        print(f"Processing complete. Results: {relative_results}")
        
        return jsonify({
            'success': True,
            'results': relative_results,
            'output_dir': os.path.basename(outdir),
            'message': 'File processed successfully'
        }), 200
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<output_dir>/<filename>', methods=['GET'])
def download_file(output_dir, filename):
    """Download result files"""
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], output_dir, filename)
        print(f"Attempting to download: {file_path}")
        
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=filename)
        else:
            return jsonify({'error': f'File not found: {filename}'}), 404
    except Exception as e:
        print(f"Download error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'output_folder': app.config['OUTPUT_FOLDER']
    }), 200

if __name__ == '__main__':
    print("Starting Flask server...")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Output folder: {app.config['OUTPUT_FOLDER']}")
    app.run(debug=True, host='0.0.0.0', port=5000)