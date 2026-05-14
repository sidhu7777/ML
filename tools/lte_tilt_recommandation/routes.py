import os
import uuid
from flask import Blueprint, request, jsonify, send_file, current_app
from werkzeug.utils import secure_filename
from .services import RFOptimizationService

rf_optimization_bp = Blueprint("rf_optimization", __name__)
service = RFOptimizationService()

# ==========================================================
# RUN OPTIMIZATION (POST)
# ==========================================================
@rf_optimization_bp.route("/optimize", methods=["POST"])
def run_optimized():
    if request.is_json:
        data = request.get_json() or {}
    else:
        data = request.form.to_dict()
        upload = request.files.get("file") or request.files.get("threshold_file")
        if upload and upload.filename:
            upload_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], "tilt_thresholds")
            os.makedirs(upload_dir, exist_ok=True)
            safe_name = secure_filename(upload.filename) or f"tilt_threshold_{uuid.uuid4().hex}.csv"
            saved_path = os.path.join(upload_dir, f"{uuid.uuid4().hex}_{safe_name}")
            upload.save(saved_path)
            data["threshold_file_path"] = saved_path

    # project_id is the only strictly required field
    required_fields = ["project_id"]

    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"{field} is required"}), 400

    # Submit the job with the full payload
    result = service.submit(data)
    return jsonify(result), 202

# ==========================================================
# CHECK STATUS (GET)
# ==========================================================
@rf_optimization_bp.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    job = service.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job), 200

# ==========================================================
# DOWNLOAD EXCEL FILE (GET)
# ==========================================================
@rf_optimization_bp.route("/download", methods=["GET"])
def download():
    file_path = request.args.get("file")
    if not file_path:
        return jsonify({"error": "file path required"}), 400
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found or expired on server"}), 404
        
    try:
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"Failed to download file: {str(e)}"}), 500
