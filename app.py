import sys
import subprocess
import importlib
import os

# ===========================
# تثبيت المكتبات تلقائياً
# ===========================
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

required_libs = [
    "flask",
    "insightface",
    "onnxruntime",
    "opencv-python-headless",
    "numpy"
]

for lib in required_libs:
    try:
        importlib.import_module(lib.replace('-', '_'))
    except ImportError:
        print(f"🔹 تثبيت {lib} تلقائيًا...")
        install(lib)

# ===========================
# استدعاء المكتبات بعد التأكد من التثبيت
# ===========================
from flask import Flask, request, jsonify
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ===========================
# إعداد API
# ===========================
app = Flask(__name__)

# تحميل النموذج الخفيف مرة واحدة عند التشغيل (CPU)
face_app = FaceAnalysis(name="antelope")
face_app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU

@app.route("/gender", methods=["POST"])
def gender():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    faces = face_app.get(img)
    if not faces:
        return jsonify({"error": "No face detected"}), 404

    gender_val = int(faces[0].gender) if faces[0].gender is not None else -1
    if gender_val == -1:
        return jsonify({"error": "Could not determine gender"}), 500

    return jsonify({"gender": "ذكر" if gender_val == 1 else "أنثى"})

# ===========================
# تشغيل الخادم
# ===========================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"🌐 API جاهز على: http://0.0.0.0:{port}/gender")
    app.run(host="0.0.0.0", port=port)
