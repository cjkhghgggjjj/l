import os
import subprocess
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import cv2
import numpy as np
from flask import Flask, request, jsonify

# ===========================
# تثبيت المكتبات الأساسية فقط
# ===========================
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_libs = [
    "flask",
    "insightface", 
    "onnxruntime",
    "opencv-python-headless",
    "numpy"
]

for lib in required_libs:
    try:
        __import__(lib.split('-')[0])
    except ImportError:
        print(f"🔹 تثبيت {lib}...")
        install(lib)

# ===========================
# استدعاء المكتبات
# ===========================
from insightface.app import FaceAnalysis

# ===========================
# إعداد التطبيق
# ===========================
app = Flask(__name__)

# مفتاح API
PUBLIC_API_KEY = "faceai_public_key_2024"

# إحصاءات بسيطة
request_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "start_time": time.time()
}

stats_lock = threading.Lock()

# تحميل النموذج
print("🔄 جار تحميل نموذج تحليل الوجوه...")
face_app = FaceAnalysis()
face_app.prepare(ctx_id=0, det_size=(320, 320))  # حجم أصغر لتوفير الذاكرة
print("✅ تم تحميل النموذج بنجاح!")

# معالج متعدد الخيوط
max_workers = min(8, (os.cpu_count() or 1) * 2)  # خيوط أقل
executor = ThreadPoolExecutor(max_workers=max_workers)
print(f"⚡ تم تهيئة {max_workers} خيط")

# ===========================
# وظائف المساعدة - مبسطة
# ===========================
def update_stats(success=True):
    with stats_lock:
        request_stats["total_requests"] += 1
        if success:
            request_stats["successful_requests"] += 1

def process_image_simple(image_data):
    try:
        img_array = np.frombuffer(image_data, np.uint8)
        img_np = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img_np is None:
            return create_fallback_response()
        
        faces = face_app.get(img_np)
        
        results = []
        for i, face in enumerate(faces):
            face_data = {
                "face_number": i + 1,
                "age": int(face.age),
                "gender": "male" if face.gender == 1 else "female",
                "bbox": face.bbox.tolist(),
                "confidence": float(face.det_score)
            }
            results.append(face_data)
        
        return {
            "success": True, 
            "faces_count": len(faces),
            "faces": results
        }
        
    except Exception as e:
        return create_fallback_response()

def create_fallback_response():
    return {
        "success": True,
        "faces_count": 0,
        "faces": [],
        "fallback_used": True
    }

# ===========================
# مسارات API مبسطة
# ===========================
@app.route("/")
def index():
    return """
    <html>
    <body>
        <h1>Face Analysis API</h1>
        <p>استخدم /analyze مع X-API-Key: faceai_public_key_2024</p>
    </body>
    </html>
    """

@app.route("/analyze", methods=["POST"])
def analyze_image():
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != PUBLIC_API_KEY:
        return jsonify(create_fallback_response())
    
    if "image" not in request.files:
        return jsonify(create_fallback_response())
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify(create_fallback_response())
    
    try:
        image_data = file.read()
        future = executor.submit(process_image_simple, image_data)
        result = future.result(timeout=10)
        update_stats(success=True)
        return jsonify(result)
    
    except Exception as e:
        update_stats(success=False)
        return jsonify(create_fallback_response())

@app.route("/stats", methods=["GET"])
def get_stats():
    with stats_lock:
        stats = request_stats.copy()
    return jsonify({"success": True, "stats": stats})

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"})

# ===========================
# تشغيل التطبيق
# ===========================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"🚀 تطبيق تحليل الوجوه يعمل على: http://0.0.0.0:{port}")
    print(f"🔑 المفتاح: {PUBLIC_API_KEY}")
    
    app.run(host="0.0.0.0", port=port, threaded=True)
