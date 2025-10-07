import sys
import subprocess

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
        __import__(lib.split('-')[0])
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

# تحميل النموذج الخفيف مرة واحدة عند التشغيل
face_app = FaceAnalysis(name="antelope")  # أصغر نموذج متاح
face_app.prepare(ctx_id=0, det_size=(640, 640))

@app.route("/gender", methods=["POST"])
def gender():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    # قراءة الصورة مباشرة من الذاكرة
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    faces = face_app.get(img)
    if not faces:
        return jsonify({"error": "No face detected"}), 404

    gender = int(faces[0].gender)  # 1=ذكر, 0=أنثى
    return jsonify({"gender": "ذكر" if gender == 1 else "أنثى"})

# ===========================
# تشغيل الخادم
# ===========================
if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 5000))
    print(f"🌐 API جاهز على: http://0.0.0.0:{port}/gender")
    app.run(host="0.0.0.0", port=port)
