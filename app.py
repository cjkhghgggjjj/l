# ===========================
# تثبيت المكتبات تلقائيًا قبل الاستدعاء
# ===========================
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_libs = [
    "flask",
    "onnxruntime",
    "opencv-python-headless",
    "numpy",
    "requests"
]

for lib in required_libs:
    try:
        __import__(lib.split('-')[0])
    except ImportError:
        print(f"🔹 تثبيت {lib}...")
        install(lib)

# ===========================
# استدعاء المكتبات بعد تثبيتها
# ===========================
import requests
import io
import cv2
import numpy as np
from flask import Flask, render_template_string, request, send_file
import onnxruntime as ort
import os
import traceback

# ===========================
# روابط النماذج على Hugging Face
# ===========================
DET_URL = "https://huggingface.co/vkhghjjhcc/mkk/resolve/main/det_500m.onnx"
GENDER_URL = "https://huggingface.co/vkhghjjhcc/mkk/resolve/main/w600k_mbf.onnx"

# ===========================
# تحميل النماذج مباشرة إلى الذاكرة
# ===========================
def load_model_from_url(url):
    print(f"⬇️ تحميل النموذج من: {url}")
    resp = requests.get(url)
    resp.raise_for_status()
    print(f"✅ تم تحميل {len(resp.content)//1024} KB من النموذج")
    return io.BytesIO(resp.content)

det_model_bytes = load_model_from_url(DET_URL)
gender_model_bytes = load_model_from_url(GENDER_URL)

# ===========================
# إنشاء جلسات ONNX مباشرة من الذاكرة
# ===========================
det_sess = ort.InferenceSession(det_model_bytes.getvalue(), providers=['CPUExecutionProvider'])
gender_sess = ort.InferenceSession(gender_model_bytes.getvalue(), providers=['CPUExecutionProvider'])
print("✅ تم تهيئة جلسات ONNX من الذاكرة بدون حفظ أي ملفات")

# ===========================
# إعداد Flask
# ===========================
app = Flask(__name__)

# ===========================
# صفحة HTML
# ===========================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ar">
<head>
<meta charset="UTF-8">
<title>تحليل الجنس - Face AI</title>
<style>
body {font-family: Arial; text-align:center; background:#f5f5f5;}
h2 {color:#333;}
form {margin:30px auto; padding:20px; background:white; border-radius:15px; width:350px; box-shadow:0 0 10px #ccc;}
input[type=file]{margin:10px;}
img {margin-top:20px; width:250px; border-radius:10px;}
.info {background:#fff; display:inline-block; margin-top:20px; padding:15px; border-radius:10px; box-shadow:0 0 5px #aaa;}
.error {background:#ffe6e6; color:#d00; padding:15px; border-radius:10px;}
.success {background:#e6ffe6; color:#060; padding:10px; border-radius:10px;}
.male {color: blue; font-weight: bold;}
.female {color: deeppink; font-weight: bold;}
</style>
</head>
<body>
<div class="success">
<h2>🧠 نظام تحليل الجنس</h2>
<p>باستخدام ONNX و Hugging Face (نماذج في الذاكرة)</p>
</div>

<form method="POST" enctype="multipart/form-data">
<input type="file" name="image" accept="image/*" required>
<br><br>
<button type="submit">🔍 تحليل الصورة</button>
</form>

{% if error %}
<div class="error">
<h3>⚠️ خطأ:</h3>
<p>{{ error }}</p>
</div>
{% endif %}

{% if result %}
<div class="info">
<h3>👤 النتيجة:</h3>
<p class="{{ 'male' if result.gender == 1 else 'female' }}">
🚹🚺 الجنس: <strong>{{ 'ذكر' if result.gender == 1 else 'أنثى' }}</strong>
</p>
<p>👥 عدد الوجوه المكتشفة: <strong>{{ result.faces }}</strong></p>
</div>
<img src="{{ image_url }}" alt="الصورة المحللة">
{% endif %}
</body>
</html>
"""

# ===========================
# دوال مساعدة لكشف الوجوه وتحليل الجنس
# ===========================
def detect_faces(img):
    """كشف الوجه باستخدام النموذج"""
    img_resized = cv2.resize(img, (640, 640))
    img_input = np.expand_dims(img_resized.transpose(2,0,1).astype(np.float32), axis=0)
    outputs = det_sess.run(None, {det_sess.get_inputs()[0].name: img_input})
    faces = outputs[0]
    if len(faces) == 0:
        return []
    return faces

def predict_gender(face_crop):
    """توقع الجنس من الوجه المقتطع"""
    img_input = np.expand_dims(face_crop.transpose(2,0,1).astype(np.float32), axis=0)
    outputs = gender_sess.run(None, {gender_sess.get_inputs()[0].name: img_input})
    gender = int(np.argmax(outputs[0], axis=1)[0])
    return gender  # 1=ذكر, 0=أنثى

# ===========================
# مسار Flask الرئيسي
# ===========================
@app.route("/", methods=["GET","POST"])
def index():
    try:
        if request.method == "POST":
            if "image" not in request.files:
                return render_template_string(HTML_PAGE, error="❌ لم يتم إرسال أي صورة")

            file = request.files["image"]
            img_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if img is None:
                return render_template_string(HTML_PAGE, error="❌ تعذر قراءة الصورة")

            faces = detect_faces(img)
            if len(faces) == 0:
                return render_template_string(HTML_PAGE, error="🚫 لم يتم العثور على أي وجه")

            results = []
            for face_data in faces:
                if len(face_data) < 4:
                    continue  # تجاهل الوجوه غير المكتملة
                x1, y1, x2, y2 = [int(v) for v in face_data[:4]]
                h, w, _ = img.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w-1, x2), min(h-1, y2)
                face_crop = img[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                gender = predict_gender(face_crop)
                results.append(gender)

            if len(results) == 0:
                return render_template_string(HTML_PAGE, error="🚫 لم يتم تحليل أي وجه صالح")

            # حفظ الصورة مؤقتًا للعرض
            cv2.imwrite("uploaded.jpg", img)
            result = {"gender": results[0], "faces": len(results)}
            return render_template_string(HTML_PAGE, result=result, image_url="/image")

        return render_template_string(HTML_PAGE)

    except Exception as e:
        print(traceback.format_exc())
        return render_template_string(HTML_PAGE, error=f"حدث خطأ: {str(e)}")

@app.route("/image")
def serve_image():
    try:
        return send_file("uploaded.jpg", mimetype="image/jpeg")
    except:
        return "الصورة غير متوفرة", 404

# ===========================
# تشغيل Flask
# ===========================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"🚀 التطبيق يعمل على http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
