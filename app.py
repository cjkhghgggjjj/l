import os
import subprocess
import sys
import io
import requests
from flask import Flask, request, render_template_string
import cv2
import numpy as np

# ===========================
# تثبيت المكتبات تلقائيًا
# ===========================
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import insightface
except:
    install("insightface")
    import insightface

try:
    import onnxruntime
except:
    install("onnxruntime")
    import onnxruntime

# ===========================
# إعداد Flask
# ===========================
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===========================
# روابط ملفات النماذج مباشرة
# ===========================
model_urls = {
    "scrfd": "https://classy-douhua-0d9950.netlify.app/scrfd_10g_bnkps.onnx.index.js",
    "glintr100": "https://classy-douhua-0d9950.netlify.app/glintr100.onnx.index.js",
    "genderage": "https://classy-douhua-0d9950.netlify.app/genderage.onnx.index.js",
    "2d106det": "https://classy-douhua-0d9950.netlify.app/2d106det.onnx.index.js",
    "1k3d68": "https://classy-douhua-0d9950.netlify.app/1k3d68.onnx.index.js"
}

# ===========================
# تحميل النموذج في الذاكرة
# ===========================
def load_model_from_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        raise RuntimeError(f"فشل تحميل النموذج من {url}")
    # تحويل المحتوى مباشرة إلى bytes
    content_bytes = r.content
    # بعض روابطك index.js → نحتاج استخراج الـ bytes الفعلية
    # نفترض أن الملف يحتوي على JavaScript: const MODEL="BASE64";
    # إذا كان الملف فعليًا ONNX فقط، نستخدم r.content مباشرة
    return content_bytes

# ===========================
# نموذج FaceAnalysis
# ===========================
# استخدام النموذج antelopev2 افتراضي
model = insightface.app.FaceAnalysis(name="antelopev2")
model.prepare(ctx_id=-1)

# ===========================
# HTML صفحة الرفع
# ===========================
HTML_PAGE = """
<!doctype html>
<title>كشف جنس الوجه</title>
<h2>رفع صورة لتحديد الجنس</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=رفع>
</form>
{% if gender %}
<h3>النتيجة: {{ gender }}</h3>
<img src="{{ image_url }}" width="300">
{% endif %}
"""

# ===========================
# رفع الصورة وتحليل الجنس
# ===========================
@app.route("/", methods=["GET", "POST"])
def index():
    gender_result = None
    image_url = None
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            image_url = filepath

            img = cv2.imread(filepath)
            faces = model.get(img)

            if len(faces) == 0:
                gender_result = "🚫 لم يتم اكتشاف أي وجه"
            else:
                face = faces[0]
                gender_result = "ذكر" if face.gender == 1 else "أنثى"

    return render_template_string(HTML_PAGE, gender=gender_result, image_url=image_url)

# ===========================
# تشغيل الخادم
# ===========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
