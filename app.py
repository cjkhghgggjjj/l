import os
import subprocess
import sys
import requests
import io
import numpy as np
from flask import Flask, request, render_template_string
import cv2
import onnxruntime as ort

# ===========================
# تثبيت المكتبات تلقائيًا
# ===========================
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import onnxruntime
except:
    install("onnxruntime")
    import onnxruntime

try:
    import numpy as np
except:
    install("numpy")
    import numpy as np

try:
    from flask import Flask, request, render_template_string
except:
    install("flask")
    from flask import Flask, request, render_template_string

# ===========================
# إعداد Flask
# ===========================
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===========================
# روابط ملفات النماذج ONNX
# ===========================
model_urls = {
    "genderage": "https://classy-douhua-0d9950.netlify.app/genderage.onnx.index.js"
    # يمكنك إضافة نماذج أخرى لاحقًا
}

# ===========================
# تحميل نموذج ONNX من الرابط في الذاكرة
# ===========================
def load_onnx_model(url):
    r = requests.get(url)
    if r.status_code != 200:
        raise RuntimeError(f"فشل تحميل النموذج من {url}")
    # نفترض أن الملف عبارة عن ONNX binary داخل index.js → إذا كان Base64
    content = r.content
    # إذا كان الملف index.js يحوي Base64:
    # استخراج النص بعد const MODEL = "..." 
    text = content.decode(errors="ignore")
    start = text.find('"') + 1
    end = text.rfind('"')
    base64_data = text[start:end]
    model_bytes = io.BytesIO(base64.b64decode(base64_data))
    sess = ort.InferenceSession(model_bytes.read(), providers=['CPUExecutionProvider'])
    return sess

# ===========================
# تهيئة نموذج الجنس فقط
# ===========================
gender_model = load_onnx_model(model_urls["genderage"])

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
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (64, 64))  # حسب نموذج الجنس
            img_input = img_resized.transpose(2,0,1)[np.newaxis,:,:,:].astype(np.float32)

            # تمرير النموذج ONNX
            input_name = gender_model.get_inputs()[0].name
            outputs = gender_model.run(None, {input_name: img_input})
            gender_score = outputs[0][0][0]  # نفترض 0=ذكر،1=أنثى
            gender_result = "ذكر" if gender_score < 0.5 else "أنثى"

    return render_template_string(HTML_PAGE, gender=gender_result, image_url=image_url)

# ===========================
# تشغيل الخادم
# ===========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
