import os
import sys
import subprocess

# ===========================
# دالة لتثبيت المكتبات تلقائيًا
# ===========================
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# ===========================
# تثبيت واستيراد المكتبات المطلوبة
# ===========================
packages = {
    "requests": "requests",
    "flask": "flask",
    "numpy": "numpy",
    "opencv-python": "cv2",
    "onnxruntime": "onnxruntime",
    "base64": "base64",
    "io": "io"
}

for pkg_name, import_name in packages.items():
    try:
        globals()[import_name] = __import__(import_name)
    except ImportError:
        if pkg_name not in ["io", "base64"]:  # مدمجة في بايثون
            print(f"📦 تثبيت المكتبة: {pkg_name} ...")
            install(pkg_name)
        globals()[import_name] = __import__(import_name)

# ===========================
# استيراد Flask بعد التثبيت
# ===========================
from flask import Flask, request, render_template_string

# ===========================
# إعداد Flask
# ===========================
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===========================
# رابط نموذج الجنس ONNX
# ===========================
model_url = "https://classy-douhua-0d9950.netlify.app/genderage.onnx.index.js"

# ===========================
# تحميل النموذج من الرابط مباشرة في الذاكرة
# ===========================
def load_onnx_model(url):
    r = requests.get(url)
    if r.status_code != 200:
        raise RuntimeError(f"فشل تحميل النموذج من {url}")
    
    content = r.content
    text = content.decode(errors="ignore")
    start = text.find('"') + 1
    end = text.rfind('"')
    base64_data = text[start:end]
    model_bytes = io.BytesIO(base64.b64decode(base64_data))
    
    sess = onnxruntime.InferenceSession(model_bytes.read(), providers=['CPUExecutionProvider'])
    return sess

gender_model = load_onnx_model(model_url)

# ===========================
# صفحة HTML للرفع
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
            if img is None:
                gender_result = "🚫 خطأ في قراءة الصورة"
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (64, 64))  # حسب نموذج الجنس
                img_input = img_resized.transpose(2,0,1)[np.newaxis,:,:,:].astype(np.float32)

                input_name = gender_model.get_inputs()[0].name
                outputs = gender_model.run(None, {input_name: img_input})
                gender_score = outputs[0][0][0]
                gender_result = "ذكر" if gender_score < 0.5 else "أنثى"

    return render_template_string(HTML_PAGE, gender=gender_result, image_url=image_url)

# ===========================
# تشغيل الخادم
# ===========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
