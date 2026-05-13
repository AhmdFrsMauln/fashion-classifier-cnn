import streamlit as st
import numpy as np
from PIL import Image
import json
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fashion Item Classifier",
    page_icon="👗",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

:root {
    --bg:       #0f0f0f;
    --surface:  #1a1a1a;
    --border:   #2e2e2e;
    --accent:   #e8ff5a;
    --text:     #f0f0f0;
    --muted:    #888;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Header */
.app-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.app-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    letter-spacing: -1px;
    color: var(--accent);
    margin: 0;
}
.app-header p {
    color: var(--muted);
    font-size: 0.9rem;
    margin-top: 0.4rem;
}

/* Upload zone */
.upload-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}

/* Result card */
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin-top: 1.5rem;
}
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}
.result-class {
    font-size: 2.2rem;
    font-weight: 600;
    color: var(--accent);
    margin: 0;
}
.result-conf {
    font-size: 0.95rem;
    color: var(--muted);
    margin-top: 0.3rem;
}

/* Progress bar override */
.stProgress > div > div > div {
    background-color: var(--accent) !important;
}

/* Bar chart label */
.bar-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
    font-size: 0.82rem;
}
.bar-name  { width: 110px; color: var(--text); text-align: right; }
.bar-track { flex: 1; background: var(--border); border-radius: 4px; height: 10px; }
.bar-fill  { height: 10px; border-radius: 4px; background: var(--accent); }
.bar-pct   { width: 46px; color: var(--muted); font-family: 'Space Mono', monospace; font-size: 0.75rem; }

/* Info box */
.info-box {
    background: var(--surface);
    border-left: 3px solid var(--accent);
    padding: 1rem 1.2rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 2rem;
}

/* Divider */
hr { border-color: var(--border); margin: 2rem 0; }

/* Button */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.55rem 1.4rem !important;
    font-weight: 700 !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
</style>
""", unsafe_allow_html=True)

# ── Emoji & info per kelas ────────────────────────────────────────────────────
CLASS_INFO = {
    "T-shirt/top": ("👕", "Pakaian atasan kasual berbentuk T atau mirip."),
    "Trouser":     ("👖", "Celana panjang formal maupun kasual."),
    "Pullover":    ("🧥", "Sweater atau atasan pullover tanpa kancing."),
    "Dress":       ("👗", "Gaun atau dress wanita satu bagian."),
    "Coat":        ("🧣", "Mantel tebal untuk cuaca dingin."),
    "Sandal":      ("👡", "Alas kaki terbuka, biasa dipakai di cuaca panas."),
    "Shirt":       ("👔", "Kemeja berkerah, biasanya berkancing."),
    "Sneaker":     ("👟", "Sepatu olahraga atau kasual."),
    "Bag":         ("👜", "Tas tangan atau aksesori serupa."),
    "Ankle boot":  ("👢", "Sepatu bot pendek setinggi pergelangan kaki."),
}

CLASS_NAMES = list(CLASS_INFO.keys())

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import tensorflow as tf
    model_path = "fashion_cnn_model.keras"
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

# ── Preprocess gambar ─────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("L")                  # grayscale
    img = img.resize((28, 28), Image.LANCZOS) # resize ke 28x28
    arr = np.array(img, dtype=np.float32) / 255.0
    # Fashion-MNIST: background gelap, objek terang → invert jika perlu
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    return arr.reshape(1, 28, 28, 1)

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>👗 FASHION CLASSIFIER</h1>
  <p>Deep Computer Vision · CNN · Fashion-MNIST · UAS AIB02</p>
</div>
""", unsafe_allow_html=True)

model = load_model()

if model is None:
    st.error(
        "⚠️ File model `fashion_cnn_model.keras` tidak ditemukan.\n\n"
        "Jalankan dulu notebook `Project_UAS_AhmadFarisMaulana.ipynb` "
        "untuk melatih dan menyimpan model, lalu letakkan file `.keras` "
        "di folder yang sama dengan `app.py` ini."
    )
    st.stop()

# ── Upload gambar ─────────────────────────────────────────────────────────────
st.markdown('<p class="upload-label">Upload gambar item fashion</p>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    label="",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Upload foto item fashion (baju, celana, sepatu, tas, dll.)"
)

# ── Demo dengan gambar dari dataset ──────────────────────────────────────────
use_demo = st.checkbox("Atau gunakan contoh gambar dari dataset Fashion-MNIST")

if use_demo:
    import tensorflow as tf
    (_, _), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    idx = st.slider("Pilih indeks gambar test (0–9999)", 0, 9999, 42)
    demo_arr = X_test[idx].astype(np.float32) / 255.0
    demo_img = Image.fromarray((demo_arr * 255).astype(np.uint8), mode="L")
    true_label = CLASS_NAMES[y_test[idx]]

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(demo_img.resize((140, 140), Image.NEAREST),
                 caption=f"Gambar test #{idx}", use_column_width=False)

    input_arr = demo_arr.reshape(1, 28, 28, 1)

    with st.spinner("Mengklasifikasi..."):
        proba = model.predict(input_arr, verbose=0)[0]
    pred_idx   = int(np.argmax(proba))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(proba[pred_idx])
    emoji, desc = CLASS_INFO[pred_class]

    match = "✅" if pred_class == true_label else "❌"
    st.markdown(f"""
    <div class="result-card">
      <div class="result-label">Hasil Prediksi</div>
      <p class="result-class">{emoji} {pred_class}</p>
      <p class="result-conf">Confidence: <strong>{confidence*100:.1f}%</strong>
         &nbsp;·&nbsp; Label sebenarnya: <strong>{true_label}</strong> {match}</p>
      <p style="color:#888;font-size:0.85rem;margin-top:0.6rem;">{desc}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Distribusi probabilitas semua kelas:**")
    for i, (name, p) in enumerate(zip(CLASS_NAMES, proba)):
        pct  = p * 100
        fill = int(pct)
        bold = "font-weight:700;color:var(--accent)" if i == pred_idx else ""
        st.markdown(f"""
        <div class="bar-row">
          <span class="bar-name" style="{bold}">{name}</span>
          <div class="bar-track">
            <div class="bar-fill" style="width:{fill}%"></div>
          </div>
          <span class="bar-pct">{pct:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

elif uploaded is not None:
    image = Image.open(uploaded)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Gambar yang diupload", use_column_width=True)

    with st.spinner("Memproses gambar..."):
        input_arr = preprocess(image)
        proba     = model.predict(input_arr, verbose=0)[0]

    pred_idx   = int(np.argmax(proba))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(proba[pred_idx])
    emoji, desc = CLASS_INFO[pred_class]

    st.markdown(f"""
    <div class="result-card">
      <div class="result-label">Hasil Prediksi</div>
      <p class="result-class">{emoji} {pred_class}</p>
      <p class="result-conf">Confidence: <strong>{confidence*100:.1f}%</strong></p>
      <p style="color:#888;font-size:0.85rem;margin-top:0.6rem;">{desc}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Distribusi probabilitas semua kelas:**")
    for i, (name, p) in enumerate(zip(CLASS_NAMES, proba)):
        pct  = p * 100
        fill = int(pct)
        bold = "font-weight:700;color:var(--accent)" if i == pred_idx else ""
        st.markdown(f"""
        <div class="bar-row">
          <span class="bar-name" style="{bold}">{name}</span>
          <div class="bar-track">
            <div class="bar-fill" style="width:{fill}%"></div>
          </div>
          <span class="bar-pct">{pct:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="info-box">
      📌 <strong>Cara penggunaan:</strong><br>
      Upload foto item fashion (baju, celana, sepatu, tas, dll.) atau
      centang opsi demo untuk mencoba gambar langsung dari dataset Fashion-MNIST.
      Model CNN akan memprediksi kategori item beserta probabilitasnya.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center;color:#444;font-size:0.78rem;font-family:'Space Mono',monospace;">
Ahmad Faris Maulana &nbsp;·&nbsp; UAS AIB02 Pemrograman Kecerdasan Buatan<br>
Universitas Bunda Mulia &nbsp;·&nbsp; Fashion-MNIST CNN Classifier
</p>
""", unsafe_allow_html=True)
