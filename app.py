import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# --- 1. ページ設定とリッチなCSS ---
st.set_page_config(page_title="Yoga Pose AI", layout="centered")

st.markdown("""
    <style>
    .main { background: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 20px; background: #00f2fe; color: black; font-weight: bold; }
    .result-card {
        padding: 20px; border-radius: 15px; 
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid #00f2fe; text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. モデル読み込み ---
@st.cache_resource
def load_models():
    movenet = tf.lite.Interpreter(model_path="movenet_thunder.tflite")
    movenet.allocate_tensors()
    classifier = tf.lite.Interpreter(model_path="pose_classifier.tflite")
    classifier.allocate_tensors()
    with open("pose_labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return movenet, classifier, labels

movenet, classifier, labels = load_models()

# --- 3. UI部分 ---
st.title("🧘 Yoga Pose Analyzer Pro")
st.write("写真を撮るか、お手持ちの画像をアップロードしてください")

# タブに分けてスッキリさせる
tab1, tab2 = st.tabs(["📸 カメラで撮影", "📁 ファイルを選択"])

img_source = None

with tab1:
    camera_img = st.camera_input("ポーズを撮る")
    if camera_img:
        img_source = camera_img

with tab2:
    uploaded_file = st.file_uploader("ヨガの写真をアップロード", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        img_source = uploaded_file

# --- 4. 推論ロジック ---
if img_source:
    # 画像を展開
    img = Image.open(img_source)
    img_array = np.array(img)
    # RGB -> BGR (OpenCV用)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape

    if st.button("AI判定を開始！"):
        with st.spinner("AIがあなたのポーズを分析中..."):
            # MoveNet 推論
            input_img = cv2.resize(img_bgr, (256, 256))
            input_tensor = tf.cast(tf.expand_dims(input_img, axis=0), dtype=tf.int32)
            
            m_in = movenet.get_input_details()[0]['index']
            m_out = movenet.get_output_details()[0]['index']
            movenet.set_tensor(m_in, input_tensor.numpy())
            movenet.invoke()
            keypoints = movenet.get_tensor(m_out)[0, 0, :, :]

            # 分類推論
            input_data = keypoints.flatten().astype(np.float32).reshape(1, -1)
            c_in = classifier.get_input_details()[0]['index']
            c_out = classifier.get_output_details()[0]['index']
            classifier.set_tensor(c_in, input_data)
            classifier.invoke()
            prediction = classifier.get_tensor(c_out)[0]
            
            idx = np.argmax(prediction)
            conf = prediction[idx]
            pose_name = labels[idx]

            # 骨格描画
            for i in range(17):
                y, x, s = keypoints[i]
                if s > 0.3:
                    cv2.circle(img_bgr, (int(x*w), int(y*h)), 10, (0, 255, 0), -1)

            # 結果表示
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            st.markdown(f"""
                <div class="result-card">
                    <p style="color: #00f2fe; font-size: 1.2rem;">分析結果</p>
                    <h2 style="font-size: 3rem;">{pose_name}</h2>
                    <p>AIの確信度: {int(conf*100)}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            if conf < 0.5:
                st.warning("少し判定が不安定です。もう少し全身が写るように撮ってみてください！")
