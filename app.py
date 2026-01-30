import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# --- 1. ページ設定とリッチなデザイン ---
st.set_page_config(page_title="Yoga Pose AI Analyzer", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button { 
        width: 100%; border-radius: 25px; 
        background: linear-gradient(45deg, #00f2fe, #4facfe); 
        color: white; font-weight: bold; border: none; padding: 10px;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3);
    }
    .result-card {
        padding: 25px; border-radius: 15px; 
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 242, 254, 0.5); 
        text-align: center; margin-top: 20px;
    }
    .pose-label { font-size: 0.9rem; color: #00f2fe; text-transform: uppercase; }
    .pose-name { font-size: 3rem; font-weight: 800; margin: 5px 0; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. モデル読み込み（キャッシュ利用） ---
@st.cache_resource
def load_models():
    # MoveNet Thunder (int32入力を期待するモデル)
    movenet = tf.lite.Interpreter(model_path="movenet_thunder.tflite")
    movenet.allocate_tensors()
    # 自作分類器
    classifier = tf.lite.Interpreter(model_path="pose_classifier.tflite")
    classifier.allocate_tensors()
    # ラベル読み込み
    with open("pose_labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return movenet, classifier, labels

movenet, classifier, labels = load_models()

# --- 3. メインUI ---
st.title("🧘 Yoga Pose AI")
st.write("カメラで撮影するか、画像をアップロードしてAI判定を開始してください。")

tab1, tab2 = st.tabs(["📸 カメラで撮影", "📁 ファイル選択"])
img_source = None

with tab1:
    camera_img = st.camera_input("Take a photo")
    if camera_img:
        img_source = camera_img

with tab2:
    uploaded_file = st.file_uploader("画像を選択してください", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        img_source = uploaded_file

# --- 4. 解析実行 ---
if img_source:
    # PILで開いてNumPy配列(RGB)に変換
    img = Image.open(img_source)
    img_array = np.array(img)
    # OpenCV形式(BGR)に変換
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape

    if st.button("ポーズを分析する"):
        with st.spinner("AIが解析しています..."):
            try:
                # --- MoveNet推論 (ここがエラーの急所) ---
                # 1. 256x256にリサイズ
                input_img = cv2.resize(img_bgr, (256, 256))
                # 2. NumPyを使用して確実に int32 型の 1x256x256x3 テンソルを作成
                input_tensor = np.expand_dims(input_img, axis=0).astype(np.uint8)
                
                # 3. テンソルをセット
                input_details = movenet.get_input_details()
                movenet.set_tensor(input_details[0]['index'], input_tensor)
                
                # 4. 実行
                movenet.invoke()
                
                # 5. 結果取得 (17個のキーポイント)
                output_details = movenet.get_output_details()
                keypoints = movenet.get_tensor(output_details[0]['index'])[0, 0, :, :]

                # --- ポーズ分類推論 ---
                # キーポイント(y, x, score)をフラットにしてfloat32に変換
                input_data = keypoints.flatten().astype(np.float32).reshape(1, -1)
                
                c_in = classifier.get_input_details()[0]['index']
                c_out = classifier.get_output_details()[0]['index']
                classifier.set_tensor(c_in, input_data)
                classifier.invoke()
                prediction = classifier.get_tensor(c_out)[0]
                
                idx = np.argmax(prediction)
                conf = prediction[idx]
                pose_name = labels[idx]

                # --- 描画処理 ---
                # 骨格の点（緑色）を描画
                for i in range(17):
                    y, x, s = keypoints[i]
                    if s > 0.3: # 信頼度30%以上の点のみ描画
                        cv2.circle(img_bgr, (int(x*w), int(y*h)), 10, (0, 255, 0), -1)

                # --- 結果表示 ---
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                st.markdown(f"""
                    <div class="result-card">
                        <p class="pose-label">Detected Pose</p>
                        <h2 class="pose-name">{pose_name}</h2>
                        <p style="color: #cbd5e0;">Accuracy: {int(conf*100)}%</p>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"解析中にエラーが発生しました: {e}")
                st.info("画像の形式やモデルのパスが正しいか確認してください。")

st.markdown("---")
st.caption("Powered by TensorFlow Lite & Streamlit")

