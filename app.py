import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import tensorflow as tf
import os

# --- 1. ページ全体のデザイン設定 ---
st.set_page_config(page_title="Yoga AI Trainer Pro", layout="wide")

# カスタムCSS: ガラス風デザインとネオンエフェクト
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stApp { color: #ffffff; }
    /* ガラス風の判定カード */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .pose-name {
        font-size: 3rem;
        font-weight: 800;
        color: #00f2fe; /* ネオンブルー */
        text-shadow: 0 0 15px #00f2fe;
        margin: 10px 0;
    }
    .status-badge {
        background: #1a202c;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9rem;
        color: #a0aec0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. モデルの読み込み (キャッシュ機能で高速化) ---
@st.cache_resource
def load_models():
    # MoveNet (骨格検知)
    movenet = tf.lite.Interpreter(model_path="movenet_thunder.tflite")
    movenet.allocate_tensors()
    # 自作分類器
    classifier = tf.lite.Interpreter(model_path="pose_classifier.tflite")
    classifier.allocate_tensors()
    # ラベル
    with open("pose_labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return movenet, classifier, labels

try:
    movenet, classifier, labels = load_models()
except Exception as e:
    st.error(f"モデル読み込みエラー: {e}")
    st.stop()

# --- 3. 映像処理エンジン ---
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.movenet = movenet
        self.classifier = classifier
        self.labels = labels

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # --- MoveNet 推論 ---
        input_img = cv2.resize(img, (256, 256))
        input_tensor = tf.cast(tf.expand_dims(input_img, axis=0), dtype=tf.int32)
        
        m_in = self.movenet.get_input_details()[0]['index']
        m_out = self.movenet.get_output_details()[0]['index']
        self.movenet.set_tensor(m_in, input_tensor.numpy())
        self.movenet.invoke()
        keypoints = self.movenet.get_tensor(m_out)[0, 0, :, :]

        # --- 分類推論 ---
        input_data = keypoints.flatten().astype(np.float32).reshape(1, -1)
        c_in = self.classifier.get_input_details()[0]['index']
        c_out = self.classifier.get_output_details()[0]['index']
        self.classifier.set_tensor(c_in, input_data)
        self.classifier.invoke()
        prediction = self.classifier.get_tensor(c_out)[0]
        
        idx = np.argmax(prediction)
        conf = prediction[idx]
        pose_name = self.labels[idx]

        # --- 描画処理 (ネオン骨格) ---
        # 接続する関節のペア (MoveNet 17点用)
        EDGES = [
            (0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        # 線の描画
        for edge in EDGES:
            p1 = keypoints[edge[0]]
            p2 = keypoints[edge[1]]
            if p1[2] > 0.3 and p2[2] > 0.3:
                cv2.line(img, (int(p1[1]*w), int(p1[0]*h)), (int(p2[1]*w), int(p2[0]*h)), (255, 255, 255), 2)

        # 点の描画
        for i in range(17):
            y, x, s = keypoints[i]
            if s > 0.3:
                cv2.circle(img, (int(x*w), int(y*h)), 5, (0, 242, 254), -1)

        # 判定結果のオーバーレイ
        if conf > 0.6:
            cv2.putText(img, f"MATCH: {int(conf*100)}%", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")

# --- 4. メイン画面レイアウト ---
st.title("🧘 AI Yoga Master")
st.markdown("---")

col_left, col_right = st.columns([3, 1])

with col_left:
    # 接続設定を強化したWebrtc
    webrtc_streamer(
        key="yoga-pose",
        video_processor_factory=PoseProcessor,
        # 複数のSTUNサーバーを指定して接続性をアップ
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]}
            ]
        },
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15}
            },
            "audio": False
        },
        async_processing=True,
    )

with col_right:
    # 右側のステータスパネル
    st.markdown(f"""
        <div class="result-card">
            <span class="status-badge">LIVE ANALYSIS</span>
            <p style="color: #a0aec0; margin-top: 20px;">Current Pose</p>
            <h2 class="pose-name">Detecting...</h2>
            <div style="text-align: left; margin-top: 30px;">
                <p style="font-weight: bold; color: #00f2fe;">Target Poses:</p>
                <ul style="list-style: none; padding-left: 5px; color: #cbd5e0;">
                    {"".join([f"<li>✨ {l}</li>" for l in labels])}
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.info("💡 接続に時間がかかる場合は、ブラウザをリロードするか、回線を確認してください。")
