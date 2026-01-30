import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import tensorflow as tf
import os

# ページ基本設定
st.set_page_config(page_title="Yoga Pose AI", layout="wide")

st.title("🧘 AI Pose Classifier")
st.markdown("MoveNet Thunder + TFLite Classifier によるリアルタイム判定")

# --- ファイル存在チェック ---
required_files = ["movenet_thunder.tflite", "pose_classifier.tflite", "pose_labels.txt"]
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error(f"以下のファイルが見つかりません: {', '.join(missing_files)}")
    st.stop()

# --- モデル読み込み (キャッシュ利用) ---
@st.cache_resource
def load_resource():
    # MoveNet
    movenet = tf.lite.Interpreter(model_path="movenet_thunder.tflite")
    movenet.allocate_tensors()
    
    # 分類器
    classifier = tf.lite.Interpreter(model_path="pose_classifier.tflite")
    classifier.allocate_tensors()
    
    # ラベル
    with open("pose_labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
        
    return movenet, classifier, labels

movenet, classifier, labels = load_resource()

# --- 映像解析クラス ---
class PoseAnalyzer(VideoProcessorBase):
    def __init__(self):
        self.movenet = movenet
        self.classifier = classifier
        self.labels = labels

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # 1. MoveNet 前処理 (256x256, int32)
        input_size = 256
        resized_img = cv2.resize(img, (input_size, input_size))
        input_tensor = tf.cast(tf.expand_dims(resized_img, axis=0), dtype=tf.int32)

        # 2. キーポイント抽出
        m_input_details = self.movenet.get_input_details()
        m_output_details = self.movenet.get_output_details()
        self.movenet.set_tensor(m_input_details[0]['index'], input_tensor.numpy())
        self.movenet.invoke()
        # [1, 1, 17, 3] -> [17, 3] (y, x, score)
        keypoints = self.movenet.get_tensor(m_output_details[0]['index'])[0, 0, :, :]

        # 3. ポーズ分類
        # Classifierの期待値 [1, 51] に変換
        input_data = keypoints.flatten().astype(np.float32).reshape(1, -1)
        
        c_input_details = self.classifier.get_input_details()
        c_output_details = self.classifier.get_output_details()
        self.classifier.set_tensor(c_input_details[0]['index'], input_data)
        self.classifier.invoke()
        prediction = self.classifier.get_tensor(c_output_details[0]['index'])[0]
        
        idx = np.argmax(prediction)
        score = prediction[idx]
        pose_name = self.labels[idx]

        # 4. 描画処理
        # スケルトン（点）の描画
        for i in range(17):
            ky, kx, ks = keypoints[i]
            if ks > 0.3:
                cv2.circle(img, (int(kx * w), int(ky * h)), 6, (0, 255, 0), -1)

        # 判定結果の表示
        if score > 0.5: # 信頼度50%以上で表示
            display_text = f"{pose_name} ({int(score*100)}%)"
            cv2.putText(img, display_text, (20, 60), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(img, display_text, (20, 60), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 165, 255), 1)

        return frame.from_ndarray(img, format="bgr24")

# --- UI レイアウト ---
col_main, col_info = st.columns([3, 1])

with col_main:
    webrtc_streamer(
        key="pose-classification",
        video_processor_factory=PoseAnalyzer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

with col_info:
    st.write("### 判定対象")
    for l in labels:
        st.write(f"✅ {l}")
    
    st.divider()
    st.write("### 使い方")
    st.caption("1. カメラを許可\n2. 全身が映る位置へ移動\n3. ポーズをとるとAIが判定します")