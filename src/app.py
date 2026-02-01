import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import random
import os
import streamlit.components.v1 as components

# --- 1. UIスタイル設定 ---
st.set_page_config(page_title="AI Yoga Master - SP1.2", layout="centered")
st.markdown("""
    <style>
    div[data-testid="stCameraInput"] { max-width: 600px; margin: 0 auto; }
    .status-card { 
        text-align: center; padding: 20px; border: 2px solid #00f2fe; 
        border-radius: 15px; background: #1e1e1e; margin: 10px 0; 
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. モデル読み込み & 推論ロジック ---
@st.cache_resource
def load_models():
    # クラウド上でのパス問題を解決するため、絶対パスを取得
    base_path = os.path.dirname(__file__)
    
    movenet_path = os.path.join(base_path, "movenet_thunder.tflite")
    classifier_path = os.path.join(base_path, "pose_classifier.tflite")
    label_path = os.path.join(base_path, "pose_labels.txt")

    try:
        # MoveNetの初期化
        if not os.path.exists(movenet_path):
            raise FileNotFoundError(f"モデルが見つかりません: {movenet_path}")
            
        movenet = tf.lite.Interpreter(model_path=movenet_path)
        movenet.allocate_tensors()
        
        # 分類器の初期化
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"分類器が見つかりません: {classifier_path}")
            
        classifier = tf.lite.Interpreter(model_path=classifier_path)
        classifier.allocate_tensors()
        
        # ラベルの読み込み
        with open(label_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f.readlines()]
            
        return movenet, classifier, labels
        
    except Exception as e:
        # ここでエラーを表示させることで、原因を特定しやすくします
        st.error(f"【重大なエラー】モデルの読み込みに失敗しました: {e}")
        st.info("GitHubの 'src' フォルダ内に .tflite ファイルと .txt ファイルがあるか確認してください。")
        st.stop() # 読み込めない場合はここで処理を止める

# モデルのロード
movenet, classifier, labels = load_models()

def analyze_pose(image):
    orig_w, orig_h = image.size
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # MoveNet推論
    input_img = cv2.resize(img_bgr, (256, 256))
    input_tensor = np.expand_dims(input_img, axis=0).astype(np.uint8)
    
    movenet.set_tensor(movenet.get_input_details()[0]['index'], input_tensor)
    movenet.invoke()
    keypoints = movenet.get_tensor(movenet.get_output_details()[0]['index'])[0, 0, :, :]
    
    # 分類推論
    input_data = keypoints.flatten().astype(np.float32).reshape(1, -1)
    classifier.set_tensor(classifier.get_input_details()[0]['index'], input_data)
    classifier.invoke()
    prediction = classifier.get_tensor(classifier.get_output_details()[0]['index'])[0]
    
    # 描画処理
    draw_img = img_bgr.copy()
    for i in range(17):
        y, x, s = keypoints[i]
        if s > 0.3:
            px, py = int(x * orig_w), int(y * orig_h)
            cv2.circle(draw_img, (px, py), 10, (0, 255, 0), -1)
            
    return labels[np.argmax(prediction)], prediction[np.argmax(prediction)], draw_img

# --- 3. 自動撮影JS ---
def auto_shutter_js():
    components.html(
        """
        <script>
        const parentDoc = window.parent.document;
        const buttons = Array.from(parentDoc.querySelectorAll('button'));
        const takePhotoBtn = buttons.find(b => b.innerText.includes("Take Photo"));
        if (takePhotoBtn) {
            let count = 3;
            takePhotoBtn.innerText = "撮影まで... " + count;
            const timer = setInterval(() => {
                count--;
                if (count <= 0) {
                    clearInterval(timer);
                    takePhotoBtn.click();
                } else {
                    takePhotoBtn.innerText = "撮影まで... " + count;
                }
            }, 1000);
        } else {
            alert("先にカメラを起動してください。");
        }
        </script>
        """,
        height=0,
    )

# --- 4. 状態管理 ---
if 'mode' not in st.session_state: st.session_state.mode = "MAIN"
if 'game_status' not in st.session_state: st.session_state.game_status = "IDLE"
if 'camera_key' not in st.session_state: st.session_state.camera_key = 0

def reset_camera():
    st.session_state.camera_key += 1

# --- 5. 画面遷移 ---
if st.session_state.mode == "MAIN":
    st.title("🧘 AI Yoga Master Pro")
    col1, col2 = st.columns(2)
    if col1.button("🧘 通常トレーニング", use_container_width=True):
        st.session_state.mode = "PRACTICE"
        st.rerun()
    if col2.button("🎮 ポーズ・チャレンジ", use_container_width=True):
        st.session_state.mode = "GAME"
        st.session_state.game_status = "IDLE"
        st.rerun()

elif st.session_state.mode == "PRACTICE":
    st.sidebar.button("⬅️ メニュー", on_click=lambda: st.session_state.update({"mode":"MAIN"}))
    st.title("Practice Mode")
    
    if st.button("🔴 3秒後に自動撮影", use_container_width=True):
        auto_shutter_js()

    img_file = st.camera_input("自由にポーズを判定", key=f"cam_{st.session_state.camera_key}")
    
    if img_file:
        pose, conf, draw_img = analyze_pose(Image.open(img_file))
        st.image(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
        st.markdown(f"<div class='status-card'><h3>判定: {pose}</h3><p>信頼度: {int(conf*100)}%</p></div>", unsafe_allow_html=True)
        if st.button("リセットしてもう一度撮る", use_container_width=True):
            reset_camera()
            st.rerun()

elif st.session_state.mode == "GAME":
    st.sidebar.button("⬅️ 中断", on_click=lambda: st.session_state.update({"mode":"MAIN"}))

    if st.session_state.game_status == "IDLE":
        st.session_state.target_pose = random.choice(labels)
        st.session_state.game_status = "READY"
        st.rerun()

    elif st.session_state.game_status == "READY":
        st.markdown(f"<div class='status-card'><h2>お題: {st.session_state.target_pose}</h2></div>", unsafe_allow_html=True)
        if st.button("カメラを起動してチャレンジ", use_container_width=True):
            st.session_state.game_status = "CHALLENGE"
            st.rerun()

    elif st.session_state.game_status == "CHALLENGE":
        st.markdown(f"### 目標: {st.session_state.target_pose}")
        if st.button("🔴 3秒後に自動撮影を開始", use_container_width=True):
            auto_shutter_js()

        img_file = st.camera_input("カメラを起動してください", key=f"game_cam_{st.session_state.camera_key}")

        if img_file:
            pose, conf, draw_img = analyze_pose(Image.open(img_file))
            st.session_state.result_data = {"img": draw_img, "pose": pose, "conf": conf}
            st.session_state.game_status = "RESULT"
            st.rerun()

    elif st.session_state.game_status == "RESULT":
        res = st.session_state.result_data
        win = res["pose"] == st.session_state.target_pose
        st.title("🎉 正解！" if win else "💀 残念...")
        st.image(cv2.cvtColor(res["img"], cv2.COLOR_BGR2RGB), use_container_width=True)
        
        if st.button("次のお題へ（カメラをリセット）", use_container_width=True):
            reset_camera() 
            st.session_state.game_status = "IDLE"
            st.rerun()
