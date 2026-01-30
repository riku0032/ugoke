import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import random
import os

# --- 1. 外部ファイル注入関数 ---
def inject_assets():
    with open("style.css", "r") as f:
        css = f.read()
    with open("script.js", "r") as f:
        js = f.read()
    # 親ウィンドウに直接届くようにmarkdownで注入
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    return js

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

# --- 3. 状態管理 ---
if 'mode' not in st.session_state: st.session_state.mode = "MAIN"
if 'game_status' not in st.session_state: st.session_state.game_status = "IDLE"
if 'target_pose' not in st.session_state: st.session_state.target_pose = None

def analyze_pose(image):
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    input_img = cv2.resize(img_bgr, (256, 256))
    input_tensor = np.expand_dims(input_img, axis=0).astype(np.uint8)
    movenet.set_tensor(movenet.get_input_details()[0]['index'], input_tensor)
    movenet.invoke()
    keypoints = movenet.get_tensor(movenet.get_output_details()[0]['index'])[0, 0, :, :]
    input_data = keypoints.flatten().astype(np.float32).reshape(1, -1)
    classifier.set_tensor(classifier.get_input_details()[0]['index'], input_data)
    classifier.invoke()
    prediction = classifier.get_tensor(classifier.get_output_details()[0]['index'])[0]
    return labels[np.argmax(prediction)], prediction[np.argmax(prediction)], keypoints, img_bgr

# --- 4. メイン画面 ---
js_content = inject_assets()

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

# --- 5. 通常モード ---
elif st.session_state.mode == "PRACTICE":
    if st.sidebar.button("⬅️ メニューに戻る"):
        st.session_state.mode = "MAIN"
        st.rerun()
    st.title("Practice Mode")
    img_file = st.camera_input("ポーズを撮影")
    if img_file:
        pose, conf, keypoints, img_bgr = analyze_pose(Image.open(img_file))
        for i in range(17):
            y, x, s = keypoints[i]
            if s > 0.3: cv2.circle(img_bgr, (int(x*img_bgr.shape[1]), int(y*img_bgr.shape[0])), 10, (0, 255, 0), -1)
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        st.write(f"判定: **{pose}** ({int(conf*100)}%)")

# --- 6. ゲームモード ---
elif st.session_state.mode == "GAME":
    if st.sidebar.button("⬅️ 中断"):
        st.session_state.mode = "MAIN"
        st.rerun()

    if st.session_state.game_status == "IDLE":
        st.session_state.target_pose = random.choice(labels)
        st.session_state.game_status = "READY"
        st.rerun()

    elif st.session_state.game_status == "READY":
        st.markdown(f'<div style="text-align:center; padding:20px; border:2px solid #00f2fe; border-radius:15px;"><h2>お題: <span style="color:#00f2fe">{st.session_state.target_pose}</span></h2></div>', unsafe_allow_html=True)
        if st.button("撮影スタート！", use_container_width=True):
            st.session_state.game_status = "CHALLENGE"
            st.rerun()

    elif st.session_state.game_status == "CHALLENGE":
        st.markdown(f"### お題: {st.session_state.target_pose}")
        
        # カウントダウン用要素とJSを同時に注入
        st.markdown('<div id="countdown-display"></div>', unsafe_allow_html=True)
        st.markdown(f"<script>{js_content} startCountdown();</script>", unsafe_allow_html=True)
        
        img_file = st.camera_input("ポーズを決めて！")
        if img_file:
            st.session_state.captured_img = img_file
            st.session_state.game_status = "RESULT"
            st.rerun()

    elif st.session_state.game_status == "RESULT":
        pose, conf, _, _ = analyze_pose(Image.open(st.session_state.captured_img))
        success = (pose == st.session_state.target_pose)

        if success:
            st.balloons()
            st.markdown('<div class="success-effect"></div>', unsafe_allow_html=True)
            st.success(f"🎊 正解！ お題通りの {pose} です！")
        else:
            rain_drops = ''.join([f'<div class="drop" style="left:{random.randint(0,100)}%; animation-delay:{random.random()}s;"></div>' for _ in range(50)])
            st.markdown(f'<div class="rain">{rain_drops}</div>', unsafe_allow_html=True)
            st.error(f"💀 残念... 判定は {pose} でした。")

        st.image(st.session_state.captured_img, use_container_width=True)
        if st.button("次のお題へ", use_container_width=True):
            st.session_state.game_status = "IDLE"
            st.rerun()
