import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import random

# --- 1. 外部CSS読み込み ---
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass

local_css("style.css")

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

# --- 3. 状態管理の初期化 ---
if 'mode' not in st.session_state: st.session_state.mode = "MAIN"
if 'game_status' not in st.session_state: st.session_state.game_status = "IDLE"
if 'target_pose' not in st.session_state: st.session_state.target_pose = None

# AI解析関数 (共通利用)
def analyze_pose(image):
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape
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

# --- 4. メインメニュー ---
if st.session_state.mode == "MAIN":
    st.title("🧘 AI Yoga Master Pro")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧘 通常トレーニング", use_container_width=True):
            st.session_state.mode = "PRACTICE"
            st.rerun()
    with col2:
        if st.button("🎮 ポーズ・チャレンジ", use_container_width=True):
            st.session_state.mode = "GAME"
            st.session_state.game_status = "IDLE"
            st.rerun()

# --- 5. 通常モード (先ほどの機能を完全移植) ---
elif st.session_state.mode == "PRACTICE":
    if st.sidebar.button("⬅️ メニューに戻る"):
        st.session_state.mode = "MAIN"
        st.rerun()
    
    st.title("🧘 Practice Mode")
    img_file = st.camera_input("ポーズを撮ってください")
    if img_file:
        img = Image.open(img_file)
        if st.button("AI判定を実行"):
            pose_name, conf, keypoints, img_bgr = analyze_pose(img)
            h, w, _ = img_bgr.shape
            for i in range(17):
                y, x, s = keypoints[i]
                if s > 0.3: cv2.circle(img_bgr, (int(x*w), int(y*h)), 10, (0, 255, 0), -1)
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            st.write(f"判定: **{pose_name}** ({int(conf*100)}%)")

# --- 6. ゲームモード ---
elif st.session_state.mode == "GAME":
    if st.sidebar.button("⬅️ 中断してメニューへ"):
        st.session_state.mode = "MAIN"
        st.rerun()

    if st.session_state.game_status == "IDLE":
        # ここでお題を「一度だけ」抽選して固定する
        st.session_state.target_pose = random.choice(labels)
        st.session_state.game_status = "READY"
        st.rerun()

    elif st.session_state.game_status == "READY":
        st.markdown(f"""
            <div class="game-card">
                <h2>お題: <span style="color:#00f2fe">{st.session_state.target_pose}</span></h2>
                <p>「撮影開始」を押すとカメラが起動します。<br>5秒以内にポーズを完成させてシャッターを押して！</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("撮影開始！"):
            st.session_state.game_status = "CHALLENGE"
            st.rerun()

    elif st.session_state.game_status == "CHALLENGE":
        st.info(f"ターゲット: {st.session_state.target_pose}")
        # JSでのカウントダウンの代わりにメッセージを表示
        st.title("⏱ 5... 4... 3...")
        img_file = st.camera_input("ポーズを決めてシャッター！")
        
        if img_file:
            st.session_state.captured_img = img_file
            st.session_state.game_status = "RESULT"
            st.rerun()

    elif st.session_state.game_status == "RESULT":
        pose_name, conf, keypoints, img_bgr = analyze_pose(Image.open(st.session_state.captured_img))
        success = (pose_name == st.session_state.target_pose)

        if success:
            st.balloons()
            st.markdown('<div class="success-effect"></div>', unsafe_allow_html=True)
            st.success(f"🎊 正解！ お題通りの {pose_name} です！")
        else:
            rain_html = '<div class="rain">' + ''.join([f'<div class="drop" style="left:{random.randint(0,100)}%;"></div>' for _ in range(50)]) + '</div>'
            st.markdown(rain_html, unsafe_allow_html=True)
            st.error(f"💀 残念... 判定は {pose_name} でした。")

        st.image(st.session_state.captured_img, width=400)
        
        if st.button("もう一度（次のお題へ）"):
            st.session_state.game_status = "IDLE"
            st.rerun()
