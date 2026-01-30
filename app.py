import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import random

# --- 1. 外部CSSの読み込み ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# --- 2. モデル・状態管理の初期化 ---
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

# ゲームの状態管理
if 'mode' not in st.session_state: st.session_state.mode = "MAIN"
if 'game_status' not in st.session_state: st.session_state.game_status = "IDLE" # IDLE, CHALLENGE, RESULT

# --- 3. メインメニュー ---
if st.session_state.mode == "MAIN":
    st.title("🧘 AI Yoga Master Pro")
    st.subheader("モードを選択してください")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧘 通常トレーニング\n(自由練習モード)"):
            st.session_state.mode = "PRACTICE"
            st.rerun()
    with col2:
        if st.button("🎮 ポーズ・チャレンジ\n(5秒制限ゲーム)"):
            st.session_state.mode = "GAME"
            st.session_state.game_status = "IDLE"
            st.rerun()

# --- 4. ゲームモードの実装 ---
elif st.session_state.mode == "GAME":
    st.title("🎮 Pose Challenge")
    
    if st.session_state.game_status == "IDLE":
        st.session_state.target_pose = random.choice(labels)
        st.markdown(f"""
            <div class="game-card">
                <h3>次のお題は...</h3>
                <h1 style="font-size: 4rem; color: #00f2fe;">{st.session_state.target_pose}</h1>
                <p>準備ができたら「スタート」を押して、5秒以内にポーズをとって！</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("ゲームスタート！"):
            st.session_state.game_status = "CHALLENGE"
            st.rerun()
            
    elif st.session_state.game_status == "CHALLENGE":
        st.warning(f"お題: {st.session_state.target_pose}")
        img_file = st.camera_input("5秒数えます！ポーズを決めて！")
        
        # 自動キャプチャのシミュレーション（カメラ入力があったら即判定へ）
        if img_file:
            st.session_state.captured_img = img_file
            st.session_state.game_status = "RESULT"
            st.rerun()

    elif st.session_state.game_status == "RESULT":
        # 解析ロジック (前回の解析コードと同じ)
        img = Image.open(st.session_state.captured_img)
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # --- AI推論 ---
        input_img = cv2.resize(img_bgr, (256, 256))
        input_tensor = np.expand_dims(input_img, axis=0).astype(np.uint8)
        m_in = movenet.get_input_details()[0]['index']
        movenet.set_tensor(m_in, input_tensor)
        movenet.invoke()
        keypoints = movenet.get_tensor(movenet.get_output_details()[0]['index'])[0, 0, :, :]
        
        input_data = keypoints.flatten().astype(np.float32).reshape(1, -1)
        classifier.set_tensor(classifier.get_input_details()[0]['index'], input_data)
        classifier.invoke()
        prediction = classifier.get_tensor(classifier.get_output_details()[0]['index'])[0]
        
        result_pose = labels[np.argmax(prediction)]
        success = (result_pose == st.session_state.target_pose)

        # --- 演出 ---
        if success:
            st.balloons()
            st.markdown('<div class="success-effect" style="position:fixed;top:0;left:0;width:100%;height:100%;z-index:98;pointer-events:none;"></div>', unsafe_allow_html=True)
            st.success(f"🔥 正解！ あなたのポーズは {result_pose} です！")
        else:
            # 雨の演出（CSSのdropを複数生成）
            rain_html = '<div class="rain">' + ''.join([f'<div class="drop" style="left:{random.randint(0,100)}%; animation-delay:{random.random()}s;"></div>' for _ in range(50)]) + '</div>'
            st.markdown(rain_html, unsafe_allow_html=True)
            st.error(f"💀 残念... 判定は {result_pose} でした。")

        st.image(img, width=400)

        # --- リトライ・終了 ---
        col1, col2 = st.columns(2)
        with col1:
            if st.button("もう一度挑戦（リトライ）"):
                st.session_state.game_status = "IDLE"
                st.rerun()
        with col2:
            if st.button("終了（メインメニューへ）"):
                st.session_state.mode = "MAIN"
                st.rerun()

# 5. 通常モード (以前のコードをここに移植)
elif st.session_state.mode == "PRACTICE":
    if st.button("⬅️ メニューに戻る"):
        st.session_state.mode = "MAIN"
        st.rerun()
    st.write("通常トレーニングモード実装中...")
