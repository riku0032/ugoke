# ugoke!
ポーズ識別ゲーミフィケーションアプリ

# リンク
公開Web: URL:https://settings-mtgysucqxyuqx2i6dxsq7s.streamlit.app/  
  
Figma : https://www.figma.com/board/QDseLoYFIXYQFhLdDsT8N1/3%E5%B9%B4%E7%94%9F-%E3%83%81%E3%83%BC%E3%83%A014-%E5%A4%A7%E7%9F%B3%E3%80%81%E5%B1%B1%EF%A8%91%E3%80%81%E5%B7%9D%E9%87%8E-%E6%9C%88%E6%9B%9C5-6%E9%99%90%E5%BE%8C%E6%9C%9F?node-id=0-1&p=f&t=9A8NM1elJUJnEx7V-0  
  
画面構成図 : https://www.figma.com/design/88LxnHCYsUvbIeI4wAw1HV/%E7%94%BB%E9%9D%A2%E6%A7%8B%E6%88%90%E5%9B%B3?node-id=0-1&p=f&t=fc9ev07Wk65DQn7F-0  
  


# 概要
本アプリはアプリ側で学習済みのモデルからランダムに指定されたポーズをユーザーに実演させるアプリである  

本アプリを通し、言葉だけでないノンバーバルコミュニケーションについてユーザーに再認識してもらうことが目的である  

# 目的
先ほど述べたがユーザーにノンバーバルコミュニケーションについて認識してもらうことである。  

・昨今ではリモートワークが増加しているということもあり、  
対面であれば不自由していなかったコミュニケーションが
ひとたび画面越しであると上手く行えないことが挙げられる

・AI技術の発展に伴い、リアルタイムでのユーザーの動きを認識し判別することが可能となった  

以上2点から、最新技術を用いた、ノンバーバルコミュニケーション実践ツールの開発に着手した  

# 主な機能
・tensorflow.jsの骨格検出(予定)を用いユーザーの現在の動きを判別  
・あらかじめ用意したモデル(モデルごとにポーズ名を割り振る)から、ユーザーにポーズをとらせる  
・ユーザーのポーズを判定し、お題に沿っていればポイント(ポーズの正確さ、速度など基準を設ける)  

# 本プロジェクトについて
本プロジェクト始動時、推論モデルの学習データを自作の予定であったが、開発が難航したため、  
kaggleにて公開されているヨガのポーズのデータセット(https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset)  
を学習元データとして利用する運びとなった。


# 利用技術
## 言語・実行環境

・Python（アプリ本体とノートブック） — src/app.py, src/train_model.ipynb  
・Jupyter Notebook（モデル学習スクリプト） — src/train_model.ipynb

## Web フレームワーク / UI

Streamlit（UI・カメラ入力・ページ構成） — src/app.py  
Streamlit Components API（streamlit.components.v1 を使用） — src/app.py  
Streamlit Community Cloud (Web公開)　— URL:https://settings-mtgysucqxyuqx2i6dxsq7s.streamlit.app/

## コンピュータビジョン / 画像処理

OpenCV (cv2) — 画像処理、描画、リサイズ等 — src/app.py, src/train_model.ipynb  
Pillow (PIL.Image) — 画像入出力 — src/app.py  
ブラウザカメラ（Streamlit の camera_input 経由）、および独自の自動シャッター用 JavaScript（自動撮影 JS を埋め込む） — src/app.py（camera_input / auto_shutter_js）  

## 機械学習 / モデル関連

TensorFlow（tf を import、モデル学習や tflite 生成） — src/app.py, src/train_model.ipynb  
TensorFlow Lite（tf.lite.Interpreter を用いて .tflite モデルをロード／推論） — src/app.py  
MoveNet（movenet_thunder.tflite として利用される姿勢検出モデル） — src/app.py（movenet_thunder.tflite を参照）  
tflite 形式の分類器（pose_classifier.tflite）およびラベルファイル（pose_labels.txt） — src/pose_labels.txt, src/app.py   

## データ処理 / ML 補助ライブラリ

NumPy（数値配列操作） — src/app.py, src/train_model.ipynb  
scikit-learn（train_test_split 等、学習用ユーティリティ） — src/train_model.ipynb（from sklearn.model_selection import train_test_split）  
Matplotlib（可視化に使用） — src/train_model.ipynb（plt の呼び出し）  
CSV（train_data.csv の読み書き／データセット形式） — src/train_model.ipynb（DATASET_CSV の記述）  
collections.namedtuple（ノートブック内でのデータ構造定義など） — src/train_model.ipynb  

# 追加開発予定
・ポーズ測定をラウンド制とし、１ゲーム5ラウンドで他者とポイントを競うなどの、遊びの幅を広げる  
・ハードモード(骨格検出で可能なら複数人前提のポーズ)実装  


