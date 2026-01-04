import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import speech_recognition as sr
from streamlit_drawable_canvas import st_canvas
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import os

# --- 0. UI 高級感配置 ---
st.set_page_config(page_title="數位視覺核心", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #050505; color: #FFFFFF; }
    .stTabs [data-baseweb="tab-list"] { gap: 60px; justify-content: center; }
    .stTabs [data-baseweb="tab"] { font-size: 20px; font-weight: 200; color: #444; letter-spacing: 2px; }
    .stTabs [aria-selected="true"] { color: #00FBFF !important; border-bottom: 2px solid #00FBFF !important; }
    .res-box { background: rgba(0, 251, 255, 0.05); border: 1px solid rgba(0, 251, 255, 0.2); border-radius: 4px; padding: 30px; text-align: center; margin-top: 20px; }
    .res-val { font-size: 100px; font-weight: 100; color: #00FBFF; text-shadow: 0 0 20px rgba(0, 251, 255, 0.4); }
    </style>
    """, unsafe_allow_html=True)

# --- 1. 模型定義 (保持與原架構一致) ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9216, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# --- 2. 使用快取載入模型 (避免重複消耗記憶體) ---
@st.cache_resource
def load_model():
    model = Net()
    # 優先嘗試載入優化版，若無則載入一般版
    model_paths = ["enhanced_mnist_cnn.pth", "mnist_model.pth"]
    loaded = False
    for path in model_paths:
        if os.path.exists(path):
            try:
                model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                model.eval()
                loaded = True
                break
            except:
                continue
    return model if loaded else None

model = load_model()

# --- 3. WebRTC 影像處理器 ---
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 這裡可加入即時影像處理邏輯
        return img

# --- 4. 主介面佈局 ---
st.title("CORE VISION")
st.write("---")

tab1, tab2, tab3 = st.tabs(["✍️ 手寫辨識", "📷 實時影像", "🎙️ 語音辨識"])

with tab1:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=15,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="canvas",
        )
    with col_b:
        if canvas_result.image_data is not None and model is not None:
            img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
            res = ""
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 依照 x 座標排序，確保辨識順序由左至右
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
            
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 5 and h > 5:
                    roi = img[y:y+h, x:x+w]
                    # 邊框填充與縮放
                    pad = max(w, h)
                    canvas = np.zeros((pad, pad), dtype="uint8")
                    canvas[(pad-h)//2:(pad-h)//2+h, (pad-w)//2:(pad-w)//2+w] = roi
                    t = (torch.from_numpy(cv2.resize(canvas, (28, 28))).unsqueeze(0).unsqueeze(0).float()/255.0 - 0.1307)/0.3081
                    with torch.no_grad():
                        res += str(torch.argmax(model(t)).item())
            
            st.markdown(f'<div class="res-box"><p class="res-val">{res if res else "..."}</p></div>', unsafe_allow_html=True)

with tab2:
    col_c1, col_c2, col_c3 = st.columns([0.2, 2.6, 0.2])
    with col_c2:
        # 增加 RTC 配置以確保在線上環境能運作
        webrtc_streamer(
            key="main-cam",
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_transformer_factory=VideoTransformer,
            async_processing=True,
        )

with tab3:
    st.write("### 聲學感應辨識")
    st.info("點擊後請直接說話。說完請停頓約 1.5 秒。")
    
    if st.button("🎙️ 開始辨識"):
        r = sr.Recognizer()
        r.energy_threshold = 300
        r.pause_threshold = 1.2
        r.dynamic_energy_threshold = True 
        
        try:
            with sr.Microphone() as source:
                with st.spinner("聆聽中..."):
                    audio = r.listen(source, timeout=5, phrase_time_limit=10)
                text = r.recognize_google(audio, language="zh-TW")
                st.success(f"辨識結果：{text}")
        except Exception as e:
            st.error(f"辨識發生錯誤或未偵測到聲音。")

if model is None:
    st.warning("⚠️ 找不到模型檔案 (mnist_model.pth)，辨識功能將無法運作。")
