import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import speech_recognition as sr
from streamlit_drawable_canvas import st_canvas
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
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
    .res-val { font-size: 100px; font-weight: 100; color: #00FBFF; font-family: 'Helvetica Neue', sans-serif; text-shadow: 0 0 20px rgba(0,251,255,0.4); }
    .stButton>button { border-radius: 2px; background: transparent; color: #00FBFF; border: 1px solid #00FBFF; height: 3.5em; width: 100%; transition: 0.3s; }
    .stButton>button:hover { background: rgba(0, 251, 255, 0.1); border-color: #FFF; color: #FFF; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. 模型定義與載入 ---
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 10))
    def forward(self, x): return self.classifier(self.features(x))

@st.cache_resource
def load_model():
    m = EnhancedCNN()
    model_path = 'enhanced_mnist_cnn.pth'
    if os.path.exists(model_path):
        # 關鍵：在雲端部署必須指定 map_location='cpu'
        m.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    return m.eval()

model = load_model()

# --- 2. 鏡頭辨識核心 (更新為 VideoProcessor) ---
class DigitProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3,3), 0)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        th = cv2.dilate(th, np.ones((2,2), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        found = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if cv2.contourArea(c) > 200 and 0.05 < (w/float(h)) < 1.1: 
                found.append((x, y, w, h))
        
        for (x, y, w, h) in sorted(found, key=lambda b: b[0]):
            roi = th[y:y+h, x:x+w]
            pad_h, pad_w = int(h*0.4), int(max(w, h*0.4)*0.5)
            roi = cv2.copyMakeBorder(roi, pad_h, pad_h, pad_w, pad_w, 0)
            
            # 預處理符合 MNIST 標準
            img_input = cv2.resize(roi, (28, 28))
            t = (torch.from_numpy(img_input).unsqueeze(0).unsqueeze(0).float()/255.0 - 0.1307)/0.3081
            
            with torch.no_grad():
                out = model(t)
                prob = torch.nn.functional.softmax(out, 1)
                if prob.max() > 0.5:
                    d = torch.argmax(out).item()
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 251, 255), 2)
                    cv2.putText(img, str(d), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 251, 255), 2)
        
        from av import VideoFrame
        return VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. 介面呈現 ---
st.markdown("<h1 style='text-align: center; letter-spacing: 20px; font-weight:100; margin: 30px 0;'>數位視覺核心</h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🖌️ 手寫數字辨識", "📷 鏡頭數字辨識", "🎙️ 語音辨識"])

with tab1:
    col_a1, col_a2, col_a3 = st.columns([0.1, 0.8, 0.1])
    with col_a2:
        canvas_data = st_canvas(fill_color="white", stroke_width=20, stroke_color="white",
                               background_color="black", height=450, width=1100, key="ult_canvas")
        if st.button("🚀 執行辨識", key="btn_canvas"):
            if canvas_data.image_data is not None:
                gray = cv2.cvtColor(canvas_data.image_data.astype('uint8'), cv2.COLOR_BGR2GRAY)
                _, th = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
                cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                res = ""
                for c in sorted(cnts, key=lambda c: cv2.boundingRect(c)[0]):
                    if cv2.contourArea(c) > 50:
                        x, y, w, h = cv2.boundingRect(c)
                        roi = cv2.copyMakeBorder(th[y:y+h, x:x+w], int(max(w,h)*0.6), int(max(w,h)*0.6), int(max(w,h)*0.6), int(max(w,h)*0.6), 0)
                        t = (torch.from_numpy(cv2.resize(roi, (28, 28))).unsqueeze(0).unsqueeze(0).float()/255.0 - 0.1307)/0.3081
                        res += str(torch.argmax(model(t)).item())
                if res == "": res = "0"
                st.markdown(f'<div class="res-box"><p class="res-val">{res}</p></div>', unsafe_allow_html=True)

with tab2:
    col_c1, col_c2, col_c3 = st.columns([0.1, 0.8, 0.1])
    with col_c2:
        # Streamlit Cloud 部署必須使用 WebRTC，使用者需同意瀏覽器授權
        webrtc_streamer(key="main-cam", video_processor_factory=DigitProcessor, 
                        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

with tab3:
    st.write("### 聲學感應介面")
    st.warning("⚠️ 注意：Streamlit Cloud 伺服器環境無法直接存取您的麥克風。此功能在本地執行時效果最佳。")
    
    if st.button("🎙️ 開啟語音辨識測試"):
        r = sr.Recognizer()
        r.energy_threshold = 300
        r.pause_threshold = 1.2
        
        try:
            with sr.Microphone() as source:
                st.toast("正在校準環境噪音...")
                r.adjust_for_ambient_noise(source, duration=0.8)
                st.toast("錄音中...", icon="🎤")
                audio = r.listen(source, timeout=5, phrase_time_limit=10)
                text = r.recognize_google(audio, language='zh-TW')
                st.markdown(f'<div class="res-box"><p style="font-size:12px;color:#666;">ANALYSIS_RESULT</p><p class="res-val" style="font-size:60px;">{text}</p></div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"辨識失敗：{str(e)}")
            st.info("提示：部署於雲端時，建議改用 API 方式或上傳音檔進行辨識。")
