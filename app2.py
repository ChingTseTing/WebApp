import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import speech_recognition as sr
from streamlit_drawable_canvas import st_canvas
from streamlit_webrtc import (
    webrtc_streamer,
    VideoTransformerBase,
    AudioProcessorBase,
)
import av
import os
import tempfile
import wave

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

# --- 1. 模型載入 ---
class EnhancedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

@st.cache_resource
def load_model():
    m = EnhancedCNN()
    if os.path.exists("enhanced_mnist_cnn.pth"):
        m.load_state_dict(
            torch.load("enhanced_mnist_cnn.pth", map_location="cpu"),
            strict=False,
        )
    return m.eval()

model = load_model()

# --- 2. 鏡頭辨識核心 ---
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.GaussianBlur(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0
        )
        th = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21, 10
        )
        th = cv2.dilate(th, np.ones((2, 2), np.uint8), iterations=1)

        cnts, _ = cv2.findContours(
            th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        found = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if cv2.contourArea(c) > 200 and 0.05 < (w / float(h)) < 1.1:
                found.append((x, y, w, h))

        for (x, y, w, h) in sorted(found, key=lambda b: b[0]):
            roi = th[y:y + h, x:x + w]
            pad_h = int(h * 0.4)
            pad_w = int(max(w, h * 0.4) * 0.5)
            roi = cv2.copyMakeBorder(
                roi, pad_h, pad_h, pad_w, pad_w, 0
            )
            t = (
                torch.from_numpy(cv2.resize(roi, (28, 28)))
                .unsqueeze(0).unsqueeze(0)
                .float() / 255.0 - 0.1307
            ) / 0.3081

            with torch.no_grad():
                out = model(t)
                if torch.softmax(out, 1).max() > 0.5:
                    d = torch.argmax(out).item()
                    cv2.rectangle(
                        img, (x, y), (x + w, y + h), (0, 251, 255), 1
                    )
                    cv2.putText(
                        img, str(d), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 251, 255), 1
                    )
        return img

# --- WebRTC 音訊處理 ---
class WebRTCAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv_audio(self, frame: av.AudioFrame):
        self.frames.append(frame)
        return frame

# --- 3. 介面呈現 ---
st.markdown(
    "<h1 style='text-align: center; letter-spacing: 20px; font-weight:100; margin: 30px 0;'>數位視覺核心</h1>",
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(
    ["🖌️ 手寫數字辨識", "📷 鏡頭數字辨識", "🎙️ 語音辨識"]
)

# --- Tab 1 ---
with tab1:
    canvas_data = st_canvas(
        fill_color="white",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        height=450,
        width=1100,
        key="ult_canvas",
    )

    if st.button("🚀 執行辨識"):
        if canvas_data.image_data is not None:
            gray = cv2.cvtColor(
                canvas_data.image_data.astype("uint8"),
                cv2.COLOR_BGR2GRAY
            )
            _, th = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(
                th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            res = ""
            for c in sorted(cnts, key=lambda c: cv2.boundingRect(c)[0]):
                if cv2.contourArea(c) > 50:
                    x, y, w, h = cv2.boundingRect(c)
                    roi = cv2.copyMakeBorder(
                        th[y:y + h, x:x + w],
                        int(max(w, h) * 0.6),
                        int(max(w, h) * 0.6),
                        int(max(w, h) * 0.6),
                        int(max(w, h) * 0.6),
                        0,
                    )
                    t = (
                        torch.from_numpy(cv2.resize(roi, (28, 28)))
                        .unsqueeze(0).unsqueeze(0)
                        .float() / 255.0 - 0.1307
                    ) / 0.3081
                    res += str(torch.argmax(model(t)).item())

            st.markdown(
                f'<div class="res-box"><p class="res-val">{res}</p></div>',
                unsafe_allow_html=True
            )

# --- Tab 2 ---
with tab2:
    col_c1, col_c2, col_c3 = st.columns([0.2, 2.6, 0.2])
    with col_c2:
        webrtc_streamer(
            key="main-cam",
            video_transformer_factory=VideoTransformer,
        )

# --- Tab 3（語音辨識，雲端可用） ---
with tab3:
    st.write("### 聲學感應優化版")
    st.info("點擊後請直接說話。已調高偵測靈敏度，說完請停頓約 1.5 秒。")

    ctx = webrtc_streamer(
        key="speech",
        audio_processor_factory=WebRTCAudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )

    if st.button("🎙️ 開啟語音辨識"):
        if not ctx.audio_processor or not ctx.audio_processor.frames:
            st.error("尚未偵測到聲音，請先允許瀏覽器麥克風並說話")
        else:
            r = sr.Recognizer()
            r.energy_threshold = 300
            r.pause_threshold = 1.2
            r.dynamic_energy_threshold = True

            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    wf = wave.open(f.name, "wb")
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)

                    for frame in ctx.audio_processor.frames:
                        wf.writeframes(frame.to_ndarray().tobytes())

                    wf.close()

                    with sr.AudioFile(f.name) as source:
                        audio = r.record(source)

                text = r.recognize_google(audio, language="zh-TW")

                st.markdown(
                    f'''
                    <div class="res-box">
                        <p style="font-size:12px;color:#666;">ANALYSIS_RESULT</p>
                        <p class="res-val" style="font-size:60px;">{text}</p>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            except Exception:
                st.error("未偵測到聲音或語音辨識失敗，請再試一次")
