from __future__ import annotations

import json
import base64
import hashlib
import io
import os
import tempfile
import threading
import uuid
import wave
from collections import deque
from pathlib import Path
from typing import Any

# Keep speech recognition and synthesis local to the app process.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from src.fer_app.model import EXPRESSION_CLASSES, FacialAffectNet


ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "artifacts" / "best.pt"
TEXT_MODEL_DIR = ROOT / "artifacts" / "text_model" / "model"
TEXT_THRESHOLDS_PATH = ROOT / "artifacts" / "text_model" / "thresholds.json"
TTS_MODEL_DIR = ROOT / "models" / "tts" / "kokoro"
TTS_SAMPLE_DIR = ROOT / "models" / "tts" / "samples"
TTS_WEIGHT_PATH = TTS_MODEL_DIR / "kokoro-v1_0.pth"
TTS_CONFIG_PATH = TTS_MODEL_DIR / "config.json"
TTS_VOICES = {
    "af_heart": "Heart · warm",
    "af_bella": "Bella · bright",
    "am_michael": "Michael · grounded",
}
TTS_SAMPLE_TEXT = "Hi, I’m Auri. Let’s take a closer look at the signals in this moment."
STT_MODEL_DIR = ROOT / "models" / "stt" / "whisper-tiny.en"
SPEECH_COMPONENT_DIR = ROOT / "components" / "speech_capture"
SPEECH_CAPTURE = components.declare_component("aura_speech_capture", path=str(SPEECH_COMPONENT_DIR))
FACE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
EMOJI = {
    "neutral": "😐",
    "happiness": "😄",
    "surprise": "😮",
    "sadness": "😔",
    "anger": "😠",
    "disgust": "🤢",
    "fear": "😨",
}
EMOTION_COLORS = {
    "neutral": "#aab8d2",
    "happiness": "#ffc857",
    "surprise": "#7ce7ff",
    "sadness": "#8ca5ff",
    "anger": "#ff7189",
    "disgust": "#b9f079",
    "fear": "#c6a0ff",
}
TTS_LOCK = threading.Lock()


st.set_page_config(
    page_title="AURA · Emotion Studio",
    page_icon="🪩",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { --bg:#17110f; --panel:#241a17; --panel-2:#30211c; --line:rgba(249,224,202,.14); --text:#fff8ed; --muted:#b9a79b; --hot:#ff4e24; --orange:#ff7844; --cream:#fff1dd; }
        html, body, [class*="css"] { font-family:Arial, Helvetica, sans-serif; }
        .stApp { background:radial-gradient(ellipse at 80% 0%,rgba(202,61,32,.2),transparent 35%),radial-gradient(ellipse at 0% 50%,rgba(94,47,32,.23),transparent 40%),var(--bg); color:var(--text); }
        [data-testid="stHeader"] { background:transparent; }
        [data-testid="stToolbar"] { right:1rem; }
        .block-container { max-width: 1500px; padding-top: 1.25rem; padding-bottom: 3rem; }
        [data-testid="stSidebar"] { background:linear-gradient(180deg,#211613 0%,#17110f 100%); border-right:1px solid var(--line); }
        [data-testid="stSidebar"] > div:first-child { padding-top:1.4rem; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:var(--muted); }
        h1,h2,h3 { color:var(--text)!important; letter-spacing:-.035em; }
        h1 { font-family:Impact,'Arial Narrow',sans-serif!important; font-weight:900!important; font-size:clamp(3.25rem,7.1vw,7rem)!important; line-height:.83!important; letter-spacing:-.035em!important; text-transform:uppercase; }
        h2 { font-size:1.5rem!important; }
        h3 { font-size:1.1rem!important; }
        p, label, span { color:inherit; }
        .eyebrow { color:#ffc18b; text-transform:uppercase; letter-spacing:.18em; font:bold .7rem 'Courier New',monospace; }
        .hero { min-height:510px; position:relative; overflow:hidden; display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:2rem 2.4rem; margin:.3rem 0 1.05rem; border:1px solid rgba(255,241,221,.18); border-radius:5px; background:radial-gradient(ellipse at 78% 42%,rgba(255,125,72,.26),transparent 32%),linear-gradient(112deg,#2b1715 0%,#67251d 48%,#c34222 100%); box-shadow:0 24px 70px rgba(0,0,0,.3); }
        .hero:before { content:'AURA'; position:absolute; left:-.035em; bottom:-.2em; color:rgba(255,241,221,.1); font:900 clamp(9rem,23vw,21rem)/.8 Impact,'Arial Narrow',sans-serif; letter-spacing:-.055em; pointer-events:none; white-space:nowrap; }
        .hero:after { content:''; position:absolute; width:48%; height:160%; right:-9%; top:-25%; background:rgba(255,241,221,.1); transform:rotate(31deg); border-left:1px solid rgba(255,241,221,.13); pointer-events:none; }
        .hero-copy { max-width:730px; z-index:2; }
        .hero-copy h1 { margin:.7rem 0 .85rem; text-shadow:0 5px 26px rgba(0,0,0,.14); }
        .hero-copy p { color:#f0d7c5; font-size:.95rem; margin:0; line-height:1.55; max-width:480px; }
        .hero-stamp { display:inline-flex; align-items:center; gap:.7rem; margin-top:1.3rem; padding:.55rem .7rem; border:1px solid rgba(255,224,197,.28); color:#ffe5d0; font:700 .62rem 'Courier New',monospace; letter-spacing:.1em; background:rgba(25,13,10,.18); }
        .hero-stamp span { color:#ffc177; font-size:1rem; }
        .hero-art { width:340px; min-width:300px; height:440px; display:grid; place-items:center; z-index:2; perspective:900px; }
        .hero-art svg { width:270px; height:auto; overflow:visible; filter:drop-shadow(0 24px 34px rgba(22,6,2,.42)); animation:auri-hover 4.6s ease-in-out infinite; }
        .auri-avatar { overflow:visible; }
        .auri-orbits { transform-origin:150px 235px; animation:auri-orbit 13s linear infinite; }
        .auri-character { transform-origin:150px 230px; animation:auri-dance 5.4s cubic-bezier(.45,0,.55,1) infinite; }
        .auri-hair-back { transform-origin:150px 78px; animation:auri-hair 2.9s ease-in-out infinite; }
        .auri-head { transform-origin:150px 112px; animation:auri-head-tilt 4.1s ease-in-out infinite; }
        .auri-legs { transform-origin:150px 300px; animation:auri-step 2.7s ease-in-out infinite; }
        .auri-arm-left { transform-origin:126px 157px; animation:auri-wave-left 2.6s ease-in-out infinite; }
        .auri-arm-right { transform-origin:175px 157px; animation:auri-wave-right 3.1s ease-in-out infinite; }
        .auri-eyes { transform-box:fill-box; transform-origin:center; animation:auri-blink 4.8s ease-in-out infinite; }
        .spark { transform-box:fill-box; transform-origin:center; animation:auri-twinkle 1.8s ease-in-out infinite; }
        .spark-b { animation-delay:.4s; }.spark-c { animation-delay:.8s; }.spark-d { animation-delay:1.2s; }.spark-e { animation-delay:.6s; }
        @keyframes auri-hover { 0%,100% { transform:translateY(0) rotate(-1deg) scale(1); } 50% { transform:translateY(-15px) rotate(1.2deg) scale(1.025); } }
        @keyframes auri-dance { 0%,100% { transform:translateY(0) rotate(-1deg); } 18% { transform:translateY(-11px) rotate(-3deg); } 35% { transform:translateY(1px) rotate(2deg); } 53% { transform:translateY(-8px) rotate(3deg); } 72% { transform:translateY(2px) rotate(-2deg); } 87% { transform:translateY(-5px) rotate(1deg); } }
        @keyframes auri-head-tilt { 0%,100% { transform:rotate(0); } 28% { transform:rotate(-5deg); } 58% { transform:rotate(4deg); } 82% { transform:rotate(-2deg); } }
        @keyframes auri-step { 0%,100% { transform:rotate(0); } 25% { transform:rotate(-3deg); } 55% { transform:rotate(4deg); } 78% { transform:rotate(-2deg); } }
        @keyframes auri-blink { 0%,45%,49%,100% { transform:scaleY(1); } 47% { transform:scaleY(.08); } }
        @keyframes auri-hair { 0%,100% { transform:rotate(-2deg); } 50% { transform:rotate(3deg); } }
        @keyframes auri-wave-left { 0%,100% { transform:rotate(0); } 25% { transform:rotate(-13deg); } 50% { transform:rotate(-4deg); } 76% { transform:rotate(-16deg); } }
        @keyframes auri-wave-right { 0%,100% { transform:rotate(0); } 26% { transform:rotate(15deg); } 54% { transform:rotate(4deg); } 78% { transform:rotate(12deg); } }
        @keyframes auri-orbit { to { transform:rotate(360deg); } }
        @keyframes auri-twinkle { 0%,100% { opacity:.35; transform:scale(.7) rotate(0); } 50% { opacity:1; transform:scale(1.25) rotate(40deg); } }
        @keyframes float { 0%,100% { transform:translateY(0) rotate(-2deg); } 50% { transform:translateY(-8px) rotate(2deg); } }
        .orb { filter:drop-shadow(0 20px 28px rgba(26,8,3,.4)); }
        .ticker { display:flex; justify-content:space-between; gap:1rem; overflow:hidden; margin:0 0 1.35rem; padding:.75rem 1rem; color:#26130e; background:var(--cream); font:bold .68rem 'Courier New',monospace; text-transform:uppercase; letter-spacing:.08em; }
        .brand { display:flex; align-items:center; gap:.7rem; padding:.3rem .2rem 1.2rem; }
        .brand-mark { width:38px; height:38px; border-radius:4px; display:grid; place-items:center; background:linear-gradient(145deg,#ff9b59,#ed421e); box-shadow:0 7px 24px rgba(239,83,38,.25); font-size:1.3rem; color:#1b100c; }
        .brand-name { color:#fff2df; font-weight:900; letter-spacing:.16em; font-size:.95rem; }
        .brand-sub { color:#a88b7e; font:400 .62rem 'Courier New',monospace; letter-spacing:.12em; margin-top:.15rem; }
        .side-avatar { text-align:center; padding:1.35rem 1rem 1.1rem; margin:.2rem 0 1rem; background:linear-gradient(160deg,rgba(67,38,29,.96),rgba(33,23,20,.94)); border:1px solid var(--line); border-radius:4px; }
        .side-avatar img { width:126px; height:126px; filter:drop-shadow(0 0 24px rgba(255,116,67,.2)); }
        .side-avatar svg { width:126px; height:auto; filter:drop-shadow(0 0 24px rgba(255,116,67,.2)); }
        .side-avatar .name { color:#fff3e4; font-size:1rem; font-weight:800; margin-top:.15rem; }
        .side-avatar .desc { color:#b89a8c; font-size:.77rem; margin-top:.22rem; }
        .aura-avatar .mouth-open { visibility:hidden; }
        .aura-avatar.talking .mouth-open { visibility:visible; }
        .side-live { display:inline-flex; align-items:center; gap:.38rem; color:#ffd5b2; font:500 .65rem 'Courier New',monospace; letter-spacing:.08em; text-transform:uppercase; margin-top:.9rem; padding:.35rem .58rem; border:1px solid rgba(255,157,95,.23); border-radius:2px; background:rgba(214,87,45,.1); }
        .dot { width:6px; height:6px; background:#ff8753; border-radius:50%; box-shadow:0 0 10px #ff8753; }
        .side-note { font-size:.76rem; line-height:1.6; color:#ad978b; padding:.75rem .8rem; background:rgba(255,255,255,.025); border:1px solid rgba(255,255,255,.06); border-radius:4px; }
        .section-kicker { display:flex; align-items:center; gap:.6rem; margin:.3rem 0 .8rem; color:#d4b8a5; font:500 .68rem 'Courier New',monospace; text-transform:uppercase; letter-spacing:.13em; }
        .section-kicker:before { content:''; height:2px; width:18px; background:var(--hot); }
        [data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.5rem; border-bottom:1px solid var(--line); }
        [data-testid="stTabs"] [data-baseweb="tab"] { height:54px; padding:0 1.15rem; color:#b49c8e; background:transparent; border-radius:0; font-weight:700; }
        [data-testid="stTabs"] [aria-selected="true"] { color:#fff2e3!important; background:rgba(239,82,39,.1)!important; border-bottom:2px solid var(--hot)!important; }
        [data-testid="stVerticalBlockBorderWrapper"] { background:linear-gradient(145deg,rgba(47,32,27,.92),rgba(32,23,20,.94)); border:1px solid var(--line)!important; border-radius:4px!important; }
        [data-testid="stMetric"] { background:rgba(255,255,255,.025); border:1px solid rgba(255,255,255,.07); border-radius:4px; padding:.9rem 1rem; }
        [data-testid="stMetricLabel"] { color:#b59d90!important; font-size:.75rem!important; }
        [data-testid="stMetricValue"] { color:#fff3e4!important; }
        .stButton > button, .stDownloadButton > button { border-radius:3px; border:1px solid rgba(255,145,92,.52); background:linear-gradient(120deg,#d74725,#f26938); color:#fff4e8; font-weight:900; min-height:2.8rem; transition:all .18s ease; text-transform:uppercase; letter-spacing:.045em; }
        .stButton > button:hover, .stDownloadButton > button:hover { border-color:#ffd0aa; color:white; transform:translateY(-1px); box-shadow:0 8px 26px rgba(239,83,38,.16); }
        .stButton > button:focus-visible, .stDownloadButton > button:focus-visible, [data-baseweb="tab"]:focus-visible { outline:3px solid #ffe1c2!important; outline-offset:3px; box-shadow:0 0 0 5px rgba(255,120,68,.2)!important; }
        [data-baseweb="select"] > div, [data-testid="stTextArea"] textarea { background:#241a17!important; border-color:rgba(249,224,202,.2)!important; border-radius:3px!important; color:#fff2e3!important; }
        [data-testid="stFileUploader"] { background:rgba(255,255,255,.018); border:1px dashed rgba(255,133,80,.42); border-radius:4px; padding:.5rem; }
        [data-testid="stProgress"] > div > div { background:linear-gradient(90deg,#ff4e24,#ffbc75); border-radius:0; }
        [data-testid="stAlert"] { border-radius:4px; }
        .result-card { border:1px solid var(--line); border-radius:4px; background:linear-gradient(145deg,rgba(55,37,30,.96),rgba(34,24,21,.98)); padding:1.2rem; height:100%; }
        .result-card .tiny { color:#c29e89; font:500 .66rem 'Courier New',monospace; letter-spacing:.13em; text-transform:uppercase; }
        .emotion-line { display:flex; align-items:center; justify-content:space-between; gap:.7rem; padding:.42rem 0; border-bottom:1px solid rgba(255,255,255,.06); }
        .emotion-line:last-child { border-bottom:0; }
        .emotion-name { color:#f3dfcd; font-size:.83rem; text-transform:capitalize; }
        .emotion-value { color:#c7a997; font:500 .74rem 'Courier New',monospace; }
        .mascot-row { display:flex; align-items:center; gap:1rem; padding:.35rem 0 .8rem; }
        .mascot-face { display:grid; place-items:center; width:76px; height:76px; border-radius:50%; font-size:2.8rem; background:radial-gradient(circle at 35% 25%,#f78d55,#a7301b 72%); border:1px solid rgba(255,209,167,.28); box-shadow:inset 0 0 22px rgba(255,231,194,.17),0 0 28px rgba(227,75,37,.13); }
        .mascot-label { color:#fff3e4; font-size:1.35rem; font-weight:800; letter-spacing:-.04em; text-transform:capitalize; }
        .muted { color:#bea393; font-size:.8rem; line-height:1.6; }
        .footer-note { margin-top:1.25rem; padding:.8rem 1rem; border:1px solid rgba(255,255,255,.08); border-radius:3px; background:rgba(255,255,255,.018); color:#b69d8e; font-size:.73rem; line-height:1.6; }
        .voice-box { border:1px solid rgba(255,145,92,.26); background:linear-gradient(120deg,rgba(122,52,32,.2),rgba(73,39,31,.24)); border-radius:3px; padding:.85rem 1rem; }
        .stCaption { color:#b49b8b!important; }
        @media (max-width: 760px) { .hero { min-height:340px; padding:1.2rem; } .hero-art { width:170px; height:290px; min-width:145px; } .hero-art svg { width:155px; height:auto; } h1 { font-size:3rem!important; } .hero-copy p { font-size:.84rem; } .ticker span:nth-child(n+3) { display:none; } }
        @media (max-width: 560px) { .block-container { padding-left:1rem; padding-right:1rem; } .hero { min-height:0; flex-direction:column; align-items:flex-start; gap:.25rem; padding:1.1rem 1rem .2rem; } .hero-copy { width:100%; } .hero-copy h1 { font-size:clamp(2.8rem,15vw,4.2rem)!important; } .hero-art { align-self:center; width:100%; min-width:0; height:250px; margin-top:-.35rem; } .hero-art svg { width:175px; } .ticker { gap:.6rem; padding:.7rem; font-size:.58rem; letter-spacing:.035em; } .ticker span:nth-child(n+2) { display:none; } }
        @media (prefers-reduced-motion: reduce) { *,*::before,*::after { animation-duration:.01ms!important; animation-iteration-count:1!important; scroll-behavior:auto!important; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def avatar_svg(expression: str = "neutral", size: int = 260) -> str:
    features = {
        "happiness": ("M125 82q10-9 22-1", "M157 81q12-8 23 2", "M133 111q18 18 37 0", 3.4, 3.4),
        "surprise": ("M124 76q11-15 24-3", "M157 73q13-11 24 4", "M139 113q10-10 21 0 3 16-10 16t-11-16", 5.2, 5.2),
        "sadness": ("M124 84q11-12 23-5", "M158 80q12 7 23 13", "M133 124q18-16 37 0", 3.1, 3.1),
        "anger": ("M124 80l24 7", "M157 87l24-8", "M134 122q18-8 36 0", 2.6, 2.6),
        "disgust": ("M124 84q12-10 23-2", "M158 80q12 8 23 2", "M134 120q17 6 35-2", 3.1, 3.1),
        "fear": ("M123 74q12-15 25-4", "M157 71q14-10 25 6", "M139 114q10-10 21 0 3 14-10 14t-11-14", 5.1, 5.1),
        "neutral": ("M125 82q10-9 22-1", "M157 82q12-8 23 1", "M134 117q18 1 36 0", 3.2, 3.2),
    }
    left_brow, right_brow, mouth, eye_rx, eye2_rx = features.get(expression, features["neutral"])
    eye_size = 5.1 if expression in {"surprise", "fear"} else 4
    uid = uuid.uuid4().hex[:10]
    height = round(size * 1.53)
    return f"""<svg class="auri-avatar aura-avatar" viewBox="0 0 300 460" width="{size}" height="{height}" aria-label="Auri, full-body feminine avatar with a {expression} expression">
      <defs>
        <linearGradient id="auri-bg-{uid}" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#ffb46e"/><stop offset=".52" stop-color="#f45b36"/><stop offset="1" stop-color="#792718"/></linearGradient>
        <linearGradient id="auri-skin-{uid}" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#ffd9bb"/><stop offset="1" stop-color="#bf775d"/></linearGradient>
        <linearGradient id="auri-coat-{uid}" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#fff4e7"/><stop offset="1" stop-color="#cfb9a6"/></linearGradient>
        <linearGradient id="auri-hair-{uid}" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#100d12"/><stop offset=".36" stop-color="#44231d"/><stop offset=".62" stop-color="#211213"/><stop offset="1" stop-color="#0d0d12"/></linearGradient>
        <radialGradient id="auri-iris-{uid}" cx="35%" cy="25%"><stop stop-color="#d59a62"/><stop offset=".45" stop-color="#8b4c2f"/><stop offset="1" stop-color="#321a17"/></radialGradient>
        <radialGradient id="auri-halo-{uid}"><stop stop-color="#ffb570" stop-opacity=".28"/><stop offset="1" stop-color="#ff6840" stop-opacity="0"/></radialGradient>
        <clipPath id="auri-clip-{uid}"><ellipse cx="150" cy="234" rx="119" ry="214"/></clipPath>
      </defs>
      <ellipse cx="150" cy="235" rx="145" ry="215" fill="url(#auri-halo-{uid})"/>
      <g class="auri-orbits" fill="none" stroke="#ffe6ce">
        <ellipse cx="150" cy="235" rx="117" ry="206" stroke-opacity=".3" stroke-width="1.4"/>
        <ellipse cx="150" cy="235" rx="131" ry="222" stroke="#ffb982" stroke-opacity=".5" stroke-width="1.2" stroke-dasharray="2 9"/>
        <path d="M27 251c45-25 201-24 246 0" stroke-opacity=".25" stroke-width="1"/>
      </g>
      <g class="auri-sparkles" fill="#fff0d8">
        <path class="spark spark-a" d="M42 111l4 10 10 4-10 4-4 10-4-10-10-4 10-4z"/>
        <path class="spark spark-b" d="M252 178l3 7 7 3-7 3-3 7-3-7-7-3 7-3z"/>
        <circle class="spark spark-c" cx="48" cy="322" r="3"/><circle class="spark spark-d" cx="254" cy="339" r="4"/>
        <circle class="spark spark-e" cx="217" cy="69" r="2.5"/>
      </g>
      <g class="auri-character" clip-path="url(#auri-clip-{uid})">
        <g class="auri-hair-back">
          <path d="M111 81c-5-48 21-69 53-63 38 7 51 39 38 88-9 33 18 72 10 117-4 24-17 45-34 51l-7-142z" fill="url(#auri-hair-{uid})"/>
          <path d="M111 75c-21 32-9 69-22 104-9 24-7 52 8 71 12-13 19-45 17-68l19-87z" fill="url(#auri-hair-{uid})"/>
          <path d="M197 76c17 35 4 72 18 108 7 20 8 46-5 66-12-12-19-40-17-60l-15-91z" fill="url(#auri-hair-{uid})"/>
          <path d="M116 91c-6 43-10 86-18 112M201 89c9 45 5 88 14 117" fill="none" stroke="#7a3d2b" stroke-opacity=".48" stroke-width="4" stroke-linecap="round"/>
        </g>
        <g class="auri-legs">
        <path d="M123 296c-9 35-8 72-13 111l29 2 19-90 12 90 29-2c-7-44-6-78-17-111z" fill="#312622"/>
        <path d="M112 397l29 2-5 19 2 12c-10 9-29 10-45 3l8-12zM170 398l29-2 9 24 11 12c-12 9-32 8-46 0l1-13z" fill="#b64b31"/>
        <path d="M99 425q17 8 39-1l-2 9q-17 14-43 2zM179 425q19 8 40 0l10 11q-17 17-49 3z" fill="#f1c29e"/>
        </g>
        <g class="auri-arm auri-arm-left">
          <path d="M126 157c-19 2-31 18-43 44l-15 37c-4 11 1 21 11 24 10 3 18-4 21-15l16-31 28-38z" fill="url(#auri-coat-{uid})"/>
          <path d="M73 237q-12 8-16 19-2 8 5 11 7 2 12-5l8-13z" fill="url(#auri-skin-{uid})"/>
          <path d="M93 202l18 10-13 29-18-8z" fill="#d9c5b3"/>
        </g>
        <g class="auri-arm auri-arm-right">
          <path d="M175 157c19 2 31 18 43 44l15 37c4 11-1 21-11 24-10 3-18-4-21-15l-16-31-28-38z" fill="url(#auri-coat-{uid})"/>
          <path d="M227 237q12 8 16 19 2 8-5 11-7 2-12-5l-8-13z" fill="url(#auri-skin-{uid})"/>
          <path d="M207 202l-18 10 13 29 18-8z" fill="#d9c5b3"/>
        </g>
        <path class="auri-torso" d="M126 145q24-14 49 0l27 38-13 112q-38 19-79 0l-13-112z" fill="url(#auri-coat-{uid})"/>
        <path d="M132 151l18 25 18-25-7-11h-22z" fill="#fff9ef"/>
        <path d="M150 176v107M112 193l33 15-17 21-26-27M188 193l-33 15 17 21 26-27" fill="none" stroke="#b9927e" stroke-opacity=".7" stroke-width="2"/>
        <path d="M120 281q31 12 61 0l8 18q-36 18-77 0z" fill="#6b2920"/>
        <path d="M145 285h12v13h-12z" fill="#ffbc78"/>
        <path d="M121 301l27-2-3 63-18 48-19-1 17-70zM151 299l29 3-3 38 16 70-20 1-22-48z" fill="#e4d6c8"/>
        <path d="M134 327l14 2M153 326l16 1" stroke="#ba9f8b" stroke-width="2"/>
        <path d="M121 351l18 8M155 358l19-8" stroke="#ba9f8b" stroke-opacity=".65" stroke-width="1.5"/>
        <g class="auri-head">
        <path d="M141 134v25q8 14 17 0v-27z" fill="url(#auri-skin-{uid})"/>
        <ellipse cx="150" cy="91" rx="42" ry="54" fill="url(#auri-skin-{uid})"/>
        <path d="M108 88c-5-47 19-70 48-70 39 0 54 29 41 75l-12-18-6-25c-17 16-39 23-69 16z" fill="url(#auri-hair-{uid})"/>
        <path d="M109 86q-15 35-10 91M191 79q15 43 18 97" fill="none" stroke="url(#auri-hair-{uid})" stroke-width="14" stroke-linecap="round"/>
        <path d="M119 44q17-28 43-21M145 29q24-17 41 1" fill="none" stroke="#b77750" stroke-opacity=".64" stroke-width="3" stroke-linecap="round"/>
        <path d="M114 60q-7 35-3 64M198 84q10 34 8 64" fill="none" stroke="#a75f43" stroke-opacity=".48" stroke-width="2.4" stroke-linecap="round"/>
        <path d="M117 61c1-20 13-35 27-40" fill="none" stroke="#fff0d8" stroke-opacity=".28" stroke-width="3" stroke-linecap="round"/>
        <path d="{left_brow}" fill="none" stroke="#4c2920" stroke-width="3.7" stroke-linecap="round"/>
        <path d="{right_brow}" fill="none" stroke="#4c2920" stroke-width="3.7" stroke-linecap="round"/>
        <path d="M129 87q8-7 16 0M158 87q8-7 16 0" fill="none" stroke="#6f3a2b" stroke-width="1.5" stroke-linecap="round"/>
        <g class="auri-eyes">
          <ellipse cx="137" cy="91" rx="6.2" ry="7" fill="#fff7ea"/><ellipse cx="166" cy="91" rx="6.2" ry="7" fill="#fff7ea"/>
          <ellipse cx="137" cy="91" rx="5.1" ry="5.8" fill="url(#auri-iris-{uid})"/><ellipse cx="166" cy="91" rx="5.1" ry="5.8" fill="url(#auri-iris-{uid})"/>
          <ellipse cx="137" cy="91" rx="{eye_rx}" ry="{eye_size}" fill="#241712"/><ellipse cx="166" cy="91" rx="{eye2_rx}" ry="{eye_size}" fill="#241712"/>
          <circle cx="138" cy="89.5" r="1.7" fill="#fff4e3"/><circle cx="167" cy="89.5" r="1.7" fill="#fff4e3"/>
        </g>
        <path d="M151 93l-4 16q4 4 9 0" fill="none" stroke="#a76249" stroke-width="2.5" stroke-linecap="round"/>
        <path class="mouth-smile" d="{mouth}" fill="none" stroke="#7a3228" stroke-width="3.8" stroke-linecap="round"/>
        <ellipse class="mouth-open" cx="151" cy="119" rx="7" ry="9" fill="#702a25" opacity="0"/>
        <ellipse cx="124" cy="108" rx="7" ry="3.6" fill="#d77c69" fill-opacity=".42"/><ellipse cx="178" cy="108" rx="7" ry="3.6" fill="#d77c69" fill-opacity=".42"/>
        <path d="M122 136q28 9 57 0" fill="none" stroke="#bb886f" stroke-opacity=".5" stroke-width="1.5"/>
        <circle cx="190" cy="112" r="3.5" fill="#ffc77f"/><path d="M190 116v12" stroke="#ffc77f" stroke-width="2"/>
        </g>
      </g>
    </svg>"""


def render_shell() -> None:
    mood = st.session_state.get("auri_mood", "neutral")
    st.session_state.setdefault("auri_voice", "af_heart")
    with st.sidebar:
        st.markdown(
            "<div class='brand'><div class='brand-mark'>✦</div><div><div class='brand-name'>AURA</div><div class='brand-sub'>EMOTION STUDIO · LOCAL AI</div></div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='side-avatar'>{avatar_svg(mood,132)}<div class='name'>Auri · she/her</div><div class='desc'>Your affective signal guide</div><div class='side-live'><i class='dot'></i>Models running locally</div></div>",
            unsafe_allow_html=True,
        )
        face_status = "Ready" if CHECKPOINT_PATH.exists() else "Checkpoint missing"
        text_status = "Ready" if (TEXT_MODEL_DIR / "config.json").exists() else "Checkpoint missing"
        st.markdown("<div class='section-kicker'>Model status</div>", unsafe_allow_html=True)
        st.caption(f"◉ Face expression · {face_status}")
        st.caption(f"◉ Text emotion · {text_status}")
        tts_status = "Ready" if TTS_WEIGHT_PATH.exists() and (TTS_MODEL_DIR / "voices").is_dir() else "Download needed"
        st.caption(f"◉ Auri voice · {tts_status}")
        stt_status = "Ready" if (STT_MODEL_DIR / "model.bin").is_file() else "Download needed"
        st.caption(f"◉ Voice transcription · {stt_status} · local")
        st.selectbox(
            "Auri’s offline voice",
            options=list(TTS_VOICES),
            format_func=lambda voice: TTS_VOICES[voice],
            key="auri_voice",
        )
        with st.expander("Hear the three local voices"):
            for voice, label in TTS_VOICES.items():
                if st.button(f"Listen · {label}", key=f"preview_voice_{voice}", width="stretch"):
                    try:
                        sample_path = TTS_SAMPLE_DIR / f"{voice}.wav"
                        with st.spinner("Loading the local voice sample…"):
                            st.session_state[f"auri_sample_{voice}"] = (
                                sample_path.read_bytes()
                                if sample_path.is_file()
                                else synthesize_local_speech(TTS_SAMPLE_TEXT, voice)
                            )
                            st.session_state["auri_preview_voice"] = voice
                    except Exception as exc:
                        st.error(f"The local voice sample could not be generated: {exc}")
            preview_voice = st.session_state.get("auri_preview_voice")
            if preview_voice and f"auri_sample_{preview_voice}" in st.session_state:
                render_spoken_avatar(
                    st.session_state[f"auri_sample_{preview_voice}"],
                    "neutral",
                    f"Sample · {TTS_VOICES[preview_voice]}",
                    key=f"preview_audio_{preview_voice}",
                )
        st.markdown(
            "<div class='side-note'>AURA reads visible expression cues and language patterns. It can’t know what someone truly feels.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:1.4rem'></div><div class='brand-sub'>BUILT FOR EXPLORATION · V1.0</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='hero'><div class='hero-copy'><div class='eyebrow'>Local models · meet your signal guide</div><h1>Feel the<br><span style='color:#ffbc88'>moment.</span></h1><p>Meet Auri: a warm, expressive guide through the signals in a face or a sentence. Explore the model’s read, then let her tell you what stands out.</p><div class='hero-stamp'>AURI / EMOTION STUDIO <span>✦</span> FULL-BODY AVATAR</div></div><div class='hero-art'>{avatar_svg(mood,270)}</div></div><div class='ticker'><span>FACE · VOICE · WORDS</span><span>LIVE EXPRESSION SCAN</span><span>LOCAL MODEL INFERENCE</span><span>EXPLORE THE SIGNAL ✦</span></div>",
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_face_model(path: str) -> tuple[FacialAffectNet, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    device = choose_device()
    model = FacialAffectNet.from_checkpoint(checkpoint).to(device)
    model.eval()
    return model, checkpoint


@st.cache_resource
def load_text_model(path: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    model.to(choose_device())
    model.eval()
    return tokenizer, model


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_tts_pipeline():
    if not TTS_WEIGHT_PATH.is_file() or not TTS_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            "The local Kokoro voice model is missing. Run `python tools/download_tts_assets.py` once."
        )
    # Misaki otherwise downloads its spaCy English model the first time it starts.
    # Requiring the already-installed local package keeps inference fully offline.
    import spacy

    if not spacy.util.is_package("en_core_web_sm"):
        raise RuntimeError(
            "The local English phoneme model is missing. Install requirements.txt before using Auri’s voice."
        )

    from kokoro import KModel, KPipeline

    device = str(choose_device())
    model = KModel(
        repo_id=str(TTS_MODEL_DIR),
        config=str(TTS_CONFIG_PATH),
        model=str(TTS_WEIGHT_PATH),
    ).to(device).eval()
    return KPipeline(lang_code="a", repo_id=str(TTS_MODEL_DIR), model=model, trf=False)


@st.cache_resource
def load_stt_model():
    if not (STT_MODEL_DIR / "model.bin").is_file():
        raise FileNotFoundError(
            "The local Whisper speech model is missing. Run `python tools/download_stt_assets.py` once."
        )
    from faster_whisper import WhisperModel

    return WhisperModel(
        str(STT_MODEL_DIR),
        device="cpu",
        compute_type="int8",
        cpu_threads=max(2, min(8, (os.cpu_count() or 4) - 1)),
        local_files_only=True,
    )


def transcribe_local_audio(audio_bytes: bytes) -> str:
    if not audio_bytes:
        return ""
    model = load_stt_model()
    segments, _ = model.transcribe(
        io.BytesIO(audio_bytes),
        language="en",
        beam_size=3,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()


@st.cache_data(max_entries=24, show_spinner=False)
def synthesize_local_speech(text: str, voice: str) -> bytes:
    if voice not in TTS_VOICES:
        raise ValueError("Choose one of Auri’s bundled local voices.")
    voice_path = TTS_MODEL_DIR / "voices" / f"{voice}.pt"
    if not voice_path.is_file():
        raise FileNotFoundError(f"The downloaded local speaker file is missing: {voice_path.name}")
    pipeline = load_tts_pipeline()
    clips: list[np.ndarray] = []
    with TTS_LOCK, torch.inference_mode():
        for result in pipeline(text, voice=str(voice_path), speed=0.96):
            if result.audio is not None:
                clips.append(result.audio.detach().cpu().numpy().astype(np.float32, copy=False))
    if not clips:
        raise RuntimeError("The local speech model did not return audio for this summary.")
    samples = np.concatenate(clips)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm)
    return wav_buffer.getvalue()


def text_expression(label: str) -> str:
    label = label.lower().replace("_", " ")
    for expression, words in {
        "happiness": {"joy", "amusement", "approval", "caring", "desire", "excitement", "gratitude", "love", "optimism", "pride", "relief", "admiration", "happy"},
        "sadness": {"sadness", "disappointment", "grief", "remorse", "embarrassment"},
        "anger": {"anger", "annoyance", "disapproval"},
        "fear": {"fear", "nervousness"},
        "surprise": {"surprise", "realization", "confusion"},
        "disgust": {"disgust"},
    }.items():
        if label in words:
            return expression
    return "neutral"


def text_summary(rows: list[dict[str, Any]], selected: list[str]) -> tuple[list[str], str]:
    if not rows:
        return [], "neutral"
    top = rows[0]
    label = top["label"].replace("_", " ")
    runner_up = rows[1] if len(rows) > 1 else None
    active_rows = [row for row in rows if row["selected"]]
    active = ", ".join(row["label"].replace("_", " ") for row in active_rows[:3])
    other = (
        f"The next strongest cue is {runner_up['label'].replace('_', ' ')} ({runner_up['score'] * 100:.0f}%)."
        if runner_up
        else "There are no other scored cues to compare."
    )
    flagged = f"The tuned thresholds flagged {active}." if active else "No emotion label crossed its tuned threshold."
    lines = [
        f"The strongest language cue is {label}, with a model score of {top['score'] * 100:.0f} percent.",
        other,
        flagged,
        "These are patterns in the words, not certainty about anyone’s private feelings.",
    ]
    return lines, text_expression(top["label"])


def face_summary(results: list[dict[str, Any]]) -> tuple[list[str], str]:
    first = results[0]
    scores = first["scores"]
    order = np.argsort(scores)[::-1]
    second_label = EXPRESSION_CLASSES[int(order[1])] if len(order) > 1 else first["label"]
    second_score = float(scores[int(order[1])]) if len(order) > 1 else first["score"]
    count = len(results)
    lines = [
        f"AURA found {count} visible face{'s' if count != 1 else ''} in this image.",
        f"The largest face’s leading expression cue is {first['label']}, scored at {first['score'] * 100:.0f} percent.",
        f"The next strongest cue is {second_label}, at {second_score * 100:.0f} percent.",
        "These are visual patterns from one frame, not proof of how anyone feels.",
    ]
    return lines, first["label"]


def render_spoken_avatar(audio: bytes, expression: str, caption: str, key: str) -> None:
    avatar = avatar_svg(expression, 88)
    audio_src = "data:audio/wav;base64," + base64.b64encode(audio).decode("ascii")
    safe_caption = caption.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    wrapper_id = "audio_" + "".join(char for char in key if char.isalnum() or char == "_")
    html = f"""
    <style>
      body {{ margin:0; background:transparent; color:#fff1dd; font-family:Arial, sans-serif; }}
      .speak-row {{ display:flex; align-items:center; gap:16px; min-height:140px; padding:8px 6px; }}
      .aura-avatar {{ width:88px; height:135px; flex:none; filter:drop-shadow(0 8px 18px rgba(255,95,50,.22)); }}
      audio {{ width:100%; min-width:220px; accent-color:#ff7040; }}
      .listen-copy {{ color:#f4d6bf; font-size:12px; letter-spacing:.12em; text-transform:uppercase; margin:0 0 8px 2px; }}
      .auri-avatar {{ animation:guide-bob 3.4s ease-in-out infinite; }}
      .auri-character {{ transform-origin:150px 230px; animation:guide-breathe 2.8s ease-in-out infinite; }}
      .auri-hair-back {{ transform-origin:150px 78px; animation:guide-hair 3.3s ease-in-out infinite; }}
      .auri-arm-left {{ transform-origin:126px 157px; animation:guide-wave 3s ease-in-out infinite; }}
      .auri-arm-right {{ transform-origin:175px 157px; animation:guide-wave 3.8s ease-in-out infinite reverse; }}
      .auri-orbits {{ transform-origin:150px 235px; animation:guide-orbit 20s linear infinite; }}
      .mouth-open {{ transform-box:fill-box; transform-origin:center; transform:scaleY(0); }}
      .aura-avatar.talking .mouth-smile {{ opacity:0; }}
      .aura-avatar.talking .mouth-open {{ opacity:1; animation:lip 190ms infinite alternate ease-in-out; }}
      .aura-avatar.talking circle:nth-of-type(2) {{ animation:orbit 1.4s linear infinite; }}
      @keyframes guide-bob {{ 0%,100% {{ transform:translateY(0) rotate(-1deg); }} 50% {{ transform:translateY(-6px) rotate(1deg); }} }}
      @keyframes guide-breathe {{ 0%,100% {{ transform:translateY(0) scaleY(1); }} 50% {{ transform:translateY(-2px) scaleY(1.012); }} }}
      @keyframes guide-hair {{ 0%,100% {{ transform:rotate(-1deg); }} 50% {{ transform:rotate(1.8deg); }} }}
      @keyframes guide-wave {{ 0%,100% {{ transform:rotate(0); }} 50% {{ transform:rotate(5deg); }} }}
      @keyframes guide-orbit {{ to {{ transform:rotate(360deg); }} }}
      @keyframes lip {{ from {{ transform:scaleY(.18); }} to {{ transform:scaleY(1); }} }}
      @keyframes orbit {{ to {{ transform:rotate(360deg); transform-origin:100px 100px; }} }}
      @media(max-width:420px) {{ .aura-avatar {{ width:72px;height:110px; }} .speak-row {{ gap:8px; }} }}
    </style>
    <div id="{wrapper_id}" class="speak-row"><div>{avatar}</div><div style="flex:1;min-width:0"><div class="listen-copy">{safe_caption}</div><audio controls preload="none" src="{audio_src}"></audio></div></div>
    <script>
      const a = document.querySelector('audio');
      const v = document.querySelector('.aura-avatar');
      a.addEventListener('play', () => v.classList.add('talking'));
      a.addEventListener('pause', () => v.classList.remove('talking'));
      a.addEventListener('ended', () => v.classList.remove('talking'));
    </script>
    """
    components.html(html, height=158, scrolling=False)


def render_speak_button(summary: list[str], expression: str, key: str) -> None:
    voice = st.session_state.get("auri_voice", "af_heart")
    audio_key = f"auri_audio_{key}"
    if st.button("▶  Auri reads this", key=f"speak_{key}", width="stretch"):
        try:
            with st.spinner(f"Auri is speaking with the local {TTS_VOICES[voice]} voice…"):
                st.session_state[audio_key] = {
                    "bytes": synthesize_local_speech(" ".join(summary), voice),
                    "voice": voice,
                }
        except Exception as exc:
            st.error(f"I couldn’t create local speech: {exc}")
    if audio_key in st.session_state:
        saved_audio = st.session_state[audio_key]
        saved_voice = saved_audio["voice"]
        render_spoken_avatar(
            saved_audio["bytes"], expression, f"Auri · {TTS_VOICES[saved_voice]}", key=f"audio_{key}"
        )


@st.cache_resource
def load_face_detector() -> cv2.CascadeClassifier:
    cascade_type = getattr(cv2, "CascadeClassifier", None)
    if cascade_type is None:
        cascade_type = getattr(getattr(cv2, "objdetect", None), "CascadeClassifier", None)
    if cascade_type is None:
        raise RuntimeError(
            "This OpenCV build does not include Haar CascadeClassifier. Install opencv-python-headless 4.x."
        )
    detector = cascade_type(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("OpenCV could not load its frontal face detector.")
    return detector


@st.cache_resource
def face_transform() -> transforms.Compose:
    return transforms.Compose(
        [transforms.Resize((FACE_SIZE, FACE_SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )


def detect_and_score_faces(image: Image.Image, model: FacialAffectNet) -> tuple[Image.Image, list[dict[str, Any]]]:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    boxes = load_face_detector().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
    )
    if len(boxes) == 0:
        return image.convert("RGB"), []

    boxes = sorted(boxes, key=lambda box: int(box[2]) * int(box[3]), reverse=True)[:8]
    face_images: list[Image.Image] = []
    for x, y, width, height in boxes:
        pad_x, pad_y = int(width * .15), int(height * .15)
        left, top = max(0, x - pad_x), max(0, y - pad_y)
        right, bottom = min(rgb.shape[1], x + width + pad_x), min(rgb.shape[0], y + height + pad_y)
        face_images.append(Image.fromarray(rgb[top:bottom, left:right]))

    batch = torch.stack([face_transform()(face) for face in face_images]).to(next(model.parameters()).device)
    with torch.inference_mode():
        output = model(batch)
        probabilities = torch.softmax(output["expression"], dim=1).cpu().numpy()

    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("segoeuib.ttf", max(15, int(annotated.width / 60)))
    except OSError:
        font = ImageFont.load_default()
    results: list[dict[str, Any]] = []
    for box, scores in zip(boxes, probabilities):
        index = int(scores.argmax())
        label = EXPRESSION_CLASSES[index]
        x, y, width, height = [int(value) for value in box]
        color_hex = EMOTION_COLORS.get(label, "#91f5d1")
        color_rgb = tuple(int(color_hex[i : i + 2].replace("#", ""), 16) for i in (1, 3, 5))
        draw.rounded_rectangle((x, y, x + width, y + height), radius=8, outline=color_rgb, width=4)
        tag = f"{label.title()}  {scores[index] * 100:.0f}%"
        bounds = draw.textbbox((0, 0), tag, font=font)
        tag_width, tag_height = bounds[2] - bounds[0] + 18, bounds[3] - bounds[1] + 14
        tag_y = max(0, y - tag_height - 4)
        draw.rounded_rectangle((x, tag_y, x + tag_width, tag_y + tag_height), radius=9, fill=(17, 22, 36))
        draw.text((x + 9, tag_y + 5), tag, font=font, fill=color_rgb)
        results.append({"label": label, "score": float(scores[index]), "scores": scores, "box": tuple(int(v) for v in box)})
    return annotated, results


def render_face_result(result: dict[str, Any], index: int = 1) -> None:
    label = result["label"]
    score = result["score"]
    st.markdown(
        f"<div class='result-card'><div class='tiny'>Face {index} · leading expression cue</div><div class='mascot-row' style='margin-top:.55rem'>{avatar_svg(label,76)}<div><div class='mascot-label'>{label}</div><div class='muted'>Model score · {score * 100:.1f}%</div></div></div></div>",
        unsafe_allow_html=True,
    )
    scores = result["scores"]
    order = np.argsort(scores)[::-1]
    for emotion_index in order:
        name = EXPRESSION_CLASSES[int(emotion_index)]
        value = float(scores[emotion_index])
        st.markdown(
            f"<div class='emotion-line'><span class='emotion-name'>{EMOJI[name]}&nbsp;&nbsp;{name}</span><span class='emotion-value'>{value * 100:.1f}%</span></div>",
            unsafe_allow_html=True,
        )
        st.progress(value, text=None)


def face_studio() -> None:
    st.markdown("<div class='section-kicker'>Face expression lab</div>", unsafe_allow_html=True)
    st.markdown("### Choose how you want to scan")
    st.caption("Go live for continuous cues, capture a moment, or upload a photo. AURA tracks up to eight visible faces in an image.")
    mode = st.radio(
        "Face input mode",
        ["Live camera", "Camera snapshot", "Upload image"],
        horizontal=True,
        label_visibility="collapsed",
        key="face_mode",
    )
    if not CHECKPOINT_PATH.exists():
        st.warning("The trained face checkpoint is missing. Expected artifacts/best.pt.")
        return
    try:
        model, checkpoint = load_face_model(str(CHECKPOINT_PATH))
    except Exception as exc:
        st.error(f"I couldn’t load the local face model: {exc}")
        return
    if "expression" not in set(checkpoint.get("trained_heads", [])):
        st.warning("This checkpoint has no trained expression head.")
        return

    if mode == "Live camera":
        live_camera(model)
        return
    if mode == "Camera snapshot":
        st.markdown("<div class='voice-box'><b>Freeze the frame.</b> Take a snapshot when you’re ready to read the expression cue.</div>", unsafe_allow_html=True)
        uploaded = st.camera_input("Take a snapshot", key="face_camera_snapshot")
    else:
        uploaded = st.file_uploader(
            "Drop in a face photo",
            type=["jpg", "jpeg", "png", "webp"],
            key="face_image_upload",
            help="JPG, PNG, or WebP images. Face processing stays on this machine.",
        )
    if uploaded is None:
        return
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as exc:
        st.error(f"That image could not be opened: {exc}")
        return
    with st.spinner("Reading the frame…"):
        try:
            annotated, results = detect_and_score_faces(image, model)
        except Exception as exc:
            st.error(f"Face analysis failed: {exc}")
            return
    if not results:
        st.image(image, caption="Uploaded frame", width="stretch")
        st.warning("I couldn’t find a clear, forward-facing face. Try a brighter image with the face closer to the camera.")
        return
    left, right = st.columns([1.1, .9], gap="large")
    with left:
        st.image(annotated, caption="Face detection overlay", width="stretch")
        st.caption(f"Detected {len(results)} face{'s' if len(results) != 1 else ''}. The largest face is shown first.")
    with right:
        render_face_result(results[0], 1)
        summary, expression = face_summary(results)
        st.session_state["auri_mood"] = expression
        st.markdown("<div class='section-kicker'>Auri’s four-line read</div>", unsafe_allow_html=True)
        st.markdown("\n".join(f"- {line}" for line in summary))
        render_speak_button(summary, expression, "face_summary")
        if len(results) > 1:
            with st.expander(f"See {len(results) - 1} other face result{'s' if len(results) > 2 else ''}"):
                for index, result in enumerate(results[1:], start=2):
                    render_face_result(result, index)
    st.markdown(
        "<div class='footer-note'>These are visual expression cues from a model trained on synthetic faces. They do not reveal a person’s actual emotional state, and real-world performance may differ.</div>",
        unsafe_allow_html=True,
    )


def live_camera(model: FacialAffectNet) -> None:
    try:
        from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
    except ImportError:
        st.error("Live camera needs the optional streamlit-webrtc and av packages. Install the updated requirements.txt and restart the app.")
        return

    device = next(model.parameters()).device
    detector = load_face_detector()
    transform = face_transform()

    class AffectVideoProcessor(VideoProcessorBase):
        def __init__(self) -> None:
            self.frame_count = 0
            self.last_results: list[tuple[tuple[int, int, int, int], str, float]] = []
            self.recorded_frames: deque[np.ndarray] = deque(maxlen=64)
            self.record_lock = threading.Lock()

        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            if self.frame_count % 6 == 0 or self.frame_count == 1:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                boxes = detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
                )
                boxes = sorted(boxes, key=lambda box: int(box[2]) * int(box[3]), reverse=True)[:4]
                if len(boxes):
                    crops = []
                    for x, y, width, height in boxes:
                        pad_x, pad_y = int(width * .15), int(height * .15)
                        left, top = max(0, x - pad_x), max(0, y - pad_y)
                        right, bottom = min(image.shape[1], x + width + pad_x), min(image.shape[0], y + height + pad_y)
                        crop = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2RGB)
                        crops.append(transform(Image.fromarray(crop)))
                    batch = torch.stack(crops).to(device)
                    with torch.inference_mode():
                        scores = torch.softmax(model(batch)["expression"], dim=1).cpu().numpy()
                    self.last_results = [
                        ((int(box[0]), int(box[1]), int(box[2]), int(box[3])), EXPRESSION_CLASSES[int(score.argmax())], float(score.max()))
                        for box, score in zip(boxes, scores)
                    ]
                else:
                    self.last_results = []
            for (x, y, width, height), label, score in self.last_results:
                color = tuple(int(EMOTION_COLORS[label][i : i + 2].replace("#", ""), 16) for i in (1, 3, 5))
                cv2.rectangle(image, (x, y), (x + width, y + height), color, 3, cv2.LINE_AA)
                text = f"{label.title()}  {score * 100:.0f}%"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = max(.55, image.shape[1] / 1100)
                (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, 2)
                top = max(0, y - text_height - 15)
                cv2.rectangle(image, (x, top), (x + text_width + 16, top + text_height + baseline + 12), (18, 23, 38), -1)
                cv2.putText(image, text, (x + 8, top + text_height + 3), font, scale, color, 2, cv2.LINE_AA)
            if self.frame_count % 4 == 0:
                recorded = cv2.resize(image, (480, 360), interpolation=cv2.INTER_AREA)
                with self.record_lock:
                    self.recorded_frames.append(recorded.copy())
            return type(frame).from_ndarray(image, format="bgr24")

    st.markdown(
        "<div class='voice-box'><b>Live expression cues</b><br><span class='muted'>Allow camera access and look toward the camera. The largest visible face is tracked continuously.</span></div>",
        unsafe_allow_html=True,
    )
    context = webrtc_streamer(
        key="aura-live-face",
        video_processor_factory=AffectVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.caption("Click START and allow camera access. The live overlay refreshes every few frames; a rolling 8-second clip buffer stays on this machine.")
    if st.button("Save the last 8 seconds", key="save_live_clip", width="content"):
        processor = getattr(context, "video_processor", None)
        if processor is None:
            st.warning("Start the camera and wait a few seconds before saving a clip.")
        else:
            with processor.record_lock:
                frames = [frame.copy() for frame in processor.recorded_frames]
            if len(frames) < 8:
                st.warning("Let the camera run for a few seconds, then save the clip.")
            else:
                temp_path: Path | None = None
                writer = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                        temp_path = Path(temp_file.name)
                    writer = cv2.VideoWriter(
                        str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (480, 360)
                    )
                    if not writer.isOpened():
                        raise RuntimeError("This OpenCV build cannot encode an MP4 clip.")
                    for recorded_frame in frames:
                        writer.write(recorded_frame)
                    writer.release()
                    writer = None
                    clip = temp_path.read_bytes()
                    st.download_button(
                        "Download expression clip",
                        data=clip,
                        file_name="aura-expression-clip.mp4",
                        mime="video/mp4",
                        key="download_live_clip",
                    )
                except Exception as exc:
                    st.error(f"Could not create the video clip: {exc}")
                finally:
                    if writer is not None:
                        writer.release()
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)


def voice_input() -> str:
    if not SPEECH_COMPONENT_DIR.exists():
        st.warning("The microphone capture component is missing.")
        return ""
    value = SPEECH_CAPTURE(key="aura_voice_capture", default="", height=112)
    if isinstance(value, dict) and value.get("audioBase64"):
        try:
            audio_bytes = base64.b64decode(value["audioBase64"], validate=True)
            audio_digest = hashlib.sha256(audio_bytes).hexdigest()
            if audio_digest == st.session_state.get("_last_voice_audio_digest"):
                return ""
            st.session_state["_last_voice_audio_digest"] = audio_digest
            with st.spinner("Transcribing on this device with Whisper…"):
                return transcribe_local_audio(audio_bytes)
        except Exception as exc:
            st.error(f"Local speech-to-text failed: {exc}")
            return ""
    if isinstance(value, str):
        return value
    return ""


def analyze_text(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    tokenizer, model = load_text_model(str(TEXT_MODEL_DIR))
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=96)
    encoded = {name: value.to(next(model.parameters()).device) for name, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits[0]
    probabilities = torch.sigmoid(logits).float().cpu().numpy()
    labels = [
        model.config.id2label.get(index, model.config.id2label.get(str(index), str(index)))
        for index in range(len(probabilities))
    ]
    thresholds = json.loads(TEXT_THRESHOLDS_PATH.read_text(encoding="utf-8")) if TEXT_THRESHOLDS_PATH.exists() else {}
    rows = [
        {
            "label": labels[index],
            "score": float(probabilities[index]),
            "threshold": float(thresholds.get(labels[index], 0.5)),
            "selected": bool(probabilities[index] >= float(thresholds.get(labels[index], 0.5))),
        }
        for index in range(len(labels))
    ]
    return sorted(rows, key=lambda row: row["score"], reverse=True), [row["label"] for row in rows if row["selected"]]


def render_text_results(rows: list[dict[str, Any]], selected: list[str]) -> None:
    top = rows[0]
    selected_text = ", ".join(label.replace("_", " ") for label in selected[:4]) if selected else "No label crossed its tuned threshold"
    summary, expression = text_summary(rows, selected)
    st.session_state["auri_mood"] = expression
    st.markdown(
        f"<div class='result-card'><div class='tiny'>AURA’s language read</div><div class='mascot-row' style='margin-top:.55rem'>{avatar_svg(expression,76)}<div><div class='mascot-label' style='text-transform:capitalize'>{top['label'].replace('_', ' ')}</div><div class='muted'>Top model score · {top['score'] * 100:.1f}%</div></div></div><div class='muted'>Threshold-passing cues: {selected_text}</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='section-kicker'>Auri’s four-line read</div>", unsafe_allow_html=True)
    st.markdown("\n".join(f"- {line}" for line in summary))
    render_speak_button(summary, expression, "text_summary")
    st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
    for row in rows[:10]:
        status = "selected" if row["selected"] else "below threshold"
        st.markdown(
            f"<div class='emotion-line'><span class='emotion-name'>{row['label'].replace('_', ' ')}</span><span class='emotion-value'>{row['score'] * 100:.1f}% · {status}</span></div>",
            unsafe_allow_html=True,
        )
        st.progress(row["score"], text=None)


def text_studio() -> None:
    st.markdown("<div class='section-kicker'>Text emotion lab</div>", unsafe_allow_html=True)
    st.markdown("### Say it, or write it")
    st.caption("Your words are scored against 28 GoEmotions labels. Several emotion labels can be active at once.")
    if not (TEXT_MODEL_DIR / "config.json").exists():
        st.warning("The trained text checkpoint is missing. Expected artifacts/text_model/model.")
        return
    spoken = voice_input()
    if spoken and spoken != st.session_state.get("_last_voice_transcript", ""):
        st.session_state["text_to_analyze"] = spoken
        st.session_state["_last_voice_transcript"] = spoken
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.markdown("<div class='section-kicker'>Your message</div>", unsafe_allow_html=True)
        text = st.text_area(
            "Text to analyze",
            key="text_to_analyze",
            height=172,
            max_chars=2000,
            placeholder="Type a thought, a sentence, or a short message…\n\nOr click Start recording above, speak at your own pace, then click Stop & transcribe.",
            label_visibility="collapsed",
        )
        analyze = st.button("✦  Read the emotion signals", type="primary", width="stretch", key="analyze_text")
        st.markdown(
            "<div class='footer-note'>Voice dictation records until you choose Stop & transcribe. The audio is transcribed with the local Whisper model on this device.</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("<div class='section-kicker'>Signal map</div>", unsafe_allow_html=True)
        if analyze:
            if not text.strip():
                st.warning("Write or dictate a message first.")
            else:
                try:
                    with st.spinner("Listening to the language…"):
                        rows, selected = analyze_text(text.strip())
                    st.session_state["text_result"] = {
                        "source": text.strip(),
                        "rows": rows,
                        "selected": selected,
                    }
                except Exception as exc:
                    st.error(f"I couldn’t load or run the local text model: {exc}")
        stored_result = st.session_state.get("text_result")
        if stored_result and stored_result["source"] == text.strip():
            render_text_results(stored_result["rows"], stored_result["selected"])
        else:
            st.markdown(
                "<div class='result-card' style='min-height:280px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center'><div class='mascot-face' style='width:92px;height:92px;font-size:3.2rem'>🫧</div><div class='mascot-label' style='margin-top:1rem'>Waiting for your words</div><div class='muted' style='max-width:270px;margin-top:.45rem'>Start with a sentence. AURA will surface the strongest emotion signals and their model scores.</div></div>",
                unsafe_allow_html=True,
            )


def main() -> None:
    inject_styles()
    render_shell()
    face, text = st.tabs(["◉  Face studio", "✧  Text studio"])
    with face:
        face_studio()
    with text:
        text_studio()


if __name__ == "__main__":
    main()
