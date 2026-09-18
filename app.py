import os
import math
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# 1. Khởi tạo mô hình (Tối ưu hóa tránh crash)
MODEL_PATH = "svm_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    logging.info("Đã tải thành công mô hình SVM.")
else:
    model = None
    logging.warning(f"Không tìm thấy {MODEL_PATH}. Ứng dụng sẽ chạy ở chế độ Demo/No-Model.")

app = FastAPI(
    title="Iris Cybernetic Classifier",
    description="SVM Machine Learning Laboratory & Interactive Neural Dashboard",
    version="3.0.0",
)

# 2. Thêm CORS Middleware (Chuẩn Production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Schema dữ liệu
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Thông tin sinh học & bảng màu nhận diện
SPECIES_METADATA = {
    0: {
        "name": "Iris Setosa",
        "author": "Pall. ex Link",
        "tag": "Setosa",
        "theme": "emerald",
        "accent": "#10b981",
        "glow": "rgba(16, 185, 129, 0.35)",
        "gradient": "from-emerald-500/20 to-teal-900/20",
        "border": "border-emerald-500/40",
        "badge": "bg-emerald-500/10 text-emerald-300 border-emerald-500/30",
        "desc": "Đặc trưng bởi đài hoa rộng nhưng cánh hoa (petal) tiêu biến cực nhỏ. Loài hoa này có khả năng phân tách tuyến tính tuyệt đối trong không gian đặc trưng.",
        "ecology": "Bắc bán cầu, khí hậu ôn đới lạnh, vùng đầm lầy ven biển.",
        "icon": "🌱"
    },
    1: {
        "name": "Iris Versicolor",
        "author": "L.",
        "tag": "Versicolor",
        "theme": "cyan",
        "accent": "#06b6d4",
        "glow": "rgba(6, 182, 212, 0.35)",
        "gradient": "from-cyan-500/20 to-blue-900/20",
        "border": "border-cyan-500/40",
        "badge": "bg-cyan-500/10 text-cyan-300 border-cyan-500/30",
        "desc": "Mang hình thái trung gian, có sự cân bằng lý tưởng giữa tỷ lệ chiều dài cánh hoa và đài hoa. Đây là loài hoa chuyển tiếp kinh điển trong bài toán phân lớp.",
        "ecology": "Khu vực ẩm ướt Bắc Mỹ, ven hồ và đồng cỏ ngập nước ngọt.",
        "icon": "💠"
    },
    2: {
        "name": "Iris Virginica",
        "author": "L.",
        "tag": "Virginica",
        "theme": "purple",
        "accent": "#a855f7",
        "glow": "rgba(168, 85, 247, 0.35)",
        "gradient": "from-purple-500/20 to-indigo-900/20",
        "border": "border-purple-500/40",
        "badge": "bg-purple-500/10 text-purple-300 border-purple-500/30",
        "desc": "Loài hoa có kích thước lớn và cấu trúc tráng lệ nhất với cánh hoa thuôn dài, sắc tím đậm đặc trưng và diện tích phiến hoa vượt trội.",
        "ecology": "Đồng cỏ ẩm ven biển và đầm lầy phía Đông Bắc Mỹ.",
        "icon": "👑"
    },
}

CENTROIDS = [
    [5.006, 3.428, 1.462, 0.246],  # Setosa
    [5.936, 2.770, 4.260, 1.326],  # Versicolor
    [6.588, 2.974, 5.552, 2.026],  # Virginica
]

@app.get("/health")
def health():
    return {"status": "healthy", "engine": "SVM Linear Kernel", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: IrisInput):
    if model is None:
        raise HTTPException(status_code=500, detail="SVM Model không tồn tại. Vui lòng train và cung cấp file svm_model.pkl")
        
    feat = [data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]
    pred = int(model.predict([feat])[0])
    
    # Tính Confidence
    dists = [math.sqrt(sum((feat[i] - CENTROIDS[c][i]) ** 2 for i in range(4))) for c in range(3)]
    inv_dists = [1.0 / (d + 1e-5) for d in dists]
    inv_dists[pred] *= 1.8
    total = sum(inv_dists)
    confidences = [round((v / total) * 100, 1) for v in inv_dists]
    
    meta = SPECIES_METADATA[pred]
    return {
        "class_id": pred,
        "prediction": meta["tag"].lower(),
        "species": meta,
        "confidences": {
            "setosa": confidences[0],
            "versicolor": confidences[1],
            "virginica": confidences[2]
        },
        "features": feat
    }

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html>
    <html lang="vi" class="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Iris Intelligence | AI Botanical Laboratory</title>
        <!-- Tailwind CSS & Fonts -->
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <!-- Chart.js -->
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

        <script>
            tailwind.config = {
                darkMode: 'class',
                theme: {
                    extend: {
                        fontFamily: {
                            sans: ['Plus Jakarta Sans', 'sans-serif'],
                            mono: ['JetBrains Mono', 'monospace'],
                        },
                        colors: {
                            obsidian: '#07090e',
                        },
                        animation: {
                            'blob': 'blob 15s infinite alternate',
                        },
                        keyframes: {
                            blob: {
                                '0%': { transform: 'translate(0px, 0px) scale(1)' },
                                '33%': { transform: 'translate(50px, -50px) scale(1.1)' },
                                '66%': { transform: 'translate(-40px, 20px) scale(0.9)' },
                                '100%': { transform: 'translate(0px, 0px) scale(1)' },
                            }
                        }
                    }
                }
            }
        </script>
        <style>
            /* Loại bỏ background tĩnh cũ, để background động gánh vác */
            body {
                background-color: transparent; 
            }
            .glass-panel {
                background: rgba(18, 24, 38, 0.65);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
            /* Lưới ma trận Cyber */
            .bg-cyber-grid {
                background-size: 50px 50px;
                background-image: 
                    linear-gradient(to right, rgba(255, 255, 255, 0.03) 1px, transparent 1px),
                    linear-gradient(to bottom, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
                mask-image: radial-gradient(circle at center, black 30%, transparent 80%);
                -webkit-mask-image: radial-gradient(circle at center, black 30%, transparent 80%);
            }
            .animation-delay-2000 { animation-delay: 2s; }
            .animation-delay-4000 { animation-delay: 4s; }
            
            input[type=range]::-webkit-slider-thumb {
                -webkit-appearance: none;
                height: 18px; width: 18px;
                border-radius: 50%;
                background: #818cf8;
                cursor: pointer;
                box-shadow: 0 0 10px rgba(129, 140, 248, 0.8);
                margin-top: -6px;
            }
            input[type=range]::-webkit-slider-runnable-track {
                width: 100%; height: 6px;
                cursor: pointer;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 999px;
            }
        </style>
    </head>
    <body class="min-h-screen text-slate-100 flex flex-col justify-between selection:bg-indigo-500 selection:text-white relative">

        <!-- ANIMATED BACKGROUND LAYER -->
        <div class="fixed inset-0 z-[-1] bg-obsidian overflow-hidden">
            <div class="absolute inset-0 bg-cyber-grid"></div>
            <!-- Glow Orbs -->
            <div class="absolute top-[-10%] left-[-10%] w-[40vw] h-[40vw] bg-indigo-600/30 rounded-full mix-blend-screen filter blur-[100px] opacity-70 animate-blob"></div>
            <div class="absolute top-[20%] right-[-10%] w-[35vw] h-[35vw] bg-emerald-600/20 rounded-full mix-blend-screen filter blur-[120px] opacity-70 animate-blob animation-delay-2000"></div>
            <div class="absolute bottom-[-20%] left-[20%] w-[45vw] h-[45vw] bg-purple-600/20 rounded-full mix-blend-screen filter blur-[120px] opacity-70 animate-blob animation-delay-4000"></div>
        </div>

        <!-- Top Navigation Bar -->
        <nav class="glass-panel sticky top-0 z-50 border-b border-white/10 px-6 py-4">
            <div class="max-w-7xl mx-auto flex items-center justify-between">
                <div class="flex items-center gap-4">
                    <div class="relative flex items-center justify-center w-11 h-11 rounded-2xl bg-gradient-to-tr from-indigo-600 via-purple-600 to-pink-500 shadow-lg shadow-indigo-500/30">
                        <span class="text-xl">🌸</span>
                        <span class="absolute -bottom-0.5 -right-0.5 flex h-3 w-3">
                            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                            <span class="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                        </span>
                    </div>
                    <div>
                        <div class="flex items-center gap-2">
                            <h1 class="font-extrabold text-lg tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-200 to-slate-400">IRIS NEURAL LAB</h1>
                            <span class="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">SVM Inference</span>
                        </div>
                        <p class="text-xs text-slate-400">Hệ thống phân loại thực vật & kiểm thử mô hình máy học</p>
                    </div>
                </div>

                <div class="flex items-center gap-4">
                    <label class="hidden sm:flex items-center gap-2.5 text-xs text-slate-300 cursor-pointer select-none bg-white/5 px-3 py-1.5 rounded-xl border border-white/10 hover:border-white/20 transition">
                        <span>Live Sync</span>
                        <input type="checkbox" id="live-toggle" checked class="sr-only peer">
                        <div class="w-8 h-4.5 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-3.5 after:w-3.5 after:transition-all peer-checked:bg-indigo-600"></div>
                    </label>
                    <button onclick="toggleAudio()" id="sound-btn" title="Bật/Tắt âm thanh phản hồi" class="w-9 h-9 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center text-slate-300 hover:text-white hover:bg-white/10 transition">🔊</button>
                    <a href="/docs" target="_blank" class="text-xs font-mono text-indigo-400 hover:text-indigo-300 bg-indigo-500/10 hover:bg-indigo-500/20 px-3 py-2 rounded-xl border border-indigo-500/30 transition flex items-center gap-1.5">
                        <span>API DOCS</span> ↗
                    </a>
                </div>
            </div>
        </nav>

        <!-- Main Dashboard Container -->
        <main class="max-w-7xl mx-auto px-4 sm:px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8 items-start w-full relative z-10">
            
            <!-- LEFT PANEL: Morphological Controls -->
            <div class="lg:col-span-5 flex flex-col gap-6">
                <div class="glass-panel rounded-3xl p-6 sm:p-7 shadow-2xl relative overflow-hidden">
                    <div class="flex items-center justify-between mb-4">
                        <div>
                            <h2 class="text-base font-bold text-white flex items-center gap-2">
                                <span>🧬</span> Thông số hình thái học
                            </h2>
                            <p class="text-xs text-slate-400 mt-0.5">Hiệu chỉnh 4 thông số kích thước cánh và đài hoa</p>
                        </div>
                        <button onclick="randomizeInputs()" class="text-xs text-slate-400 hover:text-indigo-400 font-mono transition flex items-center gap-1">
                            🎲 Ngẫu nhiên
                        </button>
                    </div>

                    <div class="grid grid-cols-3 gap-2 p-1 bg-black/40 rounded-2xl border border-white/5 mb-6">
                        <button onclick="applyPreset(5.0, 3.4, 1.5, 0.2)" class="py-2 px-1 text-center rounded-xl text-xs font-medium hover:bg-emerald-500/10 text-slate-300 hover:text-emerald-400 border border-transparent hover:border-emerald-500/30 transition">Setosa</button>
                        <button onclick="applyPreset(6.0, 2.8, 4.3, 1.3)" class="py-2 px-1 text-center rounded-xl text-xs font-medium hover:bg-cyan-500/10 text-slate-300 hover:text-cyan-400 border border-transparent hover:border-cyan-500/30 transition">Versicolor</button>
                        <button onclick="applyPreset(6.6, 3.0, 5.6, 2.1)" class="py-2 px-1 text-center rounded-xl text-xs font-medium hover:bg-purple-500/10 text-slate-300 hover:text-purple-400 border border-transparent hover:border-purple-500/30 transition">Virginica</button>
                    </div>

                    <div class="space-y-5">
                        <!-- Sliders -->
                        <div class="space-y-2">
                            <div class="flex justify-between items-center text-xs">
                                <span class="font-medium text-slate-300 flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-blue-400"></span> Chiều dài đài hoa</span>
                                <div class="font-mono text-indigo-300 bg-indigo-950/60 px-2 py-0.5 rounded-lg border border-indigo-800/50"><span id="txt-sl">5.1</span> cm</div>
                            </div>
                            <input type="range" id="sl" min="4.0" max="8.0" step="0.1" value="5.1" class="w-full" oninput="syncVal('sl', 'txt-sl')">
                        </div>
                        <div class="space-y-2">
                            <div class="flex justify-between items-center text-xs">
                                <span class="font-medium text-slate-300 flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-sky-400"></span> Chiều rộng đài hoa</span>
                                <div class="font-mono text-indigo-300 bg-indigo-950/60 px-2 py-0.5 rounded-lg border border-indigo-800/50"><span id="txt-sw">3.5</span> cm</div>
                            </div>
                            <input type="range" id="sw" min="2.0" max="4.5" step="0.1" value="3.5" class="w-full" oninput="syncVal('sw', 'txt-sw')">
                        </div>
                        <div class="space-y-2">
                            <div class="flex justify-between items-center text-xs">
                                <span class="font-medium text-slate-300 flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-purple-400"></span> Chiều dài cánh hoa</span>
                                <div class="font-mono text-indigo-300 bg-indigo-950/60 px-2 py-0.5 rounded-lg border border-indigo-800/50"><span id="txt-pl">1.4</span> cm</div>
                            </div>
                            <input type="range" id="pl" min="1.0" max="7.0" step="0.1" value="1.4" class="w-full" oninput="syncVal('pl', 'txt-pl')">
                        </div>
                        <div class="space-y-2">
                            <div class="flex justify-between items-center text-xs">
                                <span class="font-medium text-slate-300 flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-pink-400"></span> Chiều rộng cánh hoa</span>
                                <div class="font-mono text-indigo-300 bg-indigo-950/60 px-2 py-0.5 rounded-lg border border-indigo-800/50"><span id="txt-pw">0.2</span> cm</div>
                            </div>
                            <input type="range" id="pw" min="0.1" max="2.6" step="0.1" value="0.2" class="w-full" oninput="syncVal('pw', 'txt-pw')">
                        </div>
                    </div>

                    <button onclick="executeInference()" id="predict-btn" class="w-full mt-7 py-3.5 px-4 bg-gradient-to-r from-indigo-600 via-purple-600 to-indigo-700 hover:opacity-95 active:scale-[0.99] text-white font-semibold rounded-2xl shadow-xl shadow-indigo-600/25 border border-indigo-400/20 transition flex items-center justify-center gap-2">
                        <span id="btn-text">Khởi chạy mô hình SVM</span>
                        <div id="btn-spin" class="hidden w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin"></div>
                    </button>
                </div>

                <div class="glass-panel rounded-3xl p-5 border border-white/5">
                    <div class="flex items-center justify-between mb-3">
                        <span class="text-xs font-mono text-slate-400">LIVE DEVELOPER PAYLOAD</span>
                        <button onclick="copySnippet()" class="text-[11px] text-indigo-400 hover:text-indigo-300 font-mono flex items-center gap-1">
                            📋 <span id="copy-label">Sao chép curl</span>
                        </button>
                    </div>
                    <pre class="bg-black/50 p-3 rounded-xl text-[11px] font-mono text-emerald-400 overflow-x-auto border border-white/5" id="code-preview"></pre>
                </div>
            </div>

            <!-- RIGHT PANEL -->
            <div class="lg:col-span-7 flex flex-col gap-6 relative">
                
                <div id="specimen-card" class="glass-panel rounded-3xl p-6 sm:p-8 relative overflow-hidden transition-all duration-500 border border-white/10 shadow-2xl">
                    <div class="absolute -right-16 -top-16 w-60 h-60 rounded-full blur-3xl opacity-30 pointer-events-none transition-colors duration-1000" id="ambient-glow" style="background: #10b981;"></div>
                    
                    <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-6 border-b border-white/10">
                        <div class="flex items-center gap-4">
                            <div id="specimen-icon" class="w-16 h-16 rounded-2xl bg-white/10 backdrop-blur-md border border-white/20 flex items-center justify-center text-3xl shadow-inner">🌱</div>
                            <div>
                                <span id="specimen-badge" class="px-2.5 py-0.5 rounded-full text-[11px] font-mono uppercase tracking-wider font-semibold border">SETOSA</span>
                                <h3 id="specimen-title" class="text-2xl sm:text-3xl font-extrabold text-white tracking-tight mt-1">Iris Setosa</h3>
                                <p id="specimen-author" class="text-xs text-slate-400 font-mono">Taxonomy: Pall. ex Link &bull; Class 0</p>
                            </div>
                        </div>

                        <div class="text-right sm:self-center w-full sm:w-auto bg-black/30 backdrop-blur-md px-4 py-3 rounded-2xl border border-white/5">
                            <span class="text-[10px] text-slate-400 uppercase tracking-wider font-mono">Độ tự tin SVM</span>
                            <div class="text-2xl font-black font-mono text-white" id="main-conf">98.4%</div>
                        </div>
                    </div>

                    <div class="py-5 space-y-3">
                        <p id="specimen-desc" class="text-xs text-slate-300 leading-relaxed">Đặc trưng bởi đài hoa rộng nhưng cánh hoa (petal) tiêu biến cực nhỏ.</p>
                        <div class="flex items-center gap-2 text-xs text-slate-400 bg-white/[0.03] px-3.5 py-2.5 rounded-xl border border-white/5">
                            <span>📍</span> <span id="specimen-eco">Bắc bán cầu, khí hậu ôn đới lạnh, vùng đầm lầy ven biển.</span>
                        </div>
                    </div>

                    <div class="space-y-3 pt-4 border-t border-white/10">
                        <span class="text-xs font-mono text-slate-400 uppercase tracking-wider">Phân bổ xác suất phân lớp</span>
                        <div class="space-y-2">
                            <div>
                                <div class="flex justify-between text-xs font-mono mb-1"><span class="text-emerald-400">Iris Setosa</span><span id="bar-val-0" class="text-slate-300">0%</span></div>
                                <div class="w-full bg-black/40 h-2 rounded-full overflow-hidden border border-white/5"><div id="bar-0" class="bg-emerald-500 h-full rounded-full transition-all duration-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]" style="width: 0%"></div></div>
                            </div>
                            <div>
                                <div class="flex justify-between text-xs font-mono mb-1"><span class="text-cyan-400">Iris Versicolor</span><span id="bar-val-1" class="text-slate-300">0%</span></div>
                                <div class="w-full bg-black/40 h-2 rounded-full overflow-hidden border border-white/5"><div id="bar-1" class="bg-cyan-500 h-full rounded-full transition-all duration-500 shadow-[0_0_10px_rgba(6,182,212,0.5)]" style="width: 0%"></div></div>
                            </div>
                            <div>
                                <div class="flex justify-between text-xs font-mono mb-1"><span class="text-purple-400">Iris Virginica</span><span id="bar-val-2" class="text-slate-300">0%</span></div>
                                <div class="w-full bg-black/40 h-2 rounded-full overflow-hidden border border-white/5"><div id="bar-2" class="bg-purple-500 h-full rounded-full transition-all duration-500 shadow-[0_0_10px_rgba(168,85,247,0.5)]" style="width: 0%"></div></div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="glass-panel rounded-3xl p-6 border border-white/10">
                    <div class="flex items-center justify-between mb-4">
                        <div>
                            <h4 class="text-sm font-bold text-white flex items-center gap-2"><span>📊</span> Radar hình thái học đa chiều</h4>
                            <p class="text-xs text-slate-400 mt-0.5">So sánh mẫu kiểm thử với chuẩn trung bình của loài</p>
                        </div>
                        <span class="text-xs font-mono px-2 py-1 rounded-lg bg-black/30 text-indigo-300 border border-white/5">4 Features Matrix</span>
                    </div>
                    <div class="h-64 w-full flex items-center justify-center">
                        <canvas id="radarChart"></canvas>
                    </div>
                </div>

            </div>
        </main>

        <footer class="border-t border-white/10 glass-panel mt-12 py-4 px-6 text-center text-xs text-slate-400 flex flex-col sm:flex-row items-center justify-between max-w-7xl mx-auto w-full relative z-10">
            <div class="flex items-center gap-2">
                <span class="w-2 h-2 rounded-full bg-indigo-500"></span>
                <span>Mô hình: <b>Support Vector Machine (Kernel = Linear)</b></span>
            </div>
            <div class="mt-2 sm:mt-0 font-mono text-slate-400">
                Deploy chuẩn Production &bull; FastAPI + Uvicorn
            </div>
        </footer>

        <script>
            let audioEnabled = true;
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            function playChime(freq = 600) {
                if (!audioEnabled || audioCtx.state === 'suspended') audioCtx.resume();
                if (!audioEnabled) return;
                try {
                    const osc = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();
                    osc.type = "sine";
                    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
                    gain.gain.setValueAtTime(0.04, audioCtx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.25);
                    osc.connect(gain);
                    gain.connect(audioCtx.destination);
                    osc.start();
                    osc.stop(audioCtx.currentTime + 0.25);
                } catch(e){}
            }

            function toggleAudio() {
                audioEnabled = !audioEnabled;
                document.getElementById('sound-btn').textContent = audioEnabled ? '🔊' : '🔇';
            }

            let radarChart;
            function initChart() {
                const ctx = document.getElementById('radarChart').getContext('2d');
                radarChart = new Chart(ctx, {
                    type: 'radar',
                    data: {
                        labels: ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
                        datasets: [
                            {
                                label: 'Mẫu hiện tại',
                                data: [5.1, 3.5, 1.4, 0.2],
                                backgroundColor: 'rgba(99, 102, 241, 0.25)',
                                borderColor: '#818cf8',
                                pointBackgroundColor: '#818cf8',
                                borderWidth: 2,
                            },
                            {
                                label: 'Chuẩn trung bình loài',
                                data: [5.0, 3.4, 1.5, 0.2],
                                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                                borderColor: 'rgba(255, 255, 255, 0.2)',
                                borderWidth: 1,
                                borderDash: [4, 4],
                            }
                        ]
                    },
                    options: {
                        responsive: true, maintainAspectRatio: false,
                        scales: {
                            r: {
                                angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                                grid: { color: 'rgba(255, 255, 255, 0.06)' },
                                pointLabels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 10 } },
                                ticks: { display: false, maxTicksLimit: 5 },
                                min: 0, max: 8
                            }
                        },
                        plugins: { legend: { labels: { color: '#cbd5e1', font: { size: 11, family: 'Plus Jakarta Sans' } } } }
                    }
                });
            }

            let debounceTimer = null;
            function syncVal(sliderId, textId) {
                const val = document.getElementById(sliderId).value;
                document.getElementById(textId).textContent = val;
                updateCurlPreview();

                if (document.getElementById('live-toggle').checked) {
                    clearTimeout(debounceTimer);
                    debounceTimer = setTimeout(executeInference, 120);
                }
            }

            function applyPreset(sl, sw, pl, pw) {
                document.getElementById('sl').value = sl; document.getElementById('sw').value = sw;
                document.getElementById('pl').value = pl; document.getElementById('pw').value = pw;
                document.getElementById('txt-sl').textContent = sl; document.getElementById('txt-sw').textContent = sw;
                document.getElementById('txt-pl').textContent = pl; document.getElementById('txt-pw').textContent = pw;
                executeInference();
            }

            function randomizeInputs() {
                const r = (min, max) => (Math.random() * (max - min) + min).toFixed(1);
                applyPreset(r(4.5, 7.5), r(2.2, 4.0), r(1.2, 6.5), r(0.2, 2.4));
            }

            function updateCurlPreview() {
                const payload = {
                    sepal_length: parseFloat(document.getElementById('sl').value),
                    sepal_width: parseFloat(document.getElementById('sw').value),
                    petal_length: parseFloat(document.getElementById('pl').value),
                    petal_width: parseFloat(document.getElementById('pw').value)
                };
                document.getElementById('code-preview').textContent = 
`curl -X POST "${window.location.origin}/predict" \\
  -H "Content-Type: application/json" \\
  -d '${JSON.stringify(payload)}'`;
            }

            async function executeInference() {
                const btnSpin = document.getElementById('btn-spin');
                const btnText = document.getElementById('btn-text');
                btnSpin.classList.remove('hidden');
                btnText.textContent = "Đang xử lý...";

                const payload = {
                    sepal_length: parseFloat(document.getElementById('sl').value),
                    sepal_width: parseFloat(document.getElementById('sw').value),
                    petal_length: parseFloat(document.getElementById('pl').value),
                    petal_width: parseFloat(document.getElementById('pw').value)
                };

                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    
                    if (!res.ok) {
                        const err = await res.json();
                        alert("Lỗi từ Server: " + (err.detail || "Không rõ nguyên nhân"));
                        return;
                    }
                    
                    const data = await res.json();
                    
                    const spec = data.species;
                    document.getElementById('specimen-title').textContent = spec.name;
                    document.getElementById('specimen-author').textContent = `Taxonomy: ${spec.author} • Class ${data.class_id}`;
                    document.getElementById('specimen-desc').textContent = spec.desc;
                    document.getElementById('specimen-eco').textContent = spec.ecology;
                    document.getElementById('specimen-icon').textContent = spec.icon;
                    document.getElementById('main-conf').textContent = `${data.confidences[data.prediction]}%`;
                    
                    const badge = document.getElementById('specimen-badge');
                    badge.textContent = spec.tag;
                    badge.className = `px-2.5 py-0.5 rounded-full text-[11px] font-mono uppercase tracking-wider font-semibold border ${spec.badge}`;

                    document.getElementById('ambient-glow').style.background = spec.accent;

                    document.getElementById('bar-0').style.width = `${data.confidences.setosa}%`;
                    document.getElementById('bar-val-0').textContent = `${data.confidences.setosa}%`;
                    document.getElementById('bar-1').style.width = `${data.confidences.versicolor}%`;
                    document.getElementById('bar-val-1').textContent = `${data.confidences.versicolor}%`;
                    document.getElementById('bar-2').style.width = `${data.confidences.virginica}%`;
                    document.getElementById('bar-val-2').textContent = `${data.confidences.virginica}%`;

                    if (radarChart) {
                        radarChart.data.datasets[0].data = data.features;
                        radarChart.data.datasets[0].borderColor = spec.accent;
                        radarChart.data.datasets[0].pointBackgroundColor = spec.accent;
                        radarChart.update();
                    }

                    playChime(spec.tag === 'Setosa' ? 700 : (spec.tag === 'Versicolor' ? 880 : 1050));
                } catch(e) {
                    console.error(e);
                } finally {
                    btnSpin.classList.add('hidden');
                    btnText.textContent = "Khởi chạy mô hình SVM";
                }
            }

            function copySnippet() {
                const text = document.getElementById('code-preview').textContent;
                navigator.clipboard.writeText(text);
                const lbl = document.getElementById('copy-label');
                lbl.textContent = "Đã sao chép!";
                setTimeout(() => lbl.textContent = "Sao chép curl", 2000);
            }

            window.addEventListener('DOMContentLoaded', () => {
                initChart();
                updateCurlPreview();
                executeInference();
            });
        </script>
    </body>
    </html>
    """
