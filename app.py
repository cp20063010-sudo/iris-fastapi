import os
import math
import logging
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# 1. Khởi tạo mô hình
MODEL_PATH = "svm_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    logging.info("Đã tải thành công mô hình SVM.")
else:
    model = None
    logging.warning(f"Không tìm thấy {MODEL_PATH}. Ứng dụng sẽ chạy ở chế độ Demo/No-Model.")

app = FastAPI(
    title="Iris Cyber-Lab Classifier",
    description="Cyberpunk Botanical Quality Control Dashboard",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API TRẢ VỀ ẢNH AN TOÀN
@app.get("/img/{img_name}")
def get_image(img_name: str):
    allowed_images = ["anh_setosa.jpg", "anh_versicolor.jpg", "anh_virginica.jpg"]
    if img_name in allowed_images and os.path.exists(img_name):
        return FileResponse(img_name)
    raise HTTPException(status_code=404, detail="SYS_ERR: Image Not Found in Local Directory")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Metadata cập nhật theo phong cách Cyberpunk/Neon
SPECIES_METADATA = {
    0: {
        "name": "Iris Setosa",
        "author": "Pall. ex Link",
        "tag": "SETOSA",
        "theme": "neon-green",
        "accent": "#00ff9d",
        "badge": "bg-[#00ff9d]/20 text-[#00ff9d] border-[#00ff9d]",
        "desc": "Loài có kích thước nhỏ gọn nhất. Cấu trúc đài hoa tiêu biến đặc thù, dễ dàng phân tách tuyến tính tuyệt đối khỏi các loài khác. Thường được ứng dụng trong chiết xuất hoạt chất nền.",
        "ecology": "Bắc bán cầu, đầm lầy ven biển vùng khí hậu ôn đới lạnh.",
        "image": "/img/anh_setosa.jpg" 
    },
    1: {
        "name": "Iris Versicolor",
        "author": "L.",
        "tag": "VERSICOLOR",
        "theme": "neon-cyan",
        "accent": "#00f3ff",
        "badge": "bg-[#00f3ff]/20 text-[#00f3ff] border-[#00f3ff]",
        "desc": "Hình thái trung gian, sở hữu sự cân bằng hoàn hảo giữa chiều dài và chiều rộng cánh hoa. Mã gen đa dạng, cung cấp nhiều thành phần thứ cấp tiềm năng.",
        "ecology": "Đồng cỏ ngập nước ngọt, khu vực ẩm ướt Bắc Mỹ.",
        "image": "/img/anh_versicolor.jpg"
    },
    2: {
        "name": "Iris Virginica",
        "author": "L.",
        "tag": "VIRGINICA",
        "theme": "neon-magenta",
        "accent": "#ff00fc",
        "badge": "bg-[#ff00fc]/20 text-[#ff00fc] border-[#ff00fc]",
        "desc": "Sinh khối lớn nhất trong 3 loài. Cánh hoa thuôn dài với sắc tím đậm đặc trưng. Cấu trúc màng tế bào dày, cần dung môi chiết xuất mạnh.",
        "ecology": "Đầm lầy hở và ven bờ biển phía Đông Bắc Mỹ.",
        "image": "/img/anh_virginica.jpg"
    },
}

CENTROIDS = [
    [5.006, 3.428, 1.462, 0.246],  # Setosa
    [5.936, 2.770, 4.260, 1.326],  # Versicolor
    [6.588, 2.974, 5.552, 2.026],  # Virginica
]

@app.post("/predict")
def predict(data: IrisInput):
    if model is None:
        raise HTTPException(status_code=500, detail="CRITICAL: SVM Model missing (svm_model.pkl).")
        
    feat = [data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]
    pred = int(model.predict([feat])[0])
    
    # Softmax Distance logic
    dists = [math.sqrt(sum((feat[i] - CENTROIDS[c][i]) ** 2 for i in range(4))) for c in range(3)]
    inv_dists = [1.0 / (d + 1e-5) for d in dists]
    inv_dists[pred] *= 1.8
    total = sum(inv_dists)
    confidences = [round((v / total) * 100, 1) for v in inv_dists]
    
    meta = SPECIES_METADATA[pred]
    return {
        "class_id": pred,
        "prediction": meta["tag"],
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
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cyber-Lab | Botanical QC Interface</title>
        <script src="https://cdn.tailwindcss.com"></script>
        
        <!-- Fonts sci-fi & monospace -->
        <link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;600;700&family=JetBrains+Mono:wght@300;400;700&display=swap" rel="stylesheet">
        
        <!-- Libraries -->
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>

        <style>
            :root {
                --c-bg: #050505; --c-panel: #0a0e17;
                --c-neon-cyan: #00f3ff; --c-neon-magenta: #ff00fc; --c-neon-green: #00ff9d;
            }
            body { 
                background-color: var(--c-bg); 
                color: #e2e8f0; 
                font-family: 'Chakra Petch', sans-serif; 
                background-image: 
                    linear-gradient(rgba(0, 243, 255, 0.03) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(0, 243, 255, 0.03) 1px, transparent 1px);
                background-size: 30px 30px;
                overflow-x: hidden;
            }
            .font-mono { font-family: 'JetBrains Mono', monospace; }
            
            /* CRT Scanline Effect */
            .scanlines {
                position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
                background: linear-gradient(to bottom, rgba(255,255,255,0), rgba(255,255,255,0) 50%, rgba(0,0,0,0.1) 50%, rgba(0,0,0,0.1));
                background-size: 100% 4px; pointer-events: none; z-index: 9999; opacity: 0.6;
            }

            .cyber-panel {
                background: rgba(10, 14, 23, 0.85);
                border: 1px solid rgba(0, 243, 255, 0.2);
                backdrop-filter: blur(10px);
                box-shadow: 0 0 15px rgba(0, 0, 0, 0.8), inset 0 0 20px rgba(0, 243, 255, 0.02);
            }
            
            /* Cyberpunk Button / Clip Path */
            .cyber-btn {
                clip-path: polygon(10px 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%, 0 10px);
                text-transform: uppercase; font-weight: bold; letter-spacing: 1px;
                transition: all 0.2s; position: relative;
            }
            .cyber-btn:hover { text-shadow: 0 0 8px currentColor; box-shadow: inset 0 0 15px currentColor; }
            
            /* Range Slider */
            input[type=range] { -webkit-appearance: none; background: transparent; width: 100%; height: 2px; }
            input[type=range]::-webkit-slider-runnable-track { width: 100%; height: 2px; background: rgba(255,255,255,0.2); }
            input[type=range]::-webkit-slider-thumb {
                -webkit-appearance: none; height: 16px; width: 8px; background: var(--c-neon-cyan);
                border-radius: 0; margin-top: -7px; cursor: crosshair; box-shadow: 0 0 10px var(--c-neon-cyan);
            }
            
            /* Scrollbar */
            ::-webkit-scrollbar { width: 6px; }
            ::-webkit-scrollbar-track { background: #000; }
            ::-webkit-scrollbar-thumb { background: rgba(0, 243, 255, 0.3); }
            ::-webkit-scrollbar-thumb:hover { background: var(--c-neon-cyan); }
            
            /* Glow Text */
            .glow-cyan { color: var(--c-neon-cyan); text-shadow: 0 0 10px var(--c-neon-cyan); }
            .glow-magenta { color: var(--c-neon-magenta); text-shadow: 0 0 10px var(--c-neon-magenta); }
            .glow-green { color: var(--c-neon-green); text-shadow: 0 0 10px var(--c-neon-green); }

            /* Grid Layout for Logs */
            table.cyber-table th { background: rgba(0,243,255,0.1); color: var(--c-neon-cyan); text-align: left; }
            table.cyber-table tr { border-bottom: 1px solid rgba(255,255,255,0.05); transition: 0.2s; }
            table.cyber-table tr:hover { background: rgba(255,255,255,0.02); }
        </style>
    </head>
    <body class="min-h-screen flex flex-col relative selection:bg-[#00f3ff] selection:text-black">
        <div class="scanlines"></div>

        <!-- TOP BAR -->
        <header class="border-b border-[#00f3ff]/30 bg-[#000]/80 sticky top-0 z-50 shadow-[0_0_20px_rgba(0,243,255,0.1)]">
            <div class="max-w-7xl mx-auto px-4 py-3 flex justify-between items-center">
                <div class="flex items-center gap-3">
                    <div class="w-8 h-8 bg-[#00f3ff] rounded flex items-center justify-center text-black font-bold text-xl shadow-[0_0_15px_#00f3ff] animate-pulse">☣</div>
                    <div>
                        <h1 class="font-bold text-xl tracking-widest text-white">SYS.<span class="glow-cyan">BOTANICA</span>_v4.0</h1>
                        <p class="text-[10px] text-[#00f3ff]/70 font-mono tracking-widest uppercase">Biological Quality Control Terminal</p>
                    </div>
                </div>
                <div class="flex gap-4 items-center">
                    <span class="text-xs font-mono text-green-400 flex items-center gap-2"><span class="w-2 h-2 bg-green-400 rounded-full animate-ping"></span> SERVER ONLINE</span>
                    <button onclick="toggleAudio()" id="sound-btn" class="cyber-btn border border-[#00f3ff]/50 text-[#00f3ff] px-3 py-1 text-xs hover:bg-[#00f3ff]/10">SND_ON</button>
                </div>
            </div>
        </header>

        <!-- MAIN TERMINAL -->
        <main class="max-w-7xl mx-auto px-4 py-6 grid grid-cols-1 lg:grid-cols-12 gap-6 w-full flex-grow relative z-10">
            
            <!-- LEFT: INPUT & BATCH MODULE -->
            <div class="lg:col-span-4 flex flex-col gap-6">
                
                <!-- MORPHOLOGICAL SCANNER -->
                <div class="cyber-panel p-5 relative border-l-4 border-l-[#00f3ff]">
                    <div class="flex justify-between items-end mb-4 border-b border-[#00f3ff]/20 pb-2">
                        <h2 class="text-sm font-bold glow-cyan tracking-wider">>> MANUAL_OVERRIDE</h2>
                        <button onclick="randomizeInputs()" class="text-[10px] font-mono text-[#00f3ff]/60 hover:text-[#00f3ff] uppercase transition">[ Randomize ]</button>
                    </div>

                    <div class="space-y-6">
                        <!-- SLIDERS -->
                        <div class="space-y-1">
                            <div class="flex justify-between text-[11px] font-mono text-gray-400"><span>Sepal Length [SL]</span><span id="txt-sl" class="text-white">5.1</span></div>
                            <input type="range" id="sl" min="4.0" max="8.0" step="0.1" value="5.1" oninput="syncVal('sl', 'txt-sl')">
                        </div>
                        <div class="space-y-1">
                            <div class="flex justify-between text-[11px] font-mono text-gray-400"><span>Sepal Width [SW]</span><span id="txt-sw" class="text-white">3.5</span></div>
                            <input type="range" id="sw" min="2.0" max="4.5" step="0.1" value="3.5" oninput="syncVal('sw', 'txt-sw')">
                        </div>
                        <div class="space-y-1">
                            <div class="flex justify-between text-[11px] font-mono text-gray-400"><span>Petal Length [PL]</span><span id="txt-pl" class="text-white">1.4</span></div>
                            <input type="range" id="pl" min="1.0" max="7.0" step="0.1" value="1.4" oninput="syncVal('pl', 'txt-pl')">
                        </div>
                        <div class="space-y-1">
                            <div class="flex justify-between text-[11px] font-mono text-gray-400"><span>Petal Width [PW]</span><span id="txt-pw" class="text-white">0.2</span></div>
                            <input type="range" id="pw" min="0.1" max="2.6" step="0.1" value="0.2" oninput="syncVal('pw', 'txt-pw')">
                        </div>
                    </div>

                    <button onclick="executeInference()" id="predict-btn" class="cyber-btn mt-6 w-full py-3 bg-[#00f3ff]/10 border border-[#00f3ff] text-[#00f3ff] hover:bg-[#00f3ff] hover:text-black flex justify-center items-center gap-2">
                        <span id="btn-text">INITIATE_SCAN()</span>
                        <div id="btn-spin" class="hidden w-3 h-3 border border-current border-t-transparent rounded-full animate-spin"></div>
                    </button>
                </div>

                <!-- BATCH ANALYSIS UPLOAD -->
                <div class="cyber-panel p-5 relative border-l-4 border-l-[#ff00fc]">
                    <h2 class="text-sm font-bold glow-magenta tracking-wider border-b border-[#ff00fc]/20 pb-2 mb-4">>> BATCH_PROTOCOL (CSV)</h2>
                    <p class="text-[10px] font-mono text-gray-400 mb-3">Format: 4 cột không tiêu đề (SL, SW, PL, PW). Giới hạn: 100 mẫu/lượt.</p>
                    
                    <label class="border border-dashed border-[#ff00fc]/40 p-4 flex flex-col items-center justify-center cursor-pointer hover:bg-[#ff00fc]/5 transition group">
                        <span class="text-xl mb-1 group-hover:glow-magenta transition">📁</span>
                        <span class="text-[11px] font-mono text-gray-300">UPLOAD_CSV</span>
                        <input type="file" id="csv-file" accept=".csv" class="hidden" onchange="handleCSVUpload(event)">
                    </label>
                </div>

            </div>

            <!-- RIGHT: HOLOGRAPHIC RESULT & RADAR -->
            <div class="lg:col-span-8 flex flex-col gap-6" id="report-area">
                
                <div class="cyber-panel p-1 relative border border-[#00f3ff]/30 shadow-[0_0_30px_rgba(0,243,255,0.05)]" id="card-border">
                    <div class="bg-[#050505]/90 p-5 sm:p-7 relative overflow-hidden" id="specimen-card">
                        
                        <!-- Watermark -->
                        <div class="absolute -right-10 -bottom-10 text-9xl opacity-5 font-black pointer-events-none" id="watermark">IRQ</div>

                        <div class="flex flex-col sm:flex-row justify-between gap-6 pb-5 border-b border-gray-800 relative z-10">
                            <!-- Image Hologram -->
                            <div class="flex items-start gap-5">
                                <div class="relative group">
                                    <div class="absolute -inset-1 bg-gradient-to-r from-transparent to-transparent blur opacity-50 group-hover:opacity-100 transition duration-500" id="holo-glow"></div>
                                    <img id="specimen-img" src="" class="w-24 h-24 sm:w-28 sm:h-28 object-cover border-2 border-gray-700 relative z-10 grayscale-[30%] contrast-125">
                                    <div class="absolute inset-0 bg-[#00f3ff]/10 mix-blend-overlay z-20 pointer-events-none" id="holo-tint"></div>
                                    <!-- Sci-fi brackets -->
                                    <div class="absolute top-0 left-0 w-2 h-2 border-t-2 border-l-2 border-white z-30"></div>
                                    <div class="absolute bottom-0 right-0 w-2 h-2 border-b-2 border-r-2 border-white z-30"></div>
                                </div>
                                
                                <div>
                                    <div class="text-[10px] font-mono text-gray-500 mb-1 tracking-widest">ID: MATCH_FOUND</div>
                                    <h3 id="specimen-title" class="text-3xl font-black uppercase tracking-wider text-white">AWAITING</h3>
                                    <p id="specimen-author" class="text-xs font-mono text-[#00f3ff] mt-1">TAXA: --</p>
                                    <span id="specimen-badge" class="inline-block mt-2 px-2 py-0.5 text-[10px] font-mono border border-gray-600 text-gray-400">STATUS: STANDBY</span>
                                </div>
                            </div>

                            <!-- Confidence Score -->
                            <div class="text-right sm:self-center">
                                <span class="text-[10px] font-mono text-gray-500 tracking-widest block">CONFIDENCE_LEVEL</span>
                                <div class="text-4xl sm:text-5xl font-light font-mono tracking-tighter" id="main-conf">00.0%</div>
                            </div>
                        </div>

                        <!-- Details & Radar Grid -->
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 pt-5 relative z-10">
                            <div class="space-y-4">
                                <div>
                                    <span class="text-[10px] font-mono text-gray-500 tracking-widest block mb-1">BIOLOGICAL_PROFILE</span>
                                    <p id="specimen-desc" class="text-sm text-gray-300 leading-relaxed min-h-[60px]"></p>
                                </div>
                                <div class="bg-black/50 border border-gray-800 p-3 flex gap-2">
                                    <span class="text-[#00f3ff] shrink-0">⌖</span>
                                    <span id="specimen-eco" class="text-xs text-gray-400 leading-relaxed font-mono"></span>
                                </div>
                                
                                <!-- Probability Bars -->
                                <div class="space-y-2 mt-4">
                                    <div class="relative pt-1">
                                        <div class="flex justify-between text-[9px] font-mono mb-1"><span class="text-[#00ff9d]">SETOSA</span><span id="bar-val-0">0%</span></div>
                                        <div class="w-full bg-gray-900 h-1"><div id="bar-0" class="bg-[#00ff9d] h-1" style="width: 0%"></div></div>
                                    </div>
                                    <div class="relative pt-1">
                                        <div class="flex justify-between text-[9px] font-mono mb-1"><span class="text-[#00f3ff]">VERSICOLOR</span><span id="bar-val-1">0%</span></div>
                                        <div class="w-full bg-gray-900 h-1"><div id="bar-1" class="bg-[#00f3ff] h-1" style="width: 0%"></div></div>
                                    </div>
                                    <div class="relative pt-1">
                                        <div class="flex justify-between text-[9px] font-mono mb-1"><span class="text-[#ff00fc]">VIRGINICA</span><span id="bar-val-2">0%</span></div>
                                        <div class="w-full bg-gray-900 h-1"><div id="bar-2" class="bg-[#ff00fc] h-1" style="width: 0%"></div></div>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Radar -->
                            <div class="h-48 w-full relative flex justify-center items-center">
                                <!-- Background crosshair -->
                                <div class="absolute inset-0 flex items-center justify-center opacity-20 pointer-events-none">
                                    <div class="w-full h-px bg-white"></div><div class="h-full w-px bg-white absolute"></div>
                                </div>
                                <canvas id="radarChart"></canvas>
                            </div>
                        </div>

                    </div>
                </div>

                <!-- Export Report Button -->
                <div class="flex justify-end">
                    <button onclick="downloadPDF()" class="cyber-btn bg-white/5 border border-white/20 text-white hover:bg-white hover:text-black px-4 py-2 text-xs flex gap-2 items-center">
                        🖨️ EXPORT_REPORT.PDF
                    </button>
                </div>
            </div>

            <!-- BOTTOM: DATABASE CORE & SYSTEM LOG -->
            <div class="lg:col-span-12 grid grid-cols-1 md:grid-cols-3 gap-6">
                
                <!-- SPECIES CODEX (Phân tích loại hoa đã chọn) -->
                <div class="cyber-panel p-5 col-span-1 border-t-2 border-t-[#00ff9d]">
                    <h2 class="text-sm font-bold glow-green tracking-wider mb-4 border-b border-[#00ff9d]/20 pb-2">>> SPECIES_CODEX (DATABASE)</h2>
                    <p class="text-[10px] font-mono text-gray-400 mb-3">Chọn mẫu chuẩn để nạp thông số và phân tích đặc trưng.</p>
                    <div class="flex flex-col gap-2">
                        <button onclick="loadSpeciesData(0)" class="cyber-btn bg-[#00ff9d]/10 border border-[#00ff9d]/30 text-[#00ff9d] hover:bg-[#00ff9d] hover:text-black py-2 text-xs text-left px-3">
                            [01] IRIS_SETOSA_PROFILE
                        </button>
                        <button onclick="loadSpeciesData(1)" class="cyber-btn bg-[#00f3ff]/10 border border-[#00f3ff]/30 text-[#00f3ff] hover:bg-[#00f3ff] hover:text-black py-2 text-xs text-left px-3">
                            [02] IRIS_VERSICOLOR_PROFILE
                        </button>
                        <button onclick="loadSpeciesData(2)" class="cyber-btn bg-[#ff00fc]/10 border border-[#ff00fc]/30 text-[#ff00fc] hover:bg-[#ff00fc] hover:text-black py-2 text-xs text-left px-3">
                            [03] IRIS_VIRGINICA_PROFILE
                        </button>
                    </div>
                </div>

                <!-- SYSTEM LOG -->
                <div class="cyber-panel p-5 col-span-1 md:col-span-2 flex flex-col">
                    <h2 class="text-sm font-bold tracking-wider mb-2 text-white border-b border-gray-700 pb-2 flex justify-between">
                        <span>>> SYS.LOG_HISTORY</span>
                        <span class="text-[10px] text-gray-500 font-normal">MAX_ROWS: 50</span>
                    </h2>
                    <div class="flex-grow overflow-auto h-40">
                        <table class="w-full text-[10px] font-mono cyber-table">
                            <thead class="sticky top-0 bg-[#050505]">
                                <tr>
                                    <th class="p-2">TIME</th>
                                    <th class="p-2">SL</th><th class="p-2">SW</th><th class="p-2">PL</th><th class="p-2">PW</th>
                                    <th class="p-2">PREDICTION</th>
                                    <th class="p-2 text-right">CONFIDENCE</th>
                                </tr>
                            </thead>
                            <tbody id="log-body" class="text-gray-300">
                                <!-- Logs appear here -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

        </main>

        <script>
            // Audio System
            let audioEnabled = true;
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            function playSound(type) {
                if (!audioEnabled || audioCtx.state === 'suspended') audioCtx.resume();
                if (!audioEnabled) return;
                try {
                    const osc = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();
                    
                    if(type === 'click') { osc.type = "square"; osc.frequency.setValueAtTime(800, audioCtx.currentTime); gain.gain.setValueAtTime(0.01, audioCtx.currentTime); osc.stop(audioCtx.currentTime + 0.05); }
                    else if(type === 'success') { osc.type = "sine"; osc.frequency.setValueAtTime(1200, audioCtx.currentTime); gain.gain.setValueAtTime(0.03, audioCtx.currentTime); osc.stop(audioCtx.currentTime + 0.2); }
                    else if(type === 'scan') { osc.type = "sawtooth"; osc.frequency.setValueAtTime(400, audioCtx.currentTime); gain.gain.setValueAtTime(0.01, audioCtx.currentTime); osc.stop(audioCtx.currentTime + 0.1); }
                    
                    gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.2);
                    osc.connect(gain); gain.connect(audioCtx.destination); osc.start();
                } catch(e){}
            }
            function toggleAudio() {
                audioEnabled = !audioEnabled;
                document.getElementById('sound-btn').textContent = audioEnabled ? 'SND_ON' : 'SND_OFF';
                playSound('click');
            }

            // Radar Chart Init
            let radarChart;
            function initChart() {
                const ctx = document.getElementById('radarChart').getContext('2d');
                Chart.defaults.color = '#718096';
                Chart.defaults.font.family = 'JetBrains Mono';
                radarChart = new Chart(ctx, {
                    type: 'radar',
                    data: {
                        labels: ['SL', 'SW', 'PL', 'PW'],
                        datasets: [
                            {
                                label: 'SCAN_DATA', data: [0,0,0,0],
                                backgroundColor: 'rgba(0, 243, 255, 0.2)', borderColor: '#00f3ff',
                                pointBackgroundColor: '#00f3ff', borderWidth: 2, pointRadius: 2
                            },
                            {
                                label: 'IDEAL_BASELINE', data: [5.8, 3.0, 3.7, 1.2], // AVG of all
                                backgroundColor: 'transparent', borderColor: 'rgba(255, 255, 255, 0.1)',
                                borderWidth: 1, borderDash: [2, 2], pointRadius: 0
                            }
                        ]
                    },
                    options: {
                        responsive: true, maintainAspectRatio: false,
                        scales: {
                            r: {
                                angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                                pointLabels: { color: '#a0aec0', font: { size: 10 } },
                                ticks: { display: false }, min: 0, max: 8
                            }
                        },
                        plugins: { legend: { display: false } }
                    }
                });
            }

            // Sync Slider text
            function syncVal(sliderId, textId) {
                const val = document.getElementById(sliderId).value;
                document.getElementById(textId).textContent = val;
            }

            // Randomize
            function randomizeInputs() {
                playSound('click');
                const r = (min, max) => (Math.random() * (max - min) + min).toFixed(1);
                applyInputs(r(4.5, 7.5), r(2.2, 4.0), r(1.2, 6.5), r(0.2, 2.4));
            }

            // Target Species Codex (Phân tích loại hoa được chọn)
            function loadSpeciesData(class_id) {
                playSound('click');
                // Average ideal specs for each class to simulate loading their profile
                if(class_id === 0) applyInputs(5.0, 3.4, 1.5, 0.2); // Setosa
                else if(class_id === 1) applyInputs(5.9, 2.8, 4.3, 1.3); // Versicolor
                else applyInputs(6.6, 3.0, 5.5, 2.0); // Virginica
            }

            function applyInputs(sl, sw, pl, pw) {
                document.getElementById('sl').value = sl; document.getElementById('sw').value = sw;
                document.getElementById('pl').value = pl; document.getElementById('pw').value = pw;
                syncVal('sl', 'txt-sl'); syncVal('sw', 'txt-sw'); syncVal('pl', 'txt-pl'); syncVal('pw', 'txt-pw');
                executeInference();
            }

            // Add to Sys Log
            function addToLog(sl, sw, pl, pw, pred, conf, color) {
                const tbody = document.getElementById('log-body');
                const time = new Date().toLocaleTimeString('en-US', {hour12:false});
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="p-2 text-gray-500">${time}</td>
                    <td class="p-2">${sl}</td><td class="p-2">${sw}</td><td class="p-2">${pl}</td><td class="p-2">${pw}</td>
                    <td class="p-2" style="color: ${color}">${pred}</td>
                    <td class="p-2 text-right">${conf}%</td>
                `;
                tbody.prepend(tr);
                if(tbody.children.length > 50) tbody.removeChild(tbody.lastChild);
            }

            // Main Inference Engine
            async function executeInference(isBatch = false) {
                if(!isBatch) playSound('scan');
                const btnSpin = document.getElementById('btn-spin');
                if(!isBatch) btnSpin.classList.remove('hidden');

                const p_sl = parseFloat(document.getElementById('sl').value);
                const p_sw = parseFloat(document.getElementById('sw').value);
                const p_pl = parseFloat(document.getElementById('pl').value);
                const p_pw = parseFloat(document.getElementById('pw').value);

                try {
                    const res = await fetch('/predict', {
                        method: 'POST', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ sepal_length: p_sl, sepal_width: p_sw, petal_length: p_pl, petal_width: p_pw })
                    });
                    if (!res.ok) throw new Error("API_ERR");
                    
                    const data = await res.json();
                    const spec = data.species;
                    
                    if(!isBatch) updateUI(data, spec);
                    
                    addToLog(p_sl, p_sw, p_pl, p_pw, data.prediction, data.confidences[data.prediction.toLowerCase()], spec.accent);
                    
                    if(!isBatch) setTimeout(() => playSound('success'), 300);

                } catch(e) {
                    console.error(e);
                    document.getElementById('specimen-title').textContent = "SYS_ERR";
                } finally {
                    if(!isBatch) btnSpin.classList.add('hidden');
                }
            }

            // Update UI with Glitch/Cyber effects
            function updateUI(data, spec) {
                const title = document.getElementById('specimen-title');
                title.textContent = spec.name;
                title.style.color = spec.accent;
                title.style.textShadow = `0 0 10px ${spec.accent}`;
                
                document.getElementById('specimen-author').textContent = `TAXA: ${spec.author}`;
                document.getElementById('specimen-desc').textContent = spec.desc;
                document.getElementById('specimen-eco').textContent = spec.ecology;
                
                const mainConf = document.getElementById('main-conf');
                mainConf.textContent = `${data.confidences[data.prediction.toLowerCase()]}%`;
                mainConf.style.color = spec.accent;
                
                // Update Image
                const imgEl = document.getElementById('specimen-img');
                const currentImgPath = new URL(imgEl.src, window.location.origin).pathname;
                if (currentImgPath !== spec.image) {
                    imgEl.style.opacity = 0.2; // Sci-fi fade
                    setTimeout(() => { imgEl.src = spec.image; imgEl.style.opacity = 1; }, 150);
                }
                
                // Theme borders & Glows
                document.getElementById('card-border').style.borderColor = spec.accent;
                document.getElementById('card-border').style.boxShadow = `0 0 20px ${spec.accent}40`;
                document.getElementById('holo-glow').style.background = `linear-gradient(45deg, transparent, ${spec.accent}, transparent)`;
                document.getElementById('holo-tint').style.backgroundColor = spec.accent;
                document.getElementById('watermark').textContent = spec.tag.substring(0,3);
                
                const badge = document.getElementById('specimen-badge');
                badge.textContent = `STATUS: IDENTIFIED [${spec.tag}]`;
                badge.style.borderColor = spec.accent; badge.style.color = spec.accent;

                // Bars
                document.getElementById('bar-0').style.width = `${data.confidences.setosa}%`; document.getElementById('bar-val-0').textContent = `${data.confidences.setosa}%`;
                document.getElementById('bar-1').style.width = `${data.confidences.versicolor}%`; document.getElementById('bar-val-1').textContent = `${data.confidences.versicolor}%`;
                document.getElementById('bar-2').style.width = `${data.confidences.virginica}%`; document.getElementById('bar-val-2').textContent = `${data.confidences.virginica}%`;

                // Radar
                if (radarChart) {
                    radarChart.data.datasets[0].data = data.features;
                    radarChart.data.datasets[0].borderColor = spec.accent;
                    radarChart.data.datasets[0].pointBackgroundColor = spec.accent;
                    
                    const hexToRgba = (hex, alpha) => {
                        const [r, g, b] = hex.match(/\w\w/g).map(x => parseInt(x, 16));
                        return `rgba(${r},${g},${b},${alpha})`;
                    };
                    radarChart.data.datasets[0].backgroundColor = hexToRgba(spec.accent, 0.2);
                    radarChart.update();
                }
            }

            // CSV Batch Protocol
            function handleCSVUpload(event) {
                playSound('click');
                const file = event.target.files[0];
                if (!file) return;
                
                const reader = new FileReader();
                reader.onload = async function(e) {
                    const text = e.target.result;
                    const lines = text.split('\\n');
                    
                    let processed = 0;
                    // Read line by line, skipping empty
                    for(let i=0; i<lines.length && processed < 100; i++) {
                        const cols = lines[i].split(',').map(s => s.trim());
                        if (cols.length >= 4) {
                            const sl = parseFloat(cols[0]); const sw = parseFloat(cols[1]);
                            const pl = parseFloat(cols[2]); const pw = parseFloat(cols[3]);
                            
                            if(!isNaN(sl) && !isNaN(sw) && !isNaN(pl) && !isNaN(pw)) {
                                // Put into sliders and run inference without sound/UI refresh freeze
                                document.getElementById('sl').value = sl; document.getElementById('sw').value = sw;
                                document.getElementById('pl').value = pl; document.getElementById('pw').value = pw;
                                await executeInference(true); // true = isBatch
                                processed++;
                                // Small delay for visual scanning effect
                                await new Promise(r => setTimeout(r, 50)); 
                            }
                        }
                    }
                    playSound('success');
                    // Sync the last one to UI
                    syncVal('sl', 'txt-sl'); syncVal('sw', 'txt-sw'); syncVal('pl', 'txt-pl'); syncVal('pw', 'txt-pw');
                    executeInference(); 
                    event.target.value = ''; // reset file input
                };
                reader.readAsText(file);
            }

            // PDF Generator
            function downloadPDF() {
                playSound('click');
                const element = document.getElementById('report-area');
                
                // Add a temporary solid background for PDF rendering
                const originalBg = element.style.background;
                element.style.background = '#000';
                
                const opt = {
                    margin:       0.2,
                    filename:     'Botany_QC_Report.pdf',
                    image:        { type: 'jpeg', quality: 0.98 },
                    html2canvas:  { scale: 2, useCORS: true, backgroundColor: '#050505' },
                    jsPDF:        { unit: 'in', format: 'letter', orientation: 'landscape' }
                };

                html2pdf().set(opt).from(element).save().then(() => {
                    element.style.background = originalBg;
                });
            }

            window.addEventListener('DOMContentLoaded', () => {
                initChart();
                // Khởi động load mẫu Setosa mặc định
                loadSpeciesData(0);
            });
        </script>
    </body>
    </html>
    """
