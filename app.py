from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib

# 1. Tải mô hình đã huấn luyện
model = joblib.load("svm_model.pkl")

# 2. Khởi tạo FastAPI
app = FastAPI(
    title="Iris Classification Web App",
    description="Ứng dụng phân loại hoa Iris với mô hình SVM",
    version="2.0.0",
)

# 3. Schema kiểm tra dữ liệu đầu vào
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Thông tin chi tiết kèm màu sắc nhận diện cho từng loài hoa
species_details = {
    0: {
        "name": "Iris Setosa",
        "tag": "Setosa",
        "color": "emerald",
        "badge_class": "bg-emerald-100 text-emerald-700 border-emerald-300",
        "desc": "Cánh hoa và đài hoa nhỏ gọn. Đây là loài hoa có đặc trưng kích thước petal nhỏ nhất và phân biệt rõ nhất.",
    },
    1: {
        "name": "Iris Versicolor",
        "tag": "Versicolor",
        "color": "sky",
        "badge_class": "bg-sky-100 text-sky-700 border-sky-300",
        "desc": "Kích thước các cánh hoa ở mức trung bình, có tỷ lệ hài hòa giữa đài hoa (sepal) và cánh hoa (petal).",
    },
    2: {
        "name": "Iris Virginica",
        "tag": "Virginica",
        "color": "purple",
        "badge_class": "bg-purple-100 text-purple-700 border-purple-300",
        "desc": "Kích thước cánh hoa và đài hoa lớn, dài và rực rỡ nhất trong cả ba loài của bộ dữ liệu Iris.",
    },
}

# 4. Trang chủ: Giao diện Web tối ưu và hiện đại
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Phân loại hoa Iris - AI SVM Classifier</title>
        <!-- Tailwind CSS CDN -->
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Plus Jakarta Sans', sans-serif; }
        </style>
    </head>
    <body class="bg-gradient-to-br from-slate-50 via-indigo-50/40 to-slate-100 min-h-screen text-slate-800 flex flex-col justify-between">
        
        <!-- Header -->
        <header class="border-b border-slate-200/80 bg-white/70 backdrop-blur-md sticky top-0 z-10">
            <div class="max-w-5xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 rounded-xl bg-indigo-600 text-white flex items-center justify-center font-bold text-lg shadow-md shadow-indigo-200">
                        🌸
                    </div>
                    <div>
                        <h1 class="font-bold text-lg text-slate-900 leading-tight">Iris SVM AI</h1>
                        <p class="text-xs text-slate-500">FastAPI & Machine Learning Deployment</p>
                    </div>
                </div>
                <div class="flex items-center gap-3">
                    <span id="health-badge" class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
                        <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span> Server Active
                    </span>
                    <a href="/docs" target="_blank" class="text-xs font-medium text-indigo-600 hover:text-indigo-800 transition">API Docs ↗</a>
                </div>
            </div>
        </header>

        <!-- Main Content -->
        <main class="max-w-5xl mx-auto px-4 sm:px-6 py-8 sm:py-12 w-full grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            
            <!-- Left Column: Form & Presets -->
            <div class="lg:col-span-7 bg-white rounded-2xl p-6 sm:p-8 shadow-xl shadow-slate-200/50 border border-slate-100">
                <div class="mb-6">
                    <h2 class="text-xl font-bold text-slate-900">Nhập kích thước đặc trưng</h2>
                    <p class="text-sm text-slate-500 mt-1">Điền kích thước cánh hoa (cm) hoặc chọn nhanh một mẫu dữ liệu có sẵn.</p>
                    
                    <!-- Quick sample buttons -->
                    <div class="mt-4 flex flex-wrap gap-2">
                        <span class="text-xs font-semibold text-slate-400 self-center mr-1">Thử nhanh:</span>
                        <button type="button" onclick="loadSample(5.1, 3.5, 1.4, 0.2)" class="px-2.5 py-1 text-xs rounded-lg bg-slate-100 hover:bg-emerald-100 text-slate-700 hover:text-emerald-800 transition border border-slate-200 font-medium">Mẫu Setosa</button>
                        <button type="button" onclick="loadSample(5.9, 3.0, 4.2, 1.5)" class="px-2.5 py-1 text-xs rounded-lg bg-slate-100 hover:bg-sky-100 text-slate-700 hover:text-sky-800 transition border border-slate-200 font-medium">Mẫu Versicolor</button>
                        <button type="button" onclick="loadSample(6.5, 3.0, 5.5, 1.8)" class="px-2.5 py-1 text-xs rounded-lg bg-slate-100 hover:bg-purple-100 text-slate-700 hover:text-purple-800 transition border border-slate-200 font-medium">Mẫu Virginica</button>
                    </div>
                </div>

                <form id="iris-form" class="space-y-4" onsubmit="handlePredict(event)">
                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <!-- Sepal Length -->
                        <div>
                            <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-1">Chiều dài đài hoa (Sepal Length)</label>
                            <div class="relative">
                                <input type="number" step="0.1" min="0" required id="sepal_length" value="5.1"
                                    class="w-full px-4 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition text-sm font-medium">
                                <span class="absolute right-3.5 top-2.5 text-xs text-slate-400 font-medium">cm</span>
                            </div>
                        </div>

                        <!-- Sepal Width -->
                        <div>
                            <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-1">Chiều rộng đài hoa (Sepal Width)</label>
                            <div class="relative">
                                <input type="number" step="0.1" min="0" required id="sepal_width" value="3.5"
                                    class="w-full px-4 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition text-sm font-medium">
                                <span class="absolute right-3.5 top-2.5 text-xs text-slate-400 font-medium">cm</span>
                            </div>
                        </div>

                        <!-- Petal Length -->
                        <div>
                            <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-1">Chiều dài cánh hoa (Petal Length)</label>
                            <div class="relative">
                                <input type="number" step="0.1" min="0" required id="petal_length" value="1.4"
                                    class="w-full px-4 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition text-sm font-medium">
                                <span class="absolute right-3.5 top-2.5 text-xs text-slate-400 font-medium">cm</span>
                            </div>
                        </div>

                        <!-- Petal Width -->
                        <div>
                            <label class="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-1">Chiều rộng cánh hoa (Petal Width)</label>
                            <div class="relative">
                                <input type="number" step="0.1" min="0" required id="petal_width" value="0.2"
                                    class="w-full px-4 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition text-sm font-medium">
                                <span class="absolute right-3.5 top-2.5 text-xs text-slate-400 font-medium">cm</span>
                            </div>
                        </div>
                    </div>

                    <button type="submit" id="btn-submit"
                        class="w-full mt-6 py-3 px-4 bg-indigo-600 hover:bg-indigo-700 active:scale-[0.99] text-white font-semibold rounded-xl shadow-lg shadow-indigo-200 transition duration-150 flex items-center justify-center gap-2">
                        <span id="btn-text">Dự đoán kết quả</span>
                        <div id="btn-spinner" class="hidden w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin"></div>
                    </button>
                </form>
            </div>

            <!-- Right Column: Result Card -->
            <div class="lg:col-span-5 flex flex-col gap-6">
                <div class="bg-white rounded-2xl p-6 sm:p-8 shadow-xl shadow-slate-200/50 border border-slate-100 relative overflow-hidden">
                    <h3 class="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4">Kết quả nhận diện</h3>
                    
                    <!-- Empty State -->
                    <div id="result-placeholder" class="text-center py-10">
                        <div class="text-4xl mb-3">🔍</div>
                        <p class="text-sm text-slate-500">Chưa có dự đoán nào.<br>Nhập thông số và bấm <b>"Dự đoán kết quả"</b>.</p>
                    </div>

                    <!-- Populated State -->
                    <div id="result-box" class="hidden flex-col items-center text-center animate-fade-in">
                        <div id="flower-icon" class="text-5xl mb-3">🌺</div>
                        <span id="species-badge" class="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border mb-2">Setosa</span>
                        <h4 id="species-name" class="text-2xl font-bold text-slate-900 mb-1">Iris Setosa</h4>
                        <p id="species-class" class="text-xs text-slate-400 mb-4">Mã phân lớp: 0</p>
                        
                        <div class="w-full bg-slate-50 rounded-xl p-4 border border-slate-100 text-left">
                            <p class="text-xs text-slate-600 leading-relaxed" id="species-desc">
                                Đặc trưng cánh hoa và đài hoa nhỏ gọn...
                            </p>
                        </div>
                    </div>
                </div>

                <!-- Model Info Card -->
                <div class="bg-gradient-to-r from-slate-900 to-indigo-950 text-white rounded-2xl p-5 shadow-lg">
                    <div class="flex items-center justify-between mb-2">
                        <span class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Thông tin mô hình</span>
                        <span class="text-[10px] bg-indigo-500/30 text-indigo-200 px-2 py-0.5 rounded-full border border-indigo-400/30">Scikit-Learn</span>
                    </div>
                    <p class="text-sm font-medium text-slate-200">Support Vector Machine (Kernel = Linear)</p>
                    <p class="text-xs text-slate-400 mt-1">Được đóng gói tự động bằng FastAPI và sẵn sàng phục vụ Production.</p>
                </div>
            </div>

        </main>

        <!-- Footer -->
        <footer class="border-t border-slate-200/80 bg-white/50 text-center py-4 text-xs text-slate-400">
            Triển khai mô hình SVM Iris &bull; FastAPI, Uvicorn & Render
        </footer>

        <!-- Client JavaScript -->
        <script>
            function loadSample(sl, sw, pl, pw) {
                document.getElementById('sepal_length').value = sl;
                document.getElementById('sepal_width').value = sw;
                document.getElementById('petal_length').value = pl;
                document.getElementById('petal_width').value = pw;
                handlePredict(new Event('submit'));
            }

            async function handlePredict(e) {
                e.preventDefault();
                const btnText = document.getElementById('btn-text');
                const btnSpinner = document.getElementById('btn-spinner');
                
                // Hiển thị trạng thái loading
                btnText.textContent = "Đang phân tích...";
                btnSpinner.classList.remove('hidden');

                const payload = {
                    sepal_length: parseFloat(document.getElementById('sepal_length').value),
                    sepal_width: parseFloat(document.getElementById('sepal_width').value),
                    petal_length: parseFloat(document.getElementById('petal_length').value),
                    petal_width: parseFloat(document.getElementById('petal_width').value)
                };

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });

                    if (!response.ok) throw new Error("Dự đoán thất bại");

                    const data = await response.json();
                    
                    // Hiển thị kết quả
                    document.getElementById('result-placeholder').classList.add('hidden');
                    const resBox = document.getElementById('result-box');
                    resBox.classList.remove('hidden');
                    resBox.classList.add('flex');

                    document.getElementById('species-name').textContent = data.details.name;
                    document.getElementById('species-class').textContent = "Mã phân lớp: Class " + data.class_id;
                    document.getElementById('species-desc').textContent = data.details.desc;
                    
                    const badge = document.getElementById('species-badge');
                    badge.textContent = data.details.tag;
                    badge.className = `px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border mb-2 ${data.details.badge_class}`;

                } catch (err) {
                    alert("Có lỗi khi gọi API: " + err.message);
                } finally {
                    btnText.textContent = "Dự đoán kết quả";
                    btnSpinner.classList.add('hidden');
                }
            }
        </script>
    </body>
    </html>
    """

# 5. Endpoint kiểm tra sức khỏe hệ thống
@app.get("/health")
def health():
    return {"status": "healthy"}

# 6. Endpoint dự đoán (phục vụ cả Web UI lẫn API bên ngoài)
@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]

    prediction = int(model.predict(features)[0])
    details = species_details.get(prediction, {
        "name": "Unknown",
        "tag": "Unknown",
        "badge_class": "bg-slate-100 text-slate-700",
        "desc": "Không xác định được loài hoa.",
    })

    return {
        "class_id": prediction,
        "prediction": details["tag"].lower(),
        "details": details
    }
