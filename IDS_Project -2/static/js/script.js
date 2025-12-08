// static/js/script.js (最终增强版)

// 设置超时时间 (毫秒)，这里设为 8秒
const TIMEOUT_MS = 8000;

// 1. 页面加载时：获取模型性能 & 历史记录
document.addEventListener('DOMContentLoaded', () => {
    loadPerformance();
    // 可以在这里开启定时刷新，但为了防止报错刷屏，先注释掉
    // setInterval(loadPerformance, 10000); 
});

// --- 核心工具：带超时的 Fetch 请求 ---
async function fetchWithTimeout(resource, options = {}) {
    const { timeout = TIMEOUT_MS } = options;
    
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    
    try {
        const response = await fetch(resource, {
            ...options,
            signal: controller.signal  // 绑定信号
        });
        clearTimeout(id);
        return response;
    } catch (error) {
        clearTimeout(id);
        throw error; // 把错误抛出去给具体函数处理
    }
}

// 2. 获取模型性能指标
async function loadPerformance() {
    try {
        // 使用带超时的请求
        const res = await fetchWithTimeout('/api/performance', { timeout: 5000 });
        
        if (!res.ok) throw new Error("Server offline");
        const data = await res.json();
        
        // 更新 Accuracy
        updateText('acc-display', data.accuracy, '%');
        // 更新 Precision
        updateText('prec-display', data.precision, '%');
        // 更新 Recall
        updateText('recall-display', data.recall, '%');
        // 更新 F1 Score
        updateText('f1-display', data.f1_score, '%');

        // 更新图表
        if(window.perfChart) {
            window.perfChart.data.datasets[0].data = [
                data.accuracy, data.precision, data.recall, data.f1_score, data.auc || 0.95
            ];
            window.perfChart.update();
        }

    } catch (e) {
        console.warn("加载指标失败或超时:", e);
        // 如果超时，显示 "--" 而不是一直 Loading
        ['acc-display', 'prec-display', 'recall-display', 'f1-display'].forEach(id => {
            const el = document.getElementById(id);
            if(el && el.innerText.includes('Loading')) el.innerText = "Offline";
        });
    }
}

// 辅助函数：安全更新文本
function updateText(id, value, suffix='') {
    const el = document.getElementById(id);
    if(el && value !== undefined) {
        el.innerText = (value * 100).toFixed(1) + suffix;
    }
}

// 3. 核心功能：模拟流量并预测
async function simulateTraffic(type) {
    const statusEl = document.getElementById('scanStatus');
    const resultCard = document.getElementById('resultCard'); 
    
    // 显示 Loading 动画
    if(statusEl) statusEl.style.display = 'block';

    try {
        // A. 第一步：获取模拟数据
        let url = '/api/random';
        if (type === 'normal') url = '/api/sample';
        if (type === 'attack') url = '/api/simulate-attack';

        // 设置 5秒超时
        const dataRes = await fetchWithTimeout(url, { timeout: 5000 });
        if(!dataRes.ok) throw new Error("Failed to generate traffic data");
        const dataJson = await dataRes.json();

        // B. 第二步：发送预测请求
        const predictRes = await fetchWithTimeout('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: dataJson.features }),
            timeout: 5000
        });
        
        if(!predictRes.ok) throw new Error("Prediction API failed");
        const result = await predictRes.json();

        // C. 第三步：更新 UI
        updateDashboardUI(result);

    } catch (err) {
        console.error(err);
        // --- 这里的反馈就是你要的 ---
        if (err.name === 'AbortError') {
            alert("⚠️ Request Timed Out!\nThe server took too long to respond. Please try again.");
        } else {
            alert("❌ Error: " + err.message);
        }
    } finally {
        // 关键：无论成功失败，必须关掉动画
        if(statusEl) statusEl.style.display = 'none';
    }
}

// 4. 更新界面显示的 UI 逻辑
function updateDashboardUI(data) {
    const typeEl = document.getElementById('resultType');
    const badgeEl = document.getElementById('resultSeverity');
    const consoleEl = document.getElementById('logConsole');

    if(typeEl) {
        typeEl.textContent = data.predicted_label;
        // 颜色映射
        typeEl.className = "fw-bold mb-0 h3 "; // 保持基础样式
        if(data.threat_level === 'High') typeEl.classList.add('text-danger');
        else if(data.threat_level === 'Medium') typeEl.classList.add('text-warning');
        else typeEl.classList.add('text-success');

        if(badgeEl) {
            badgeEl.textContent = data.threat_level;
            badgeEl.className = "badge rounded-pill px-3 py-2 shadow-sm ";
            if(data.threat_level === 'High') badgeEl.classList.add('bg-danger');
            else if(data.threat_level === 'Medium') badgeEl.classList.add('bg-warning', 'text-dark');
            else badgeEl.classList.add('bg-success');
        }
    }

    if(consoleEl) {
        const timestamp = new Date().toLocaleTimeString();
        const colorClass = data.threat_level === 'None' ? 'text-success' : 'text-danger';
        // 插入日志
        const newLog = document.createElement('div');
        newLog.className = colorClass;
        newLog.innerHTML = `[${timestamp}] DETECTED: <b>${data.predicted_label}</b> (Conf: ${(data.confidence*100).toFixed(1)}%)`;
        consoleEl.prepend(newLog); // 插入到最前面
    }
}

// 5. 重训练 (上传文件)
async function uploadAndRetrain() {
    const fileInput = document.querySelector('input[type="file"]');
    if (!fileInput || !fileInput.files[0]) {
        alert("Please select a CSV file first!");
        return;
    }

    const formData = new FormData();
    formData.append('files', fileInput.files[0]);

    const btn = document.querySelector('button[onclick="uploadAndRetrain()"]');
    const originalText = btn.innerHTML;
    
    // 锁定按钮
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Training...';

    try {
        // 训练通常比较慢，我们给它 30秒 的时间
        const res = await fetchWithTimeout('/api/upload-and-retrain', {
            method: 'POST',
            body: formData,
            timeout: 30000 
        });
        
        if(!res.ok) throw new Error("Retraining failed");
        const result = await res.json();
        
        alert("✅ " + result.message);
        loadPerformance(); // 刷新指标

    } catch (e) {
        if (e.name === 'AbortError') {
            alert("⚠️ Training Timeout!\nThe dataset might be too large. Check the console for background progress.");
        } else {
            alert("❌ Training Failed: " + e.message);
        }
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
        // 清空文件框
        fileInput.value = ''; 
    }
}