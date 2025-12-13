import React, { useState, useEffect } from 'react';
import { PredictionResult, AlertLog, PerformanceMetrics } from './types';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// API Base URL
const API_URL = 'http://127.0.0.1:5000/api';

const App: React.FC = () => {
    // State definition
    const [features, setFeatures] = useState<number[]>([]);
    const [featureNames, setFeatureNames] = useState<string[]>([]);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [alerts, setAlerts] = useState<AlertLog[]>([]);
    const [history, setHistory] = useState<AlertLog[]>([]);
    const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
    const [retraining, setRetraining] = useState<boolean>(false);
    const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
    const [uploading, setUploading] = useState<boolean>(false);
    const [editingIndex, setEditingIndex] = useState<number | null>(null);
    const [editValue, setEditValue] = useState<string>('');
    const [activeTab, setActiveTab] = useState<'alerts' | 'history'>('alerts');
    const [backendStatus, setBackendStatus] = useState<'unknown' | 'connected' | 'error'>('unknown');
    const [attackFrequency, setAttackFrequency] = useState<number>(0);
    const [timeWindow, setTimeWindow] = useState<number>(10);

    // 根据攻击类型获取基础危险等级
    const getBaseSeverity = (attackType: string): number => {
        const attackTypeUpper = attackType.toUpperCase();

        if (attackTypeUpper.includes('DDoS') || attackTypeUpper.includes('DOS')) {
            return 9; // 最高危险等级
        } else if (attackTypeUpper.includes('BOT') || attackTypeUpper.includes('GOLDENEYE') || attackTypeUpper.includes('SLOWLORIS')) {
            return 8;
        } else if (attackTypeUpper.includes('PORTSCAN') || attackTypeUpper.includes('SCAN')) {
            return 7;
        } else if (attackTypeUpper.includes('WEB ATTACK') || attackTypeUpper.includes('INFILTRATION')) {
            return 8;
        } else if (attackTypeUpper.includes('FTP-PATATOR') || attackTypeUpper.includes('SSH-PATATOR')) {
            return 6;
        } else if (attackTypeUpper.includes('HEARTBLEED')) {
            return 5;
        } else if (attackTypeUpper.includes('BENIGN')) {
            return 0;
        } else {
            return 5; // 默认中等危险
        }
    };

    // 检查后端连接
    const checkBackendConnection = async () => {
        try {
            const res = await fetch('http://127.0.0.1:5000/health');
            if (res.ok) {
                setBackendStatus('connected');
                return true;
            } else {
                setBackendStatus('error');
                return false;
            }
        } catch (error) {
            console.error("无法连接到后端:", error);
            setBackendStatus('error');
            return false;
        }
    };

    // Initial Data Load
    useEffect(() => {
        const initApp = async () => {
            const connected = await checkBackendConnection();
            if (connected) {
                fetchSampleData();
                fetchAlerts();
                fetchHistory();
                fetchMetrics();
                // Set up polling for alerts every 5 seconds
                const interval = setInterval(fetchAlerts, 5000);
                return () => clearInterval(interval);
            } else {
                console.error("后端连接失败，将使用模拟数据");
            }
        };
        initApp();
    }, []);

    const fetchSampleData = async () => {
        try {
            const res = await fetch(`${API_URL}/sample`);
            if (res.ok) {
                const data = await res.json();
                setFeatures(data.features);
                setFeatureNames(data.feature_names);
                setAttackFrequency(0); // 样本数据攻击频率为0
            } else {
                console.error("获取样本数据失败:", res.status);
            }
        } catch (error) {
            console.error("Error fetching sample:", error);
        }
    };

    const fetchRandomData = async () => {
        try {
            const res = await fetch(`${API_URL}/random`);
            if (res.ok) {
                const data = await res.json();
                // 检查是否有status字段
                if (data.status === 'success') {
                    setFeatures(data.features);
                    setFeatureNames(data.feature_names);
                    setAttackFrequency(data.attack_frequency || 0);
                } else {
                    // 如果没有status字段，直接使用数据
                    setFeatures(data.features || []);
                    setFeatureNames(data.feature_names || []);
                    setAttackFrequency(0);
                }
                setResult(null); // 重置预测结果
            } else {
                console.error("获取随机数据失败:", res.status);
                // Fallback to generating client-side random data
                const newFeatures = features.map(() => Math.random() * 100000 - 50000);
                setFeatures(newFeatures);
                setAttackFrequency(0);
            }
        } catch (error) {
            console.error("Error fetching random data:", error);
            // Fallback to generating client-side random data
            const newFeatures = features.map(() => Math.random() * 100000 - 50000);
            setFeatures(newFeatures);
            setAttackFrequency(0);
        }
    };

    const fetchAttackData = async () => {
        try {
            console.log("正在获取攻击数据...");
            // 使用正确的端点 /api/stream
            const res = await fetch(`${API_URL}/stream`);
            if (res.ok) {
                const data = await res.json();
                console.log("攻击数据响应:", data);

                if (data.status === 'success') {
                    setFeatures(data.features);
                    setFeatureNames(data.feature_names);

                    // 根据攻击类型智能生成攻击频率
                    const attackType = data.predicted_label || data.true_label || '';
                    const baseSeverity = getBaseSeverity(attackType);

                    // 根据攻击类型的基础危险等级生成相应的攻击频率
                    let generatedAttackFrequency = 0;
                    if (baseSeverity >= 8) { // 高危险攻击
                        // 生成更高的攻击频率
                        generatedAttackFrequency = Math.floor(Math.random() * 121) + 80;  // 80-200
                    } else if (baseSeverity >= 6) { // 中等危险攻击
                        generatedAttackFrequency = Math.floor(Math.random() * 71) + 50; // 50-120
                    } else { // 低危险攻击
                        generatedAttackFrequency = Math.floor(Math.random() * 41) + 30; // 30-70
                    }

                    // 如果后端返回的攻击频率更高，使用后端的数据
                    const finalFrequency = Math.max(generatedAttackFrequency, data.attack_frequency || 0);

                    setAttackFrequency(Number(finalFrequency) || 0);
                    setTimeWindow(data.time_window_seconds || 10);

                    console.log(`设置攻击频率: ${finalFrequency} 次/${data.time_window_seconds || 10}秒, 攻击类型: ${attackType}, 基础危险等级: ${baseSeverity}`);

                    // 如果有预测结果，直接显示
                    if (data.predicted_label) {
                        const attackResult: PredictionResult = {
                            status: 'success',
                            predicted_label: data.predicted_label,
                            confidence: data.confidence || 0.9,
                            threat_level: data.threat_level || 'High',
                            encoded_value: 0
                        };
                        setResult(attackResult);

                        // 立即进行预测，以生成警报
                        setTimeout(() => {
                            handlePredict();
                        }, 500);
                    }
                } else {
                    console.error("后端返回错误:", data.message);
                }
            } else {
                console.error("获取攻击数据失败:", res.status);
                // Fallback to generating client-side attack-like data
                const newFeatures = features.map((_, i) => i % 3 === 0 ? Math.random() * 100000 : Math.random() * 1000);
                setFeatures(newFeatures);

                // 生成更高的随机攻击频率
                const highAttackFrequency = Math.floor(Math.random() * 121) + 80; // 80-200
                setAttackFrequency(highAttackFrequency);
            }
        } catch (error) {
            console.error("Error fetching attack simulation data:", error);
            // Fallback to generating client-side attack-like data
            const newFeatures = features.map((_, i) => i % 3 === 0 ? Math.random() * 100000 : Math.random() * 1000);
            setFeatures(newFeatures);

            // 生成更高的随机攻击频率
            const highAttackFrequency = Math.floor(Math.random() * 121) + 80; // 80-200
            setAttackFrequency(highAttackFrequency);
        }
    };

    const fetchAlerts = async () => {
        try {
            const res = await fetch(`${API_URL}/alerts`);
            if (res.ok) {
                const data = await res.json();
                console.log("警报数据:", data);
                setAlerts(Array.isArray(data) ? data : []);
            } else {
                console.error("获取警报失败:", res.status);
            }
        } catch (error) {
            console.error("Error fetching alerts:", error);
        }
    };

    const fetchHistory = async () => {
        try {
            const res = await fetch(`${API_URL}/history`);
            if (res.ok) {
                const data = await res.json();
                setHistory(Array.isArray(data) ? data : []);
            } else {
                console.error("获取历史记录失败:", res.status);
            }
        } catch (error) {
            console.error("Error fetching history:", error);
        }
    };

    const fetchMetrics = async () => {
        try {
            const res = await fetch(`${API_URL}/performance`);
            if (res.ok) {
                const data = await res.json();
                setMetrics(data);
            } else {
                console.error("获取性能指标失败:", res.status);
            }
        } catch (error) {
            console.error("Error fetching metrics:", error);
        }
    };

    const handlePredict = async () => {
        setLoading(true);
        try {
            // 确保攻击频率足够高以产生高威胁等级
            const effectiveAttackFrequency = Math.max(attackFrequency, 30);

            const predictionData = {
                features: features,
                attack_frequency: effectiveAttackFrequency,  // 添加攻击频率
                time_window: timeWindow || 10               // 添加时间窗口
            };

            console.log("发送预测请求:", predictionData);

            const res = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(predictionData)
            });

            if (res.ok) {
                const data = await res.json();
                console.log("预测结果:", data);

                if (data.status === 'success') {
                    setResult(data);
                } else {
                    setResult({
                        status: 'error',
                        predicted_label: 'Error',
                        confidence: 0,
                        threat_level: 'None',
                        message: data.message
                    });
                }

                // Refresh alerts and history immediately
                fetchAlerts();
                fetchHistory();
            } else {
                console.error("预测请求失败:", res.status);
                setResult({
                    status: 'error',
                    predicted_label: 'Error',
                    confidence: 0,
                    threat_level: 'None',
                    message: `HTTP错误: ${res.status}`
                });
            }
        } catch (error) {
            console.error("Prediction error:", error);
            setResult({
                status: 'error',
                predicted_label: 'Error',
                confidence: 0,
                threat_level: 'None',
                message: error instanceof Error ? error.message : '未知错误'
            });
        } finally {
            setLoading(false);
        }
    };

    const handleRetrain = async () => {
        if (!confirm("This will trigger the training script on the server. Continue?")) return;
        setRetraining(true);
        try {
            const res = await fetch(`${API_URL}/retrain`, { method: 'POST' });
            if (res.ok) {
                const data = await res.json();
                alert(data.message + (data.status === 'success' ? "\nPlease restart backend to load new model." : ""));
            } else {
                alert("Retraining failed: HTTP error " + res.status);
            }
        } catch (error) {
            alert("Retraining failed check console.");
            console.error(error);
        } finally {
            setRetraining(false);
        }
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setSelectedFiles(e.target.files);
    };

    const handleUploadAndRetrain = async () => {
        if (!selectedFiles || selectedFiles.length === 0) {
            alert("Please select at least one CSV file to upload.");
            return;
        }

        const formData = new FormData();
        for (let i = 0; i < selectedFiles.length; i++) {
            formData.append('files', selectedFiles[i]);
        }

        setUploading(true);
        try {
            const res = await fetch(`${API_URL}/upload-and-retrain`, {
                method: 'POST',
                body: formData
            });

            if (res.ok) {
                const data = await res.json();

                if (data.status === 'success') {
                    alert("Model retrained successfully with uploaded data!");
                    // Refresh metrics and sample data
                    fetchMetrics();
                    fetchSampleData();
                } else {
                    alert(`Retraining failed: ${data.message}`);
                }
            } else {
                alert(`Upload failed: HTTP error ${res.status}`);
            }
        } catch (error) {
            alert("Upload and retraining failed. Check console for details.");
            console.error(error);
        } finally {
            setUploading(false);
            setSelectedFiles(null);
        }
    };

    const handleFeatureChange = (index: number, value: string) => {
        const newFeatures = [...features];
        const numValue = parseFloat(value);
        if (!isNaN(numValue)) {
            newFeatures[index] = numValue;
        } else {
            newFeatures[index] = 0;
        }
        setFeatures(newFeatures);
    };

    const startEditing = (index: number, value: number) => {
        setEditingIndex(index);
        setEditValue(value.toString());
    };

    const saveEdit = (index: number) => {
        if (editingIndex === index) {
            handleFeatureChange(index, editValue);
            setEditingIndex(null);
        }
    };

    const cancelEdit = () => {
        setEditingIndex(null);
    };

    // Helper for color coding alerts
    const getLevelColor = (level: string) => {
        switch (level) {
            case 'Critical': return 'bg-red-600 text-white border-red-700';
            case 'High': return 'bg-red-100 text-red-800 border-red-200';
            case 'Medium': return 'bg-orange-100 text-orange-800 border-orange-200';
            case 'Low': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
            case 'None': return 'bg-green-100 text-green-800 border-green-200';
            default: return 'bg-gray-100 text-gray-800 border-gray-200';
        }
    };

    // Prepare chart data
    const chartData = metrics ? [
        { name: 'Accuracy', value: (metrics.accuracy || 0) * 100 },
        { name: 'Precision', value: (metrics.precision || 0) * 100 },
        { name: 'Recall', value: (metrics.recall || 0) * 100 },
        { name: 'F1 Score', value: (metrics.f1_score || 0) * 100 },
    ] : [];

    return (
        <div className="min-h-screen p-6 bg-slate-50 font-sans">
            {/* 后端连接状态指示器 */}
            {backendStatus === 'error' && (
                <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                    <div className="flex items-center">
                        <div className="flex-shrink-0">
                            <svg className="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                            </svg>
                        </div>
                        <div className="ml-3">
                            <p className="text-sm font-medium text-red-800">无法连接到后端服务器</p>
                            <p className="text-sm text-red-700">请确保Flask服务器正在运行: <code>python app.py</code></p>
                        </div>
                    </div>
                </div>
            )}

            <header className="mb-8 flex justify-between items-center bg-white p-6 rounded-xl shadow-sm border border-slate-100">
                <div>
                    <h1 className="text-3xl font-bold text-slate-800 tracking-tight">DDoS Defense Shield</h1>
                    <p className="text-slate-500 mt-1">Real-time Traffic Analysis & Threat Intelligence</p>
                </div>
                <div className="flex gap-4">
                     <button
                        onClick={handleRetrain}
                        disabled={retraining || backendStatus === 'error'}
                        className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                            retraining || backendStatus === 'error'
                            ? 'bg-gray-300 cursor-not-allowed'
                            : 'bg-indigo-600 hover:bg-indigo-700 text-white'
                        }`}
                    >
                        {retraining ? 'Training...' : 'Retrain Model'}
                    </button>
                    <div className={`px-4 py-2 rounded-lg border font-medium ${
                        backendStatus === 'connected'
                        ? 'bg-green-50 text-green-700 border-green-100'
                        : backendStatus === 'error'
                        ? 'bg-red-50 text-red-700 border-red-100'
                        : 'bg-yellow-50 text-yellow-700 border-yellow-100'
                    }`}>
                        {backendStatus === 'connected' ? 'System Active' :
                         backendStatus === 'error' ? 'System Offline' : 'Connecting...'}
                    </div>
                </div>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

                {/* Left Column: Input & Prediction */}
                <div className="lg:col-span-8 space-y-8">

                    {/* File Upload Panel */}
                    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
                        <h2 className="text-xl font-bold text-slate-800 mb-4">Model Retraining</h2>
                        <div className="mb-4">
                            <label className="block text-sm font-medium text-slate-700 mb-2">
                                Upload CSV Files for Retraining
                            </label>
                            <input
                                type="file"
                                accept=".csv"
                                multiple
                                onChange={handleFileChange}
                                className="block w-full text-sm text-slate-500
                                    file:mr-4 file:py-2 file:px-4
                                    file:rounded-lg file:border-0
                                    file:text-sm file:font-semibold
                                    file:bg-indigo-50 file:text-indigo-700
                                    hover:file:bg-indigo-100"
                            />
                            {selectedFiles && (
                                <p className="mt-2 text-sm text-slate-500">
                                    Selected {selectedFiles.length} file(s)
                                </p>
                            )}
                        </div>
                        <button
                            onClick={handleUploadAndRetrain}
                            disabled={uploading || !selectedFiles || selectedFiles.length === 0 || backendStatus === 'error'}
                            className={`w-full py-2 rounded-lg font-medium transition-colors ${
                                uploading || !selectedFiles || selectedFiles.length === 0 || backendStatus === 'error'
                                    ? 'bg-gray-300 cursor-not-allowed'
                                    : 'bg-indigo-600 hover:bg-indigo-700 text-white'
                            }`}
                        >
                            {uploading ? 'Uploading and Retraining...' : 'Upload and Retrain Model'}
                        </button>
                    </div>

                    {/* Prediction Panel */}
                    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
                        <div className="flex justify-between items-center mb-6">
                            <h2 className="text-xl font-bold text-slate-800">Traffic Analyzer</h2>
                            <div className="flex gap-2">
                                <button
                                    onClick={fetchSampleData}
                                    className="text-sm text-indigo-600 hover:text-indigo-800 font-medium"
                                >
                                    ↺ Reset to Sample Data
                                </button>
                                <button
                                    onClick={fetchRandomData}
                                    className="text-sm text-indigo-600 hover:text-indigo-800 font-medium"
                                >
                                    🎲 Random Data
                                </button>
                                <button
                                    onClick={fetchAttackData}
                                    className="text-sm text-red-600 hover:text-red-800 font-medium"
                                >
                                    ⚔️ Simulate Attack
                                </button>
                            </div>
                        </div>

                        {/* 攻击频率显示 */}
                        {attackFrequency > 0 && (
                            <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                                <div className="flex justify-between items-center">
                                    <div>
                                        <span className="font-medium text-blue-700">攻击频率:</span>
                                        <span className="ml-2 font-bold text-blue-900">
                                            {attackFrequency} 次/{timeWindow}秒
                                        </span>
                                        <span className="ml-2 text-xs text-blue-600">
                                            ({Math.round(attackFrequency / timeWindow)} 次/秒)
                                        </span>
                                    </div>
                                    <div className="text-sm text-blue-600">
                                        威胁等级: {attackFrequency > 80 ? '极高' : attackFrequency > 50 ? '高' : attackFrequency > 20 ? '中' : '低'}
                                    </div>
                                </div>
                            </div>
                        )}

                        <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 mb-6 max-h-96 overflow-y-auto">
                            <p className="text-xs text-slate-500 mb-2 font-mono">Raw Feature Vector ({features.length} features)</p>
                            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
                                {features.map((value, index) => (
                                    <div key={index} className="flex items-center text-xs">
                                        <span className="text-slate-500 w-16 truncate mr-1" title={featureNames[index] || `Feature ${index}`}>{
                                            featureNames[index] ?
                                            featureNames[index].length > 10 ?
                                            featureNames[index].substring(0, 10) + '...' :
                                            featureNames[index] :
                                            `Feat ${index}`
                                        }</span>
                                        {editingIndex === index ? (
                                            <div className="flex">
                                                <input
                                                    type="number"
                                                    value={editValue}
                                                    onChange={(e) => setEditValue(e.target.value)}
                                                    className="w-20 px-1 py-0.5 text-xs border rounded"
                                                    autoFocus
                                                    onKeyDown={(e) => {
                                                        if (e.key === 'Enter') saveEdit(index);
                                                        if (e.key === 'Escape') cancelEdit();
                                                    }}
                                                />
                                                <button
                                                    onClick={() => saveEdit(index)}
                                                    className="ml-1 px-1 bg-green-500 text-white rounded"
                                                >
                                                    ✓
                                                </button>
                                                <button
                                                    onClick={cancelEdit}
                                                    className="ml-1 px-1 bg-red-500 text-white rounded"
                                                >
                                                    ✕
                                                </button>
                                            </div>
                                        ) : (
                                            <span
                                                className="font-mono w-24 truncate cursor-pointer hover:bg-slate-200 px-1 py-0.5 rounded"
                                                onClick={() => startEditing(index, value)}
                                                title={value.toString()}
                                            >
                                                {value.toFixed(2)}
                                            </span>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>

                        <button
                            onClick={handlePredict}
                            disabled={loading || backendStatus === 'error' || features.length === 0}
                            className={`w-full py-4 rounded-lg text-white font-bold text-lg shadow-lg transition-all transform active:scale-95 ${
                                loading || backendStatus === 'error' || features.length === 0
                                ? 'bg-slate-400 cursor-not-allowed'
                                : 'bg-indigo-600 hover:bg-indigo-700 hover:shadow-xl'
                            }`}
                        >
                            {loading ? 'Analyzing Traffic...' : 'Analyze Traffic Pattern'}
                        </button>
                    </div>

                    {/* Result Display */}
                    {result && (
                        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100 animate-fade-in">
                            <h2 className="text-xl font-bold text-slate-800 mb-4">Analysis Result</h2>

                            <div className={`p-6 rounded-xl border-l-8 flex items-center justify-between ${
                                result.predicted_label === 'BENIGN' || result.status === 'error'
                                ? 'bg-green-50 border-green-500'
                                : 'bg-red-50 border-red-500'
                            }`}>
                                <div>
                                    <h3 className="text-2xl font-extrabold uppercase tracking-wider mb-1">
                                        {result.predicted_label || 'Error'}
                                    </h3>
                                    <p className="text-slate-600">
                                        {result.status === 'success' ? (
                                            <>
                                                Confidence: <span className="font-mono font-bold">{(result.confidence * 100).toFixed(2)}%</span>
                                                {attackFrequency > 0 && result.predicted_label !== 'BENIGN' && (
                                                    <span className="block text-sm text-slate-500 mt-1">
                                                        攻击频率: {attackFrequency} 次/{timeWindow}秒
                                                    </span>
                                                )}
                                            </>
                                        ) : (
                                            <span className="text-red-600">{result.message || 'Unknown error'}</span>
                                        )}
                                    </p>
                                </div>
                                {result.status === 'success' && (
                                    <div className="text-right">
                                        <span className="block text-xs uppercase text-slate-500 font-bold mb-1">Threat Level</span>
                                        <span className={`inline-block px-4 py-1 rounded-full text-sm font-bold border ${getLevelColor(result.threat_level)}`}>
                                            {result.threat_level}
                                        </span>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Performance Metrics Chart */}
                    {metrics && (
                        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
                            <h2 className="text-xl font-bold text-slate-800 mb-4">Model Performance (RF)</h2>
                            <div className="h-64 w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={chartData}>
                                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                                        <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#64748b'}} />
                                        <YAxis hide />
                                        <Tooltip
                                            cursor={{fill: '#f1f5f9'}}
                                            contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}}
                                        />
                                        <Bar dataKey="value" fill="#6366f1" radius={[4, 4, 0, 0]} barSize={50} label={{ position: 'top', fill: '#64748b', fontSize: 12 }} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    )}
                </div>

                {/* Right Column: Alert Logs and History */}
                <div className="lg:col-span-4 space-y-8">
                    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100 h-full max-h-[800px] flex flex-col">
                        <div className="flex items-center justify-between mb-6">
                            <h2 className="text-xl font-bold text-slate-800">Threat Intelligence</h2>
                            <span className="text-xs bg-red-100 text-red-600 px-2 py-1 rounded-full font-bold animate-pulse">LIVE</span>
                        </div>

                        {/* Tabs for Alerts and History */}
                        <div className="flex border-b border-slate-200 mb-4">
                            <button
                                className={`py-2 px-4 font-medium text-sm ${activeTab === 'alerts' ? 'border-b-2 border-indigo-500 text-indigo-600' : 'text-slate-500'}`}
                                onClick={() => setActiveTab('alerts')}
                            >
                                Recent Alerts
                            </button>
                            <button
                                className={`py-2 px-4 font-medium text-sm ${activeTab === 'history' ? 'border-b-2 border-indigo-500 text-indigo-600' : 'text-slate-500'}`}
                                onClick={() => setActiveTab('history')}
                            >
                                Detection History
                            </button>
                        </div>

                        <div className="flex-1 overflow-y-auto space-y-3 pr-2 custom-scrollbar">
                            {activeTab === 'alerts' ? (
                                <>
                                    <h3 className="text-md font-bold text-slate-700">Recent Alerts (Live)</h3>
                                    {alerts.length === 0 ? (
                                        <div className="text-center text-slate-400 py-10">
                                            <p>No threats detected yet.</p>
                                            {backendStatus === 'error' && (
                                                <p className="text-xs mt-2">后端连接失败，无法获取警报</p>
                                            )}
                                        </div>
                                    ) : (
                                        alerts.map((alert, idx) => (
                                            <div key={idx} className="p-4 rounded-lg bg-slate-50 border border-slate-100 hover:bg-slate-100 transition-colors">
                                                <div className="flex justify-between items-start mb-2">
                                                    <span className={`px-2 py-0.5 rounded text-xs font-bold border ${getLevelColor(alert.level)}`}>
                                                        {alert.level === 'None' ? 'NORMAL' : alert.level + ' PRIORITY'}
                                                    </span>
                                                    <span className="text-xs text-slate-400">{alert.timestamp?.split(' ')[1] || 'N/A'}</span>
                                                </div>
                                                <p className="font-bold text-slate-800">{alert.type}</p>
                                                <p className="text-xs text-slate-500 mt-1">Confidence Score: {alert.confidence ? (alert.confidence * 100).toFixed(1) + '%' : 'N/A'}</p>
                                            </div>
                                        ))
                                    )}
                                </>
                            ) : (
                                <>
                                    <h3 className="text-md font-bold text-slate-700">Detection History</h3>
                                    {history.length === 0 ? (
                                        <div className="text-center text-slate-400 py-10">
                                            <p>No detection history available.</p>
                                            {backendStatus === 'error' && (
                                                <p className="text-xs mt-2">后端连接失败，无法获取历史记录</p>
                                            )}
                                        </div>
                                    ) : (
                                        history.map((record, idx) => (
                                            <div key={idx} className="p-4 rounded-lg bg-slate-50 border border-slate-100 hover:bg-slate-100 transition-colors">
                                                <div className="flex justify-between items-start mb-2">
                                                    <span className={`px-2 py-0.5 rounded text-xs font-bold border ${getLevelColor(record.level)}`}>
                                                        {record.level !== 'None' ? record.level + ' PRIORITY' : 'NORMAL'}
                                                    </span>
                                                    <span className="text-xs text-slate-400">{record.timestamp?.split(' ')[1] || 'N/A'}</span>
                                                </div>
                                                <p className="font-bold text-slate-800">{record.type}</p>
                                                <p className="text-xs text-slate-500 mt-1">Confidence Score: {record.confidence ? (record.confidence * 100).toFixed(1) + '%' : 'N/A'}</p>
                                            </div>
                                        ))
                                    )}
                                </>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default App;