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

    // Initial Data Load
    useEffect(() => {
        fetchSampleData();
        fetchAlerts();
        fetchHistory();
        fetchMetrics();
        // Set up polling for alerts every 5 seconds
        const interval = setInterval(fetchAlerts, 5000);
        return () => clearInterval(interval);
    }, []);

    const fetchSampleData = async () => {
        try {
            const res = await fetch(`${API_URL}/sample`);
            const data = await res.json();
            setFeatures(data.features);
            setFeatureNames(data.feature_names);
        } catch (error) {
            console.error("Error fetching sample:", error);
        }
    };

    const fetchRandomData = async () => {
        try {
            const res = await fetch(`${API_URL}/random`);
            const data = await res.json();
            setFeatures(data.features);
            setFeatureNames(data.feature_names);
        } catch (error) {
            console.error("Error fetching random data:", error);
            // Fallback to generating client-side random data
            const newFeatures = features.map(() => Math.random() * 100000 - 50000);
            setFeatures(newFeatures);
        }
    };

    const fetchAttackData = async () => {
        try {
            const res = await fetch(`${API_URL}/simulate-attack`);
            const data = await res.json();
            setFeatures(data.features);
            setFeatureNames(data.feature_names);
        } catch (error) {
            console.error("Error fetching attack simulation data:", error);
            // Fallback to generating client-side attack-like data
            const newFeatures = features.map((_, i) => i % 3 === 0 ? Math.random() * 100000 : Math.random() * 1000);
            setFeatures(newFeatures);
        }
    };

    const fetchAlerts = async () => {
        try {
            const res = await fetch(`${API_URL}/alerts`);
            const data = await res.json();
            setAlerts(data);
        } catch (error) {
            console.error("Error fetching alerts:", error);
        }
    };

    const fetchHistory = async () => {
        try {
            const res = await fetch(`${API_URL}/history`);
            const data = await res.json();
            setHistory(data);
        } catch (error) {
            console.error("Error fetching history:", error);
        }
    };

    const fetchMetrics = async () => {
        try {
            const res = await fetch(`${API_URL}/performance`);
            const data = await res.json();
            setMetrics(data);
        } catch (error) {
            console.error("Error fetching metrics:", error);
        }
    };

    const handlePredict = async () => {
        setLoading(true);
        try {
            const res = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ features: features })
            });
            const data = await res.json();
            setResult(data);
            // Refresh alerts and history immediately
            fetchAlerts();
            fetchHistory();
        } catch (error) {
            console.error("Prediction error:", error);
        } finally {
            setLoading(false);
        }
    };

    const handleRetrain = async () => {
        if (!confirm("This will trigger the training script on the server. Continue?")) return;
        setRetraining(true);
        try {
            const res = await fetch(`${API_URL}/retrain`, { method: 'POST' });
            const data = await res.json();
            alert(data.message + (data.status === 'success' ? "\nPlease restart backend to load new model." : ""));
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

            const data = await res.json();
            
            if (data.status === 'success') {
                alert("Model retrained successfully with uploaded data!");
                // Refresh metrics and sample data
                fetchMetrics();
                fetchSampleData();
            } else {
                alert(`Retraining failed: ${data.message}`);
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
            case 'High': return 'bg-red-100 text-red-800 border-red-200';
            case 'Medium': return 'bg-orange-100 text-orange-800 border-orange-200';
            case 'Low': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
            default: return 'bg-green-100 text-green-800 border-green-200';
        }
    };

    // Prepare chart data
    const chartData = metrics ? [
        { name: 'Accuracy', value: metrics.accuracy * 100 },
        { name: 'Precision', value: metrics.precision * 100 },
        { name: 'Recall', value: metrics.recall * 100 },
        { name: 'F1 Score', value: metrics.f1_score * 100 },
    ] : [];

    return (
        <div className="min-h-screen p-6 bg-slate-50 font-sans">
            <header className="mb-8 flex justify-between items-center bg-white p-6 rounded-xl shadow-sm border border-slate-100">
                <div>
                    <h1 className="text-3xl font-bold text-slate-800 tracking-tight">DDoS Defense Shield</h1>
                    <p className="text-slate-500 mt-1">Real-time Traffic Analysis & Threat Intelligence</p>
                </div>
                <div className="flex gap-4">
                     <button 
                        onClick={handleRetrain}
                        disabled={retraining}
                        className={`px-4 py-2 rounded-lg font-medium transition-colors ${retraining ? 'bg-gray-300' : 'bg-indigo-600 hover:bg-indigo-700 text-white'}`}
                    >
                        {retraining ? 'Training...' : 'Retrain Model'}
                    </button>
                    <div className="px-4 py-2 bg-green-50 text-green-700 rounded-lg border border-green-100 font-medium">
                        System Active
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
                            disabled={uploading || !selectedFiles || selectedFiles.length === 0}
                            className={`w-full py-2 rounded-lg font-medium transition-colors ${
                                uploading || !selectedFiles || selectedFiles.length === 0
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
                            disabled={loading}
                            className={`w-full py-4 rounded-lg text-white font-bold text-lg shadow-lg transition-all transform active:scale-95 ${loading ? 'bg-slate-400' : 'bg-indigo-600 hover:bg-indigo-700 hover:shadow-xl'}`}
                        >
                            {loading ? 'Analyzing Traffic...' : 'Analyze Traffic Pattern'}
                        </button>
                    </div>

                    {/* Result Display */}
                    {result && (
                        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100 animate-fade-in">
                            <h2 className="text-xl font-bold text-slate-800 mb-4">Analysis Result</h2>
                            
                            <div className={`p-6 rounded-xl border-l-8 flex items-center justify-between ${result.predicted_label === 'BENIGN' ? 'bg-green-50 border-green-500' : 'bg-red-50 border-red-500'}`}>
                                <div>
                                    <h3 className="text-2xl font-extrabold uppercase tracking-wider mb-1">
                                        {result.predicted_label}
                                    </h3>
                                    <p className="text-slate-600">
                                        Confidence: <span className="font-mono font-bold">{(result.confidence * 100).toFixed(2)}%</span>
                                    </p>
                                </div>
                                <div className="text-right">
                                    <span className="block text-xs uppercase text-slate-500 font-bold mb-1">Threat Level</span>
                                    <span className={`inline-block px-4 py-1 rounded-full text-sm font-bold border ${getLevelColor(result.threat_level)}`}>
                                        {result.threat_level}
                                    </span>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Performance Metrics Chart */}
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
                                        </div>
                                    ) : (
                                        alerts.map((alert, idx) => (
                                            <div key={idx} className="p-4 rounded-lg bg-slate-50 border border-slate-100 hover:bg-slate-100 transition-colors">
                                                <div className="flex justify-between items-start mb-2">
                                                    <span className={`px-2 py-0.5 rounded text-xs font-bold border ${getLevelColor(alert.level)}`}>
                                                        {alert.level} PRIORITY
                                                    </span>
                                                    <span className="text-xs text-slate-400">{alert.timestamp.split(' ')[1]}</span>
                                                </div>
                                                <p className="font-bold text-slate-800">{alert.type}</p>
                                                <p className="text-xs text-slate-500 mt-1">Confidence Score: {(alert.confidence * 100).toFixed(1)}%</p>
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
                                        </div>
                                    ) : (
                                        history.map((record, idx) => (
                                            <div key={idx} className="p-4 rounded-lg bg-slate-50 border border-slate-100 hover:bg-slate-100 transition-colors">
                                                <div className="flex justify-between items-start mb-2">
                                                    <span className={`px-2 py-0.5 rounded text-xs font-bold border ${getLevelColor(record.level)}`}>
                                                        {record.level !== 'None' ? record.level + ' PRIORITY' : 'NORMAL'}
                                                    </span>
                                                    <span className="text-xs text-slate-400">{record.timestamp.split(' ')[1]}</span>
                                                </div>
                                                <p className="font-bold text-slate-800">{record.type}</p>
                                                <p className="text-xs text-slate-500 mt-1">Confidence Score: {(record.confidence * 100).toFixed(1)}%</p>
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