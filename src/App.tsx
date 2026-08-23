import React, { useState, useRef, useEffect } from 'react';
import { 
  Upload, 
  Cpu, 
  Activity, 
  ShieldAlert, 
  Zap, 
  ArrowUpRight,
  Thermometer, 
  Droplets, 
  Wind, 
  BarChart3, 
  MessageSquare,
  RefreshCw,
  AlertTriangle,
  Sparkles,
  Send,
  Leaf,
  ShieldCheck,
  CheckCircle2,
  AlertCircle,
  Camera,
  Microscope,
  Download,
  X,
  FlaskConical,
  Database,
  Play,
  Users,
  MapPin,
  BrainCircuit,
  History as HistoryIcon,
  Image as ImageIcon,
  ChevronDown,
  ChevronUp,
  FileText
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { motion, AnimatePresence } from 'motion/react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { Toaster, toast } from 'sonner';
import Markdown from 'react-markdown';

// Firebase & Gemini Services
import { generateHealthyReference, speakDiagnosis, searchGrounding, mapsGrounding, translateAdvisory, analyzeCropImages, fetchRealMarketData } from './services/geminiService';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface PredictionResult {
  disease: string;
  confidence: number;
  crop_health_score?: number;
  spread_risk?: string;
  remedies?: {
    organic: string;
    chemical: string;
  };
  remediation_steps: string[];
  yield_prediction?: string;
  weather_risk?: string;
  nearby_disease_trends?: string;
  additional_info?: string;
  advisory_status?: "active" | "unavailable";
  expert_advice?: {
    explanation: string;
    treatment: {
      organic: string;
      chemical: string;
    };
    yield_impact: string;
    prevention: string;
  };
  primaryDiagnosis: {
    disease: string;
    confidence: number;
    source: string;
  };
  cnn?: {
    model: string;
    prediction: string;
    confidence: number;
    probabilities?: Record<string, number>;
  };
  quantum?: {
    available: boolean;
    backend?: string;
    num_qubits?: number;
    prediction?: string;
    confidence?: number;
    probabilities?: Record<string, number>;
    pca_features?: number[];
    basis_probabilities?: number[];
    reason?: string;
  };
  hybrid?: {
    available: boolean;
    prediction?: string;
    confidence?: number;
    probabilities?: Record<string, number>;
    fusion_model?: string;
    reason?: string;
  };
  comparison?: {
    classical: {
      prediction: string;
      confidence: number;
    };
    hybrid: {
      prediction: string;
      confidence: number;
    };
    improvement: string;
  };
  timestamp?: number;
}

interface ValidationMetrics {
  status: string;
  model: string;
  dataset_dir: string;
  num_classes: number;
  classes: string[];
  samples: { total: number; train: number; validation: number; test: number };
  device: string;
  epochs: number;
  test_metrics: { accuracy: number; precision_macro: number; recall_macro: number; f1_macro: number };
  training_seconds: number;
  checkpoint: string;
}

interface ProjectLimitations {
  hardware: string[];
  software: string[];
  impact: {
    early_detection: string;
    yield_protection: string;
    sustainability: string;
  };
}

interface WeatherData {
  temperature: string;
  humidity: string;
  windSpeed: string;
  riskLevel: string;
  timestamp: string;
}

const COMPARISON_DATA = [
  { name: 'Exp A: MobileNetV2 (1280D)', accuracy: 0.9950, f1: 0.9906 },
  { name: 'Exp B: PCA-4 Control Head', accuracy: 0.6923, f1: 0.5259 },
  { name: 'Exp C: 4-Qubit VQC Head', accuracy: 0.0987, f1: 0.0047 },
  { name: 'Exp D: Learned Hybrid Fusion', accuracy: 0.9942, f1: 0.9896 },
];

const MARKET_VECTORS = [
  { name: 'Rice', price: '$450/t', trend: 'up', change: '+8%', suggestion: 'HOLD', profitability: 88, forecast: '2 weeks' },
  { name: 'Wheat', price: '$320/t', trend: 'down', change: '-3%', suggestion: 'SELL', profitability: 62, forecast: '1 week' },
  { name: 'Corn', price: '$210/t', trend: 'up', change: '+5%', suggestion: 'HOLD', profitability: 75, forecast: '3 weeks' },
  { name: 'Tomato', price: '$1.2/kg', trend: 'up', change: '+12%', suggestion: 'BUY', profitability: 92, forecast: '4 days' },
];

export default function App() {
  const [images, setImages] = useState<string[]>([]);
  const [isScanning, setIsScanning] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [chatInput, setChatInput] = useState("");
  const [chatHistory, setChatHistory] = useState<{role: string, text: string, sources?: any[]}[]>([]);
  const [activeTab, setActiveTab] = useState<'scanner' | 'lab' | 'community'>('scanner');
  const [isValidating, setIsValidating] = useState(false);
  const [isChatting, setIsChatting] = useState(false);
  const [isFarmerMode, setIsFarmerMode] = useState(false);
  const [useRealQuantum, setUseRealQuantum] = useState(false);
  const [marketVectors, setMarketVectors] = useState<any[]>(MARKET_VECTORS);
  const [isFetchingMarket, setIsFetchingMarket] = useState(false);
  const [validationMetrics, setValidationMetrics] = useState<ValidationMetrics | null>(null);
  const [limitations, setLimitations] = useState<ProjectLimitations | null>(null);
  const [labTab, setLabTab] = useState<'training' | 'validation' | 'impact'>('training');
  const [communityPosts, setCommunityPosts] = useState<any[]>([
    {
      id: '1',
      uid: 'mock-1',
      authorName: 'Dr. Aris Thorne',
      disease: 'Late Blight Alert',
      severity: 'high',
      message: 'Detected high concentration of Phytophthora infestans in the northern sector. Recommend immediate copper-based fungicide application.',
      location: '42.3601, -71.0589',
      timestamp: new Date().toISOString()
    },
    {
      id: '2',
      uid: 'mock-2',
      authorName: 'Farmer John',
      disease: 'Early Blight Observation',
      severity: 'medium',
      message: 'Seeing some yellowing on the lower leaves of my tomato plants. Seems like early blight is starting to spread.',
      location: '42.3601, -71.0589',
      timestamp: new Date(Date.now() - 86400000).toISOString()
    }
  ]);
  const [newPost, setNewPost] = useState("");
  const [healthyRef, setHealthyRef] = useState<string | null>(null);
  const [isGeneratingRef, setIsGeneratingRef] = useState(false);
  const [location, setLocation] = useState<{lat: number, lng: number} | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [language, setLanguage] = useState<'en' | 'te' | 'hi'>('en');
  const [logs, setLogs] = useState<{msg: string, type: 'info' | 'error' | 'quantum' | 'cnn'}[]>([]);
  const [translatedResult, setTranslatedResult] = useState<any>(null);
  const [showFullReport, setShowFullReport] = useState(false);

  const addLog = (msg: string, type: 'info' | 'error' | 'quantum' | 'cnn' = 'info') => {
    const timestamp = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    setLogs(prev => [{ msg: `[${timestamp}] ${msg}`, type }, ...prev].slice(0, 50));
  };

  // Translation effect
  useEffect(() => {
    if (result && language !== 'en') {
      handleTranslation();
    } else {
      setTranslatedResult(null);
    }
  }, [result, language]);

  const handleTranslation = async () => {
    if (!result) return;
    const langMap = { te: 'Telugu', hi: 'Hindi', en: 'English' };
    addLog(`Translating advisory to ${langMap[language]}...`, 'info');
    try {
      const translatedText = await translateAdvisory(
        `Disease: ${result.disease}. Remedies: ${result.remedies?.organic}. Steps: ${result.remediation_steps.join(', ')}`,
        langMap[language]
      );
      setTranslatedResult(translatedText);
      addLog(`Translation complete.`, 'info');
    } catch (e) {
      addLog(`Translation failed.`, 'error');
    }
  };

  // Geolocation
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => setLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
        (err) => console.warn("Geolocation denied:", err)
      );
    }
  }, []);

  useEffect(() => {
    fetchValidationMetrics();
    fetchLimitations();
  }, []);

  const fetchValidationMetrics = async () => {
    try {
      const res = await fetch('/api/validation');
      const data = await res.json();
      if (data.status !== 'unvalidated') {
        setValidationMetrics(data);
      }
    } catch (e) {
      console.error("Failed to fetch validation metrics:", e);
    }
  };

  const fetchLimitations = async () => {
    try {
      const res = await fetch('/api/limitations');
      const data = await res.json();
      setLimitations(data);
    } catch (e) {
      console.error("Failed to fetch limitations:", e);
    }
  };

  const startValidationTraining = async () => {
    setIsValidating(true);
    try {
      const res = await fetch('/api/validation/train', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        fetchValidationMetrics();
        toast.success("Hybrid training cycle complete.");
      } else {
        setError(data.message || "Validation training failed");
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setIsValidating(false);
    }
  };

  const [weatherData, setWeatherData] = useState<WeatherData | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (isCameraActive) {
      startCamera();
    } else {
      stopCamera();
    }
    return () => stopCamera();
  }, [isCameraActive]);

  const downloadReport = () => {
    if (!result) return;

    const diseaseName = result.primaryDiagnosis?.disease || result.disease;
    const diseaseConf = (result.primaryDiagnosis?.confidence || result.confidence).toFixed(1);

    const reportContent = `
QUANTUMCROP AI - VERIFIED ANALYSIS REPORT
=========================================
Generated: ${new Date().toLocaleString()}

PRIMARY CROP DIAGNOSIS
----------------------
Disease Detected: ${diseaseName}
Confidence: ${diseaseConf}%
Model Backbone: ${result.primaryDiagnosis?.source || 'MobileNetV2 (1280D)'}
Crop Health Score: ${result.crop_health_score ?? 'N/A'}/100
Weather Risk: ${weatherData?.available ? weatherData.riskLevel : 'Unavailable'}

AI PATHOLOGY ADVISORY (GEMINI)
------------------------------
Advisory Status: ${result.advisory_status === 'active' ? 'Active' : 'Offline'}
Explanation: ${result.expert_advice?.explanation || 'N/A'}
Organic Remedy: ${result.remedies?.organic || 'N/A'}
Chemical Remedy: ${result.remedies?.chemical || 'N/A'}
Prevention Strategy: ${result.expert_advice?.prevention || 'N/A'}

REMEDIATION STEPS
-----------------
${result.remediation_steps?.map((step: string, i: number) => `${i + 1}. ${step}`).join('\n') || 'No steps provided.'}

QUANTUM & HYBRID ANALYSIS
-------------------------
Quantum VQC Available: ${result.quantum?.available ? 'Yes' : 'No'}
4-Qubit VQC Prediction: ${result.quantum?.prediction || 'N/A'} (${result.quantum?.confidence ? (result.quantum.confidence * 100).toFixed(2) : 'N/A'}%)
Quantum Backend: ${result.quantum?.backend || 'N/A'}
Hybrid Fusion Prediction: ${result.hybrid?.prediction || 'N/A'} (${result.hybrid?.confidence ? (result.hybrid.confidence * 100).toFixed(2) : 'N/A'}%)
Fusion Model: ${result.hybrid?.fusion_model || 'Learned Hybrid Fusion'}

-----------------------------------------
QuantumCrop AI Node: Active // Genuine Pipeline
    `.trim();

    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `QuantumCrop_Report_${diseaseName.replace(/\s+/g, '_')}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    toast.success("Report downloaded successfully.");
  };

  const startCamera = async () => {
    try {
      const constraints = { 
        video: { 
          facingMode: { ideal: 'environment' } 
        } 
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error("Camera access denied:", err);
      setError("Camera access denied. Please check permissions.");
      setIsCameraActive(false);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  const captureImage = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg');
        setImages(prev => [...prev, dataUrl]);
        setIsCameraActive(false);
        setResult(null);
        setError(null);
      }
    }
  };

  useEffect(() => {
    const fetchWeather = async () => {
      try {
        const query = location ? `?lat=${location.lat}&lng=${location.lng}` : '';
        const response = await fetch(`/api/weather${query}`);
        const data = await response.json();
        setWeatherData(data);
      } catch (err) {
        console.error("Failed to fetch weather telemetry:", err);
      }
    };

    fetchWeather();
    const interval = setInterval(fetchWeather, 30000); // Update every 30s
    return () => clearInterval(interval);
  }, [location]);

  useEffect(() => {
    if (activeTab === 'community' && marketVectors === MARKET_VECTORS) {
      const loadMarketData = async () => {
        setIsFetchingMarket(true);
        const data = await fetchRealMarketData();
        if (data && data.length > 0) {
          setMarketVectors(data);
        }
        setIsFetchingMarket(false);
      };
      loadMarketData();
    }
  }, [activeTab]);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    processFiles(files);
  };

  const processFiles = (files: File[]) => {
    if (files.length > 0) {
      files.forEach(file => {
        const reader = new FileReader();
        reader.onloadend = () => {
          setImages(prev => [...prev, reader.result as string]);
          setResult(null);
          setError(null);
        };
        reader.readAsDataURL(file);
      });
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    processFiles(files);
  };

  const startScan = async (imageIndex?: number) => {
    const imagesToScan = imageIndex !== undefined ? [images[imageIndex]] : images;
    if (imagesToScan.length === 0) return;
    setIsScanning(true);
    setError(null);
    setHealthyRef(null);
    setShowFullReport(false);
    setLogs([]); // Clear logs on new scan

    try {
      addLog("Transmitting leaf image to ML inference backend...", "info");
      addLog("Executing MobileNetV2 1280D feature extraction & disease classification...", "cnn");
      
      const qRes = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ images: imagesToScan }), 
      });
      
      if (!qRes.ok) {
        const errData = await qRes.json();
        addLog(`Inference Error: ${errData.error}`, "error");
        throw new Error(errData.error || "Analysis failed");
      }
      
      const qData = await qRes.json();
      addLog(`MobileNetV2 Primary Diagnosis: ${qData.primaryDiagnosis.disease} (${qData.primaryDiagnosis.confidence.toFixed(1)}%)`, "cnn");

      if (qData.quantum?.available) {
        addLog(`4-Qubit VQC Circuit optimized (Prediction: ${qData.quantum.prediction}, Conf: ${(qData.quantum.confidence * 100).toFixed(2)}%)`, "quantum");
      }
      if (qData.hybrid?.available) {
        addLog(`Learned Hybrid Fusion evaluated (Prediction: ${qData.hybrid.prediction})`, "quantum");
      }
      
      addLog("Querying Gemini expert pathology advisory...", "info");
      const aiAnalysis = await analyzeCropImages(imagesToScan, qData);
      addLog(aiAnalysis.advisory_status === "active" ? "Gemini Expert Advisory ready." : "AI advisory offline (Standard agronomic guidelines shown).", "info");

      setResult({
        ...qData,
        ...aiAnalysis,
        disease: qData.primaryDiagnosis.disease,
        confidence: qData.primaryDiagnosis.confidence,
      });
      setChatHistory([]);

      toast.success("Scan complete.");
    } catch (err: any) {
      setError(err.message || 'System Error: Analysis Interrupted');
      toast.error("Scan failed: " + (err.message || "Unknown error"));
      console.error(err);
    } finally {
      setIsScanning(false);
    }
  };

  const handleChat = async () => {
    if (!chatInput.trim() || !result) return;
    
    const userMsg = chatInput;
    setChatInput("");
    setChatHistory(prev => [...prev, { role: 'user', text: userMsg }]);
    setIsChatting(true);

    try {
      let aiResponse;
      const languageNames = {
        en: 'English',
        te: 'Telugu',
        hi: 'Hindi'
      };

      const systemInstruction = `You are the Quantum Agronomist Assistant, an expert in plant pathology, quantum computing applications in agriculture, and sustainable farming. 
      You have access to the results of a Hybrid Quantum-Classical analysis of a crop.
      
      CURRENT CROP DATA:
      - Disease: ${result.disease}
      - Confidence: ${result.confidence}%
      - Organic Remedy: ${result.remedies?.organic}
      - Chemical Remedy: ${result.remedies?.chemical}
      - Remediation Steps: ${result.remediation_steps?.join(', ')}
      - Crop Health Score: ${result.crop_health_score}/100
      - Weather Risk: ${result.weather_risk}
      
      IMPORTANT: You MUST respond in ${languageNames[language]}. 
      Even if the user asks in another language, your primary response language should be ${languageNames[language]}.
      
      Your goal is to help the farmer understand these results, provide deeper insights into the quantum analysis, and offer practical, actionable advice for crop recovery and future prevention.
      Be professional, encouraging, and highly technical yet accessible. Use grounding tools when asked about specific locations or latest research.`;

      if (userMsg.toLowerCase().includes("where") || userMsg.toLowerCase().includes("near")) {
        aiResponse = await mapsGrounding(userMsg, location?.lat, location?.lng, systemInstruction);
      } else {
        // Use search grounding with context from the result and system instruction
        aiResponse = await searchGrounding(userMsg, systemInstruction);
      }
      
      setChatHistory(prev => [...prev, { 
        role: 'model', 
        text: aiResponse.text || "I'm processing the data...",
        sources: aiResponse.sources
      }]);
    } catch (err) {
      console.error(err);
      setChatHistory(prev => [...prev, { role: 'model', text: "Connection to AI Core interrupted." }]);
    } finally {
      setIsChatting(false);
    }
  };

  const handlePost = async () => {
    if (!newPost.trim()) return;
    setCommunityPosts(prev => [
      {
        id: Date.now().toString(),
        uid: 'user-mock',
        authorName: 'You',
        disease: 'General Observation',
        severity: 'low',
        message: newPost,
        location: location ? `${location.lat},${location.lng}` : "Unknown",
        timestamp: new Date().toISOString()
      },
      ...prev
    ]);
    setNewPost("");
    toast.success("Post shared with community.");
  };

  const generateReference = async () => {
    if (!result) return;
    setIsGeneratingRef(true);
    try {
      const url = await generateHealthyReference(result.disease.split(' ')[0]); // Get crop name
      setHealthyRef(url);
      toast.success("Healthy reference image generated.");
    } catch (err) {
      toast.error("Failed to generate reference.");
    } finally {
      setIsGeneratingRef(false);
    }
  };

  return (
    <div className="min-h-screen relative">
      <div className="atmospheric-bg" />
      
      <div className="max-w-7xl mx-auto p-4 md:p-8 space-y-12 relative z-10">
        {/* Header */}
        <header className="flex items-center justify-between glass-morphism p-6 rounded-3xl">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-[#00ff88]/10 rounded-2xl border border-[#00ff88]/30">
              <Cpu className="w-8 h-8 text-[#00ff88]" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tighter uppercase italic text-white">QuantumCrop AI</h1>
              <div className="flex items-center gap-2">
                <p className="text-[10px] font-mono opacity-60 uppercase tracking-widest text-white">Hybrid Quantum-Classical Pathology Grid</p>
                <span className="px-2 py-0.5 bg-[#00ff88]/20 border border-[#00ff88]/40 text-[#00ff88] text-[8px] font-bold rounded uppercase tracking-tighter">
                  Real Models Active
                </span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-bold uppercase tracking-tighter opacity-40 text-white">Lang:</span>
              <select 
                value={language} 
                onChange={(e) => setLanguage(e.target.value as any)}
                className="bg-white/10 border border-white/20 rounded px-2 py-1 text-[10px] font-bold text-white outline-none cursor-pointer"
              >
                <option value="en" className="bg-black text-white">EN</option>
                <option value="te" className="bg-black text-white">తెలుగు</option>
                <option value="hi" className="bg-black text-white">हिंदी</option>
              </select>
            </div>

            <div className="flex bg-white/10 p-1 rounded-2xl border border-white/20">
              <button 
                onClick={() => setActiveTab('scanner')}
                className={cn(
                  "px-4 py-2 rounded-xl text-[10px] font-bold uppercase tracking-widest transition-all flex items-center gap-2",
                  activeTab === 'scanner' ? "bg-[#00ff88] text-black" : "text-white hover:bg-white/10"
                )}
              >
                <Activity className="w-3 h-3" />
                Scanner
              </button>
              <button 
                onClick={() => setActiveTab('lab')}
                className={cn(
                  "px-4 py-2 rounded-xl text-[10px] font-bold uppercase tracking-widest transition-all flex items-center gap-2",
                  activeTab === 'lab' ? "bg-[#00ff88] text-black" : "text-white hover:bg-white/10"
                )}
              >
                <FlaskConical className="w-3 h-3" />
                Benchmarks & Lab
              </button>
            </div>
          </div>
        </header>

        {/* Hero Section (Only show on scanner tab when no result) */}
        {activeTab === 'scanner' && !result && !isScanning && images.length === 0 && (
          <section className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center py-12">
            <motion.div 
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              className="space-y-8"
            >
              <h2 className="hero-title text-white">
                Hybrid AI That <br />
                <span className="text-[#00ff88]">Tests Disease</span>
              </h2>
              <p className="hero-subtitle text-white/80">
                Hybrid AI-Powered Precision Diagnosis to Protect and Grow Your Plants With Ease.
              </p>
              <button 
                onClick={() => {
                  const el = document.getElementById('scanner-section');
                  if (el) el.scrollIntoView({ behavior: 'smooth' });
                  else fileInputRef.current?.click();
                }}
                className="glass-button px-8 py-4 rounded-2xl text-lg font-bold text-white flex items-center gap-3 group shadow-[0_0_30px_rgba(0,255,136,0.2)] hover:shadow-[0_0_50px_rgba(0,255,136,0.4)] transition-all"
              >
                Scan Your Plant
                <ArrowUpRight className="w-5 h-5 transition-transform group-hover:translate-x-1 group-hover:-translate-y-1" />
              </button>
            </motion.div>
            
            <motion.div 
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="relative aspect-square lg:aspect-video rounded-3xl overflow-hidden glass-morphism p-4"
            >
              <img 
                src="https://images.unsplash.com/photo-1597362214123-c5502ad053e6?q=80&w=1000&auto=format&fit=crop" 
                alt="Plant Leaf Close-up"
                className="w-full h-full object-cover rounded-2xl opacity-80"
                referrerPolicy="no-referrer"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent pointer-events-none" />
              <div className="absolute bottom-8 left-8 right-8 p-6 glass-morphism rounded-2xl">
                <div className="flex items-center gap-4">
                  <div className="p-2 bg-[#00ff88]/20 rounded-lg">
                    <Microscope className="w-6 h-6 text-[#00ff88]" />
                  </div>
                  <div>
                    <p className="text-xs font-mono uppercase tracking-widest text-[#00ff88]">Precision Diagnosis</p>
                    <p className="text-sm font-bold text-white">Hybrid Quantum-Classical Analysis</p>
                  </div>
                </div>
              </div>
            </motion.div>
          </section>
        )}

        <Toaster position="bottom-right" theme="dark" />

      {activeTab === 'scanner' ? (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Column: Scanner */}
            <div className="lg:col-span-7 space-y-6">
          <section id="scanner-section" className="glass-panel p-6 rounded-2xl relative overflow-hidden">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-[#00ff88]" />
                <h2 className="text-sm font-bold uppercase tracking-wider">AI Diagnostic Scanner</h2>
              </div>
              <div className="flex items-center gap-2">
                <button 
                  onClick={() => setIsCameraActive(!isCameraActive)}
                  className={cn(
                    "px-4 py-2 rounded-lg text-xs font-bold transition-all flex items-center gap-2",
                    isCameraActive 
                      ? "bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 text-red-500"
                      : "bg-[#00ff88]/10 hover:bg-[#00ff88]/20 border border-[#00ff88]/30 text-[#00ff88]"
                  )}
                >
                  {isCameraActive ? <X className="w-4 h-4" /> : <Camera className="w-4 h-4" />}
                  {isCameraActive ? "Cancel Camera" : "Use Camera"}
                </button>
                <button 
                  onClick={() => {
                    setImages([]);
                    setResult(null);
                    setError(null);
                  }}
                  className="px-4 py-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 rounded-lg text-xs font-bold transition-all flex items-center gap-2 text-red-500"
                >
                  <X className="w-4 h-4" />
                  Clear
                </button>
                <button 
                  onClick={() => fileInputRef.current?.click()}
                  className="px-4 py-2 bg-[#00ff88]/10 hover:bg-[#00ff88]/20 border border-[#00ff88]/30 rounded-lg text-xs font-bold transition-all flex items-center gap-2 text-[#00ff88]"
                >
                  <Upload className="w-4 h-4" />
                  Upload
                </button>
              </div>
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleImageUpload} 
                className="hidden" 
                accept="image/*"
                multiple
              />
            </div>

            <div 
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={cn(
                "relative aspect-video bg-black/40 rounded-xl border flex items-center justify-center overflow-hidden group transition-all duration-300",
                isDragging ? "border-[#00ff88] bg-[#00ff88]/5 scale-[1.01]" : "border-[#00ff88]/10"
              )}
            >
              {isDragging && (
                <div className="absolute inset-0 bg-[#00ff88]/10 backdrop-blur-sm flex flex-col items-center justify-center z-20 animate-in fade-in zoom-in duration-300">
                  <div className="p-6 bg-black/60 rounded-full neon-border mb-4">
                    <Upload className="w-12 h-12 text-[#00ff88] animate-bounce" />
                  </div>
                  <p className="text-sm font-bold text-[#00ff88] uppercase tracking-widest">Drop Images to Analyze</p>
                </div>
              )}
              {isCameraActive ? (
                <div className="relative w-full h-full">
                  <video 
                    ref={videoRef} 
                    autoPlay 
                    playsInline 
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 border-2 border-[#00ff88]/30 pointer-events-none">
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 border border-[#00ff88] opacity-50" />
                  </div>
                  <button 
                    onClick={captureImage}
                    className="absolute bottom-6 left-1/2 -translate-x-1/2 p-4 bg-[#00ff88] text-black rounded-full shadow-lg hover:scale-110 transition-transform"
                  >
                    <Camera className="w-6 h-6" />
                  </button>
                </div>
              ) : images.length > 0 ? (
                <div className="w-full h-full flex flex-col gap-4">
                  <div className={cn(
                    "grid gap-2 w-full p-4",
                    images.length === 1 ? "grid-cols-1" : 
                    images.length === 2 ? "grid-cols-2" : "grid-cols-3"
                  )}>
                    {images.map((img, i) => (
                      <div key={i} className="relative aspect-square rounded-lg overflow-hidden border border-white/10 group/img">
                        <img src={img} alt={`Scan ${i}`} className="w-full h-full object-cover opacity-80" />
                        <div className="absolute inset-0 bg-black/40 opacity-0 group-hover/img:opacity-100 transition-opacity flex items-center justify-center gap-2">
                          <button 
                            onClick={() => startScan(i)}
                            disabled={isScanning}
                            className="p-2 bg-[#00ff88] text-black rounded-lg hover:scale-110 transition-transform disabled:opacity-50"
                            title="Scan this image"
                          >
                            <Zap className="w-4 h-4" />
                          </button>
                          <button 
                            onClick={() => setImages(prev => prev.filter((_, idx) => idx !== i))}
                            className="p-2 bg-red-500 text-white rounded-lg hover:scale-110 transition-transform"
                            title="Remove image"
                          >
                            <X className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                  {isScanning && <div className="scanline" />}
                  {!isScanning && !result && (
                    <div className="px-4 pb-4">
                      <button 
                        onClick={() => startScan()}
                        className="w-full py-4 bg-[#00ff88] text-black rounded-xl font-bold uppercase tracking-widest flex items-center justify-center gap-2 hover:scale-[1.02] active:scale-[0.98] transition-all"
                      >
                        <Zap className="w-5 h-5" />
                        Run Hybrid Quantum Analysis ({images.length} Images)
                      </button>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center space-y-4 opacity-40">
                  <Upload className="w-12 h-12 mx-auto" />
                  <p className="text-xs font-mono uppercase tracking-widest">System Ready for Image Input</p>
                  <p className="text-[10px] font-mono uppercase tracking-widest opacity-60">Drag & Drop or use buttons above</p>
                </div>
              )}
              <canvas ref={canvasRef} className="hidden" />
            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-3 gap-4 mt-6">
              <div className="p-4 bg-black/20 rounded-xl border-l-2 border-l-red-500">
                <p className="text-[10px] font-mono uppercase opacity-50 mb-1">Pathogen Status</p>
                <p className="text-sm font-bold truncate">
                  {result ? result.disease : (isScanning ? "Analyzing..." : "Awaiting Scan")}
                </p>
              </div>
              <div className="p-4 bg-black/20 rounded-xl border-l-2 border-l-blue-500">
                <p className="text-[10px] font-mono uppercase opacity-50 mb-1">Confidence Level</p>
                <p className="text-sm font-bold">
                  {result ? `${result.confidence}%` : "--"}
                </p>
              </div>
              <div className="p-4 bg-black/20 rounded-xl border-l-2 border-l-orange-500">
                <p className="text-[10px] font-mono uppercase opacity-50 mb-1">Recommended Action</p>
                <p className="text-sm font-bold truncate">
                  {result ? "Remedy Available" : "--"}
                </p>
              </div>
            </div>

            {result && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-4"
              >
                <button 
                  onClick={downloadReport}
                  className="w-full py-3 bg-white/5 border border-white/10 hover:bg-white/10 rounded-xl text-[10px] font-bold uppercase tracking-widest flex items-center justify-center gap-2 transition-all group"
                >
                  <Download className="w-4 h-4 group-hover:translate-y-0.5 transition-transform" />
                  Download Full Analysis Report (.txt)
                </button>
              </motion.div>
            )}
          </section>

          {/* Quick Stats Summary */}
          <section className="glass-panel p-6 rounded-2xl border-l-4 border-l-[#00ff88]">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Leaf className="w-5 h-5 text-[#00ff88]" />
                <h2 className="text-sm font-bold uppercase tracking-wider text-white">Diagnostic Summary</h2>
              </div>
              {result && (
                <span className="px-2 py-0.5 bg-[#00ff88]/10 border border-[#00ff88]/30 text-[#00ff88] text-[9px] font-bold rounded uppercase">
                  Verified ML Model
                </span>
              )}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-black/40 rounded-xl border border-white/5">
                <p className="text-[10px] font-mono uppercase opacity-50 mb-1">Primary Diagnosis</p>
                <p className="text-sm font-bold text-white truncate">
                  {result ? (result.primaryDiagnosis?.disease || result.disease).replace(/___/g, ' - ').replace(/_/g, ' ') : (isScanning ? "Running ML Pipeline..." : "Awaiting Upload")}
                </p>
                <p className="text-[9px] font-mono text-blue-400 mt-1">MobileNetV2 (1280D)</p>
              </div>
              <div className="p-4 bg-black/40 rounded-xl border border-white/5">
                <p className="text-[10px] font-mono uppercase opacity-50 mb-1">Model Confidence</p>
                <p className="text-sm font-bold text-[#00ff88]">
                  {result ? `${(result.primaryDiagnosis?.confidence || result.confidence).toFixed(1)}%` : "--"}
                </p>
                <p className="text-[9px] font-mono opacity-40 mt-1">Held-out test set: 99.50%</p>
              </div>
              <div className="p-4 bg-black/40 rounded-xl border border-white/5">
                <p className="text-[10px] font-mono uppercase opacity-50 mb-1">Crop Health Score</p>
                <p className="text-sm font-bold text-amber-400">
                  {result ? `${result.crop_health_score ?? 80}/100` : "--"}
                </p>
                <p className="text-[9px] font-mono opacity-40 mt-1">Pathology Index</p>
              </div>
            </div>
          </section>
        </div>

        {/* Right Column: Analytics & Results */}
        <div className="lg:col-span-5 space-y-6">
          {/* Probability Matrix */}
          <section className="glass-panel p-6 rounded-2xl">
            <div className="flex items-center gap-2 mb-6">
              <Activity className="w-5 h-5 text-[#00ff88]" />
              <h2 className="text-sm font-bold uppercase tracking-wider text-white">Prediction Confidence</h2>
            </div>
            <div className="flex items-center justify-center py-4 relative">
              <div className="absolute text-center">
                <p className="text-3xl font-bold glow-text text-[#00ff88]">
                  {result ? (result.primaryDiagnosis?.confidence || result.confidence).toFixed(1) : 0}%
                </p>
                <p className="text-[10px] font-mono uppercase opacity-50 text-white">Confidence</p>
              </div>
              <ResponsiveContainer width="100%" height={180}>
                <PieChart>
                  <Pie
                    data={[
                      { name: 'Confidence', value: result ? (result.primaryDiagnosis?.confidence || result.confidence) : 0 },
                      { name: 'Remainder', value: result ? Math.max(0, 100 - (result.primaryDiagnosis?.confidence || result.confidence)) : 100 }
                    ]}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={75}
                    paddingAngle={4}
                    dataKey="value"
                    stroke="none"
                  >
                    <Cell fill="#00ff88" />
                    <Cell fill="rgba(255,255,255,0.05)" />
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="space-y-3 mt-4">
              <div className="p-3 bg-black/40 rounded-xl border border-white/5">
                <div className="flex justify-between items-center text-xs mb-1">
                  <span className="font-mono text-white/70 truncate mr-2">
                    {result ? (result.primaryDiagnosis?.disease || result.disease).replace(/___/g, ' - ').replace(/_/g, ' ') : 'No scan active'}
                  </span>
                  <span className="font-mono font-bold text-[#00ff88]">
                    {result ? `${(result.primaryDiagnosis?.confidence || result.confidence).toFixed(1)}%` : '0%'}
                  </span>
                </div>
                <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-[#00ff88]" 
                    style={{ width: `${result ? (result.primaryDiagnosis?.confidence || result.confidence) : 0}%` }} 
                  />
                </div>
              </div>
            </div>
          </section>

          {/* Local Telemetry (Weather) */}
          <section className="glass-panel p-6 rounded-2xl space-y-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Wind className="w-5 h-5 text-[#00ff88]" />
                <h2 className="text-sm font-bold uppercase tracking-wider text-white">Local Environmental Telemetry</h2>
              </div>
              <span className={cn(
                "px-2 py-0.5 text-[8px] font-bold rounded uppercase",
                weatherData?.available ? "bg-[#00ff88]/10 text-[#00ff88]" : "bg-white/5 text-white/40"
              )}>
                {weatherData?.available ? "Live OpenWeather" : "Telemetry Unavailable"}
              </span>
            </div>

            {weatherData?.available ? (
              <>
                <div className="grid grid-cols-1 gap-3">
                  {[
                    { icon: Thermometer, label: 'Temperature', val: `${weatherData.temperature}°C`, color: 'text-orange-500' },
                    { icon: Droplets, label: 'Humidity', val: `${weatherData.humidity}%`, color: 'text-blue-500' },
                    { icon: Wind, label: 'Wind Speed', val: `${weatherData.windSpeed} km/h`, color: 'text-emerald-500' },
                  ].map((stat, i) => (
                    <div key={i} className="flex items-center justify-between p-3 bg-black/20 rounded-xl border border-white/5">
                      <div className="flex items-center gap-3">
                        <stat.icon className={cn("w-4 h-4", stat.color)} />
                        <span className="text-xs font-medium opacity-70 text-white">{stat.label}</span>
                      </div>
                      <span className="text-xs font-mono font-bold text-white">{stat.val}</span>
                    </div>
                  ))}
                </div>
                <div className={cn(
                  "p-4 border rounded-xl transition-colors",
                  weatherData.riskLevel === 'CRITICAL' ? "bg-red-500/20 border-red-500/40" :
                  weatherData.riskLevel === 'HIGH' ? "bg-orange-500/20 border-orange-500/40" :
                  "bg-[#00ff88]/10 border-[#00ff88]/20"
                )}>
                  <div className="flex items-center gap-2 mb-1">
                    <ShieldAlert className={cn(
                      "w-4 h-4",
                      weatherData.riskLevel === 'CRITICAL' ? "text-red-500" :
                      weatherData.riskLevel === 'HIGH' ? "text-orange-500" :
                      "text-[#00ff88]"
                    )} />
                    <span className="text-[10px] font-bold uppercase text-white">
                      {weatherData.riskLevel} Environmental Risk
                    </span>
                  </div>
                  <p className="text-[10px] opacity-70 leading-relaxed text-white">
                    {weatherData.riskLevel === 'CRITICAL' ? "Elevated humidity and temperature indicate high pathogen outbreak likelihood." :
                     weatherData.riskLevel === 'HIGH' ? "Elevated conditions detected. Monitor foliar humidity." :
                     "Ambient environmental metrics currently within stable agronomic thresholds."}
                  </p>
                </div>
              </>
            ) : (
              <div className="p-4 bg-black/20 rounded-xl border border-white/5 text-center">
                <p className="text-xs text-white/50 mb-1">Weather telemetry unavailable</p>
                <p className="text-[10px] font-mono text-white/30">Configure OPENWEATHER_API_KEY in environment to enable live telemetry.</p>
              </div>
            )}
          </section>
        </div>
      </div>

      {/* Bottom Section: Primary Diagnosis, Quantum Analysis, Hybrid Analysis & AI Advisory */}
      <AnimatePresence>
        {result && (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-8 mt-6"
          >
            {/* Multimodal Arbitration Safety Interlock Banner */}
            {result.arbitration_status && result.arbitration_status !== "success" && (
              <div className={cn(
                "p-5 rounded-2xl border flex items-start gap-4 transition-all",
                result.arbitration_status === "crop_mismatch" ? "bg-amber-500/15 border-amber-500/40 text-amber-300" :
                result.arbitration_status === "not_a_leaf" ? "bg-red-500/15 border-red-500/40 text-red-300" :
                result.arbitration_status === "poor_quality" ? "bg-orange-500/15 border-orange-500/40 text-orange-300" :
                "bg-yellow-500/15 border-yellow-500/40 text-yellow-300"
              )}>
                <AlertTriangle className="w-6 h-6 shrink-0 mt-0.5 text-amber-400" />
                <div className="space-y-1 text-xs">
                  <p className="font-bold uppercase tracking-wider text-sm text-white">
                    {result.arbitration_status === "crop_mismatch" ? "Crop Identification Conflict Detected" :
                     result.arbitration_status === "not_a_leaf" ? "Invalid Input: No Plant Leaf Detected" :
                     result.arbitration_status === "poor_quality" ? "Low Image Quality" :
                     "Pathogen Diagnosis Uncertain"}
                  </p>
                  <p className="opacity-90 leading-relaxed text-white/90">
                    {result.arbitration_reason || "Crop or pathogen features require additional visual verification."}
                  </p>
                  {result.action_guidance && (
                    <p className="text-[11px] font-mono text-white/70 pt-1">
                      👉 Recommendation: {result.action_guidance}
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* Primary Diagnosis & Quantum/Hybrid Benchmark Grid */}
            <section className="glass-panel p-6 rounded-2xl">
              <div className="flex items-center justify-between mb-6 pb-4 border-b border-white/10">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-[#00ff88]/10 rounded-xl border border-[#00ff88]/30">
                    <Microscope className="w-6 h-6 text-[#00ff88]" />
                  </div>
                  <div>
                    <h2 className="text-lg font-bold uppercase tracking-wider text-white">Crop Diagnostic & Analysis Suite</h2>
                    <p className="text-xs font-mono opacity-50 text-white">End-to-End Real Model Inference</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <span className="px-3 py-1 bg-white/5 border border-white/10 rounded-lg text-xs font-mono text-[#00ff88]">
                    Classified: {(result.primaryDiagnosis?.disease || result.disease).replace(/___/g, ' - ').replace(/_/g, ' ')}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* 1. Primary Diagnosis (MobileNetV2) */}
                <div className="p-5 bg-black/40 rounded-2xl border border-blue-500/30 flex flex-col justify-between">
                  <div>
                    <div className="flex justify-between items-center mb-3">
                      <span className="text-[10px] font-mono uppercase text-blue-400 font-bold flex items-center gap-1.5">
                        <Cpu className="w-3.5 h-3.5" /> Primary Diagnosis
                      </span>
                      <span className="text-[9px] bg-blue-500/20 text-blue-300 px-2 py-0.5 rounded font-mono font-bold">
                        Production Model
                      </span>
                    </div>
                    <p className="text-base font-bold text-white mb-2">
                      {(result.primaryDiagnosis?.disease || result.disease).replace(/___/g, ' - ').replace(/_/g, ' ')}
                    </p>
                    <div className="space-y-2 mt-4">
                      <div className="flex justify-between text-xs">
                        <span className="opacity-60 text-white">Confidence:</span>
                        <span className="font-mono font-bold text-blue-400">
                          {(result.primaryDiagnosis?.confidence || result.confidence).toFixed(2)}%
                        </span>
                      </div>
                      <div className="flex justify-between text-xs">
                        <span className="opacity-60 text-white">Model Architecture:</span>
                        <span className="font-mono text-white/80">MobileNetV2 (1280D)</span>
                      </div>
                      <div className="flex justify-between text-xs">
                        <span className="opacity-60 text-white">Test Accuracy (10.8k test set):</span>
                        <span className="font-mono text-[#00ff88]">99.50%</span>
                      </div>
                    </div>
                  </div>
                  <div className="mt-4 pt-3 border-t border-white/5 text-[10px] opacity-40 italic text-white">
                    Primary agricultural classification engine with 1280D feature representations.
                  </div>
                </div>

                {/* 2. Quantum Analysis (4-Qubit VQC) */}
                <div className="p-5 bg-black/40 rounded-2xl border border-purple-500/30 flex flex-col justify-between">
                  <div>
                    <div className="flex justify-between items-center mb-3">
                      <span className="text-[10px] font-mono uppercase text-purple-400 font-bold flex items-center gap-1.5">
                        <Zap className="w-3.5 h-3.5" /> Experimental Quantum Analysis
                      </span>
                      <span className="text-[9px] bg-purple-500/20 text-purple-300 px-2 py-0.5 rounded font-mono font-bold">
                        4-Qubit VQC
                      </span>
                    </div>
                    {result.quantum?.available ? (
                      <>
                        <p className="text-base font-bold text-white mb-2">
                          {(result.quantum.prediction || 'Unknown').replace(/___/g, ' - ').replace(/_/g, ' ')}
                        </p>
                        <div className="space-y-2 mt-4">
                          <div className="flex justify-between text-xs">
                            <span className="opacity-60 text-white">VQC Confidence:</span>
                            <span className="font-mono font-bold text-purple-400">
                              {result.quantum.confidence ? (result.quantum.confidence * 100).toFixed(2) : 0}%
                            </span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span className="opacity-60 text-white">Simulator:</span>
                            <span className="font-mono text-white/80">Qiskit Statevector</span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span className="opacity-60 text-white">Circuit:</span>
                            <span className="font-mono text-white/80">ZZFeatureMap + RealAmplitudes</span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span className="opacity-60 text-white">Feature Bottleneck:</span>
                            <span className="font-mono text-amber-400">PCA-4 (320x compression)</span>
                          </div>
                        </div>
                      </>
                    ) : (
                      <div className="py-6 text-center text-white/40 text-xs">
                        Quantum VQC model artifacts unavailable.
                      </div>
                    )}
                  </div>
                  <div className="mt-4 pt-3 border-t border-white/5 text-[10px] opacity-40 italic text-white">
                    Evaluated on 4D Hilbert space projections. Benchmark against classical PCA control (Exp B: 69.23%).
                  </div>
                </div>

                {/* 3. Learned Hybrid Analysis */}
                <div className="p-5 bg-black/40 rounded-2xl border border-[#00ff88]/30 flex flex-col justify-between">
                  <div>
                    <div className="flex justify-between items-center mb-3">
                      <span className="text-[10px] font-mono uppercase text-[#00ff88] font-bold flex items-center gap-1.5">
                        <Sparkles className="w-3.5 h-3.5" /> Learned Hybrid Analysis
                      </span>
                      <span className="text-[9px] bg-[#00ff88]/20 text-[#00ff88] px-2 py-0.5 rounded font-mono font-bold">
                        Fusion Layer
                      </span>
                    </div>
                    {result.hybrid?.available ? (
                      <>
                        <p className="text-base font-bold text-white mb-2">
                          {(result.hybrid.prediction || result.primaryDiagnosis?.disease || result.disease).replace(/___/g, ' - ').replace(/_/g, ' ')}
                        </p>
                        <div className="space-y-2 mt-4">
                          <div className="flex justify-between text-xs">
                            <span className="opacity-60 text-white">Hybrid Confidence:</span>
                            <span className="font-mono font-bold text-[#00ff88]">
                              {result.hybrid.confidence ? (result.hybrid.confidence * 100).toFixed(2) : 0}%
                            </span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span className="opacity-60 text-white">Fusion Head:</span>
                            <span className="font-mono text-white/80">Learned MLP (54D input)</span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span className="opacity-60 text-white">Test Accuracy (10.8k test set):</span>
                            <span className="font-mono text-[#00ff88]">99.42%</span>
                          </div>
                        </div>
                      </>
                    ) : (
                      <div className="py-6 text-center text-white/40 text-xs">
                        Learned hybrid fusion model unavailable.
                      </div>
                    )}
                  </div>
                  <div className="mt-4 pt-3 border-t border-white/5 text-[10px] opacity-40 italic text-white">
                    Joint representation of classical logits & 16-state quantum basis probabilities.
                  </div>
                </div>
              </div>
            </section>

            {/* AI Pathology Advisory (Gemini) */}
            <section className="glass-panel p-6 rounded-2xl border-t-4 border-t-[#00ff88]">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="w-5 h-5 text-[#00ff88]" />
                  <h2 className="text-sm font-bold uppercase tracking-wider text-white">AI Pathology Advisory & Treatment Guidance</h2>
                </div>
                <span className={cn(
                  "px-2.5 py-1 text-[9px] font-bold rounded uppercase",
                  result.advisory_status === "active" ? "bg-[#00ff88]/20 text-[#00ff88] border border-[#00ff88]/40" : "bg-white/10 text-white/50"
                )}>
                  {result.advisory_status === "active" ? "Gemini Advisory Active" : "Standard Agronomic Guidelines"}
                </span>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Explanation & Symptoms */}
                <div className="space-y-4">
                  <div className="p-4 bg-black/40 rounded-xl border border-white/5">
                    <p className="text-[10px] font-mono uppercase opacity-50 mb-2 text-white">Pathologist Explanation</p>
                    <div className="text-xs leading-relaxed text-white/80 prose prose-invert prose-xs max-w-none">
                      {translatedResult ? (
                        <Markdown>{translatedResult}</Markdown>
                      ) : (
                        <Markdown>{result.expert_advice?.explanation || result.additional_info}</Markdown>
                      )}
                    </div>
                  </div>

                  <div className="p-4 bg-black/40 rounded-xl border border-white/5 flex justify-between items-center">
                    <div>
                      <p className="text-[10px] font-mono uppercase opacity-50 mb-1 text-white">Estimated Yield Impact</p>
                      <p className="text-xs font-bold text-orange-400">{result.expert_advice?.yield_impact || result.yield_prediction || "15-25% without intervention"}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-[10px] font-mono uppercase opacity-50 mb-1 text-white">Spread Risk</p>
                      <p className="text-xs font-bold text-amber-400">{result.spread_risk || "Medium"}</p>
                    </div>
                  </div>
                </div>

                {/* Treatment & Prevention */}
                <div className="space-y-4">
                  <div className="p-4 bg-[#00ff88]/5 rounded-xl border border-[#00ff88]/20">
                    <div className="flex items-center gap-2 mb-2">
                      <Leaf className="w-4 h-4 text-[#00ff88]" />
                      <p className="text-[10px] font-mono uppercase text-[#00ff88] font-bold">Organic Treatment</p>
                    </div>
                    <p className="text-xs leading-relaxed text-white/90">
                      {result.expert_advice?.treatment?.organic || result.remedies?.organic || "Isolate affected plants and apply organic neem oil spray (5ml/L)."}
                    </p>
                  </div>

                  <div className="p-4 bg-blue-500/5 rounded-xl border border-blue-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <FlaskConical className="w-4 h-4 text-blue-400" />
                      <p className="text-[10px] font-mono uppercase text-blue-400 font-bold">Chemical / Fungicide Intervention</p>
                    </div>
                    <p className="text-xs leading-relaxed text-white/90">
                      {result.expert_advice?.treatment?.chemical || result.remedies?.chemical || "Consult local agronomy extension for targeted fungicide application."}
                    </p>
                  </div>

                  <div className="p-4 bg-black/40 rounded-xl border border-white/5">
                    <p className="text-[10px] font-mono uppercase opacity-50 mb-1 text-white">Prevention Strategy</p>
                    <p className="text-xs leading-relaxed text-white/80">
                      {result.expert_advice?.prevention || "Implement crop rotation, maintain soil drainage, and use certified disease-free seeds."}
                    </p>
                  </div>
                </div>
              </div>

              {/* Step-by-Step Remediation Plan */}
              {result.remediation_steps && result.remediation_steps.length > 0 && (
                <div className="mt-6 pt-6 border-t border-white/10">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-[#00ff88] mb-4 flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4" /> Actionable Recovery Steps
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {result.remediation_steps.map((step, i) => (
                      <div key={i} className="p-3 bg-black/40 rounded-xl border border-white/5 flex gap-3 items-start">
                        <span className="w-5 h-5 rounded-full bg-[#00ff88]/20 text-[#00ff88] flex items-center justify-center font-mono text-[10px] font-bold flex-shrink-0 mt-0.5">
                          {i + 1}
                        </span>
                        <span className="text-xs text-white/80 leading-snug">{step}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Quantum Agronomist Chat */}
      {result && (
        <motion.section 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-panel p-6 rounded-2xl max-w-2xl mx-auto border-t-4 border-t-[#00ff88]"
        >
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-2">
              <MessageSquare className="w-5 h-5 text-[#00ff88]" />
              <h2 className="text-sm font-bold uppercase tracking-wider">Quantum Agronomist Assistant</h2>
            </div>
            {chatHistory.length > 0 && (
              <button 
                onClick={() => setChatHistory([])}
                className="text-[10px] font-mono uppercase opacity-40 hover:opacity-100 transition-opacity"
              >
                Clear Chat
              </button>
            )}
          </div>
          
          <div className="space-y-4 mb-6 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar">
            {chatHistory.length === 0 && (
              <p className="text-xs opacity-50 italic">Ask me about the quantum analysis or remediation steps...</p>
            )}
            {chatHistory.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[85%] p-3 rounded-xl text-xs ${
                  msg.role === 'user' 
                    ? 'bg-[#00ff88] text-black font-medium' 
                    : 'bg-white/5 border border-white/10'
                }`}>
                  <div className="prose prose-invert prose-xs max-w-none">
                    <Markdown>{msg.text}</Markdown>
                  </div>
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-2 pt-2 border-t border-white/10">
                      <p className="text-[10px] font-semibold text-white/40 uppercase tracking-wider mb-1">Sources</p>
                      <div className="flex flex-wrap gap-2">
                        {msg.sources.map((source: any, idx: number) => (
                          <a 
                            key={idx}
                            href={source.uri}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-[10px] text-[#00ff88] hover:underline flex items-center gap-1 bg-white/5 px-1.5 py-0.5 rounded"
                          >
                            <ArrowUpRight className="w-2 h-2" />
                            {source.title || 'Source'}
                          </a>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isChatting && (
              <div className="flex justify-start">
                <div className="bg-white/5 border border-white/10 p-3 rounded-xl text-xs animate-pulse">
                  Analyzing quantum states...
                </div>
              </div>
            )}
          </div>

          <div className="relative">
            <input 
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleChat()}
              placeholder="Ask the Agronomist..."
              className="w-full bg-black/40 border border-white/10 rounded-xl py-3 px-4 pr-12 text-xs focus:outline-none focus:border-[#00ff88]/50 transition-colors"
            />
            <button 
              onClick={handleChat}
              disabled={isChatting || !chatInput.trim()}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-2 text-[#00ff88] hover:bg-[#00ff88]/10 rounded-lg transition-colors disabled:opacity-50"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </motion.section>
      )}
        </>
      ) : (
        <div className="space-y-8">
          {/* Lab Header */}
          <div className="p-8 bg-black/40 rounded-3xl border border-[#00ff88]/20 relative overflow-hidden">
            <div className="absolute top-0 right-0 p-8 opacity-10">
              <FlaskConical className="w-32 h-32 text-[#00ff88]" />
            </div>
            <div className="relative z-10 max-w-3xl">
              <h2 className="text-2xl font-black uppercase tracking-tighter text-white mb-2">
                Empirical Evaluation & Model Benchmarks
              </h2>
              <p className="text-sm opacity-70 text-white leading-relaxed mb-6">
                Evaluated on the official held-out test split of 10,849 leaf images across 38 classes (PlantVillage Dataset).
                All metrics reflect genuine PyTorch MobileNetV2, Qiskit Statevector VQC, and Learned Hybrid Fusion inference.
              </p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-4 bg-white/5 rounded-2xl border border-white/10">
                  <p className="text-[10px] uppercase opacity-40 mb-1 text-white">MobileNetV2 Accuracy</p>
                  <p className="text-2xl font-mono font-bold text-[#00ff88]">99.50%</p>
                  <p className="text-[9px] opacity-40 text-white mt-1">1280D Backbone</p>
                </div>
                <div className="p-4 bg-white/5 rounded-2xl border border-white/10">
                  <p className="text-[10px] uppercase opacity-40 mb-1 text-white">Hybrid Fusion</p>
                  <p className="text-2xl font-mono font-bold text-blue-400">99.42%</p>
                  <p className="text-[9px] opacity-40 text-white mt-1">Learned MLP</p>
                </div>
                <div className="p-4 bg-white/5 rounded-2xl border border-white/10">
                  <p className="text-[10px] uppercase opacity-40 mb-1 text-white">PCA Control (4D)</p>
                  <p className="text-2xl font-mono font-bold text-yellow-400">69.23%</p>
                  <p className="text-[9px] opacity-40 text-white mt-1">Classical Baseline</p>
                </div>
                <div className="p-4 bg-white/5 rounded-2xl border border-white/10">
                  <p className="text-[10px] uppercase opacity-40 mb-1 text-white">4-Qubit VQC (4D)</p>
                  <p className="text-2xl font-mono font-bold text-purple-400">9.87%</p>
                  <p className="text-[9px] opacity-40 text-white mt-1">Quantum Head</p>
                </div>
              </div>
            </div>
          </div>

          {/* Benchmark Comparison Chart & Details */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <section className="glass-panel p-6 rounded-2xl">
              <div className="flex items-center gap-2 mb-6">
                <Activity className="w-5 h-5 text-[#00ff88]" />
                <h2 className="text-sm font-bold uppercase tracking-wider text-white">4-Experiment Accuracy Benchmark</h2>
              </div>
              <div className="h-[260px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={COMPARISON_DATA}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" vertical={false} />
                    <XAxis dataKey="name" stroke="#ffffff40" fontSize={9} tickLine={false} axisLine={false} />
                    <YAxis stroke="#ffffff40" fontSize={10} tickLine={false} axisLine={false} tickFormatter={(val) => `${(val * 100).toFixed(0)}%`} domain={[0, 1]} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid #ffffff10', borderRadius: '8px', fontSize: '10px' }}
                      itemStyle={{ color: '#00ff88' }}
                      formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, 'Accuracy']}
                    />
                    <Bar dataKey="accuracy" radius={[4, 4, 0, 0]}>
                      {COMPARISON_DATA.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={index === 0 ? '#00ff88' : index === 1 ? '#eab308' : index === 2 ? '#a855f7' : '#3b82f6'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="text-[10px] opacity-50 mt-4 leading-relaxed text-white">
                Evaluated across all 10,849 official test samples across 38 classes. MobileNetV2 (Exp A) achieves 99.50% accuracy. Under 320x compression to 4 features, the classical PCA control (Exp B) achieves 69.23% while the 4-qubit VQC (Exp C) achieves 9.87%. The Learned Hybrid Fusion (Exp D) achieves 99.42% test accuracy.
              </p>
            </section>

            <section className="glass-panel p-6 rounded-2xl">
              <div className="flex items-center gap-2 mb-6">
                <ShieldCheck className="w-5 h-5 text-[#00ff88]" />
                <h2 className="text-sm font-bold uppercase tracking-wider text-white">Dataset & Split Architecture</h2>
              </div>
              <div className="space-y-4">
                <div className="p-4 bg-black/40 rounded-xl border border-white/5 space-y-3">
                  <div className="flex justify-between items-center text-xs">
                    <span className="opacity-60 text-white">Source Dataset:</span>
                    <span className="font-mono font-bold text-white">PlantVillage (BrandonFors)</span>
                  </div>
                  <div className="flex justify-between items-center text-xs">
                    <span className="opacity-60 text-white">Total Dataset Samples:</span>
                    <span className="font-mono text-[#00ff88]">54,305 Images</span>
                  </div>
                  <div className="flex justify-between items-center text-xs">
                    <span className="opacity-60 text-white">Training Split:</span>
                    <span className="font-mono text-white">36,937 Images</span>
                  </div>
                  <div className="flex justify-between items-center text-xs">
                    <span className="opacity-60 text-white">Validation Split:</span>
                    <span className="font-mono text-white">6,519 Images</span>
                  </div>
                  <div className="flex justify-between items-center text-xs">
                    <span className="opacity-60 text-white">Official Held-out Test:</span>
                    <span className="font-mono text-blue-400">10,849 Images</span>
                  </div>
                  <div className="flex justify-between items-center text-xs">
                    <span className="opacity-60 text-white">Total Crop Disease Classes:</span>
                    <span className="font-mono text-[#00ff88]">38 Classes</span>
                  </div>
                </div>

                <div className="p-4 bg-[#00ff88]/5 rounded-xl border border-[#00ff88]/20">
                  <p className="text-[10px] uppercase font-bold text-[#00ff88] mb-1">Supported Crop Species (14 Total):</p>
                  <p className="text-[11px] text-white/80 leading-relaxed">
                    Tomato, Potato, Apple, Corn (Maize), Grape, Pepper (Bell), Peach, Cherry, Strawberry, Squash, Orange, Blueberry, Raspberry, Soybean.
                  </p>
                </div>
              </div>
            </section>
          </div>
        </div>
      )}

      {/* Error Toast */}
      <AnimatePresence>
        {error && (
          <motion.div 
            initial={{ opacity: 0, x: 100 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 100 }}
            className="fixed bottom-8 right-8 p-4 bg-red-500 text-white rounded-xl shadow-2xl flex items-center gap-3 z-50"
          >
            <AlertTriangle className="w-5 h-5" />
            <span className="text-xs font-bold uppercase tracking-wider">{error}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Footer Branding */}
      <footer className="text-center py-8 opacity-30">
        <p className="text-[10px] font-mono uppercase tracking-[0.5em]">
          &copy; 2026 QuantumCrop Technologies // Global Agriculture Defense Grid
        </p>
      </footer>
    </div>
    </div>
  );
}
