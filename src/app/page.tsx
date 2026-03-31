"use client";
import { useEffect, useRef, useState } from 'react';
import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [landmarker, setLandmarker] = useState<HandLandmarker | null>(null);
  const [translation, setTranslation] = useState("Waiting for gesture...");
  const [isProcessing, setIsProcessing] = useState(false);

  // 1. Initialize MediaPipe (The Eyes)
  useEffect(() => {
    const initVision = async () => {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
      );
      const hl = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1
      });
      setLandmarker(hl);
    };
    initVision();
  }, []);

  // 2. The "Brain" Connection (Calling Claude)
  const syncWithClaude = async (landmarks: any) => {
    if (isProcessing) return; // Don't spam the API
    setIsProcessing(true);

    try {
      const res = await fetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ landmarks }),
      });
      const data = await res.json();
      
      if (data.translation) {
        setTranslation(data.translation);
        // Text-to-Speech (The Sound)
        const speech = new SpeechSynthesisUtterance(data.translation);
        window.speechSynthesis.speak(speech);
      }
    } catch (err) {
      console.error("Sync Error:", err);
    }

    // Wait 3 seconds before allowing another request
    setTimeout(() => setIsProcessing(false), 3000);
  };

  // 3. The Vision Loop
  const predict = async () => {
    if (landmarker && videoRef.current && canvasRef.current) {
      const results = landmarker.detectForVideo(videoRef.current, performance.now());
      const ctx = canvasRef.current.getContext("2d");
      
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        if (results.landmarks && results.landmarks.length > 0) {
          const currentLandmarks = results.landmarks[0];
          
          // Draw the dots
          currentLandmarks.forEach(point => {
            ctx.fillStyle = "#3b82f6";
            ctx.beginPath();
            ctx.arc(point.x * canvasRef.current!.width, point.y * canvasRef.current!.height, 5, 0, 2 * Math.PI);
            ctx.fill();
          });

          // If we aren't busy, send the data to Claude
          if (!isProcessing) {
            syncWithClaude(currentLandmarks);
          }
        }
      }
    }
    requestAnimationFrame(predict);
  };

  const startCamera = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.addEventListener("loadeddata", predict);
    }
  };

  return (
    <main className="flex flex-col items-center justify-center min-h-screen bg-slate-950 text-white p-6">
      <div className="text-center mb-6">
        <h1 className="text-4xl font-black text-blue-500 tracking-tighter uppercase">Sight2Sound AI</h1>
        <div className="mt-2 px-4 py-1 bg-blue-500/10 border border-blue-500/20 rounded-full inline-block">
          <p className="text-xs font-mono text-blue-400">STATUS: {isProcessing ? "THINKING..." : "READY"}</p>
        </div>
      </div>

      <div className="relative w-full max-w-2xl aspect-video rounded-3xl overflow-hidden border-4 border-slate-800 shadow-2xl">
        <video ref={videoRef} autoPlay playsInline className="absolute inset-0 w-full h-full object-cover mirror" />
        <canvas ref={canvasRef} width={640} height={360} className="absolute inset-0 w-full h-full pointer-events-none" />
        
        {/* Real-time Overlay */}
        <div className="absolute bottom-4 left-4 right-4 bg-slate-900/90 backdrop-blur-md p-4 rounded-2xl border border-slate-700">
          <p className="text-sm text-slate-400 uppercase font-bold tracking-widest mb-1">Translation</p>
          <p className="text-2xl font-bold text-white">{translation}</p>
        </div>
      </div>

      <button 
        onClick={startCamera}
        className="mt-8 bg-blue-600 hover:bg-blue-500 active:scale-95 px-12 py-4 rounded-full font-bold transition-all shadow-lg shadow-blue-600/20"
      >
        INITIALIZE SENSORY SYNC
      </button>
    </main>
  );
}