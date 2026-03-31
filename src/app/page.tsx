"use client";
import { useEffect, useRef, useState } from 'react';
import { HandLandmarker, FilesetResolver, ObjectDetector } from "@mediapipe/tasks-vision";

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [landmarker, setLandmarker] = useState<HandLandmarker | null>(null);
  const [objectDetector, setObjectDetector] = useState<ObjectDetector | null>(null);
  const [mode, setMode] = useState<null | "deaf" | "blind">(null);
  const [translation, setTranslation] = useState("Waiting for gesture...");
  const [confidence, setConfidence] = useState(0);
  const [handDetected, setHandDetected] = useState(false);
  const [fps, setFps] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [modelsReady, setModelsReady] = useState(false);
  const [srStatus, setSrStatus] = useState("Loading AI systems…");
  const [liveCaption, setLiveCaption] = useState("");
  const modeRef = useRef<null | "deaf" | "blind">(null);
  const mutedRef = useRef(false);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const objectDetectorRef = useRef<ObjectDetector | null>(null);
  const rafIdRef = useRef<number | null>(null);
  const runningRef = useRef(false);
  const lastSpokenRef = useRef<string>("");
  const lastUiUpdateMsRef = useRef(0);
  const frameCountRef = useRef(0);
  const lastFpsSampleMsRef = useRef(0);
  const lastFpsUiUpdateMsRef = useRef(0);
  const lastObjectDetectMsRef = useRef(0);
  const lastObjectResultsRef = useRef<any>(null);
  const lastSpokenObjectRef = useRef<string>("");
  const lastSpokenObjectAtMsRef = useRef(0);
  const lastCaptionRef = useRef<string>("");
  const lastCaptionAtMsRef = useRef(0);
  const lastHoverPromptAtMsRef = useRef(0);
  const pendingUiRef = useRef<{ translation: string; confidence: number; handDetected: boolean }>({
    translation: "Waiting for gesture...",
    confidence: 0,
    handDetected: false,
  });

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
      landmarkerRef.current = hl;
      setLandmarker(hl);

      // ObjectDetector init (runs once)
      const od = await ObjectDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        scoreThreshold: 0.35,
        maxResults: 5,
      });
      objectDetectorRef.current = od;
      setObjectDetector(od);

      setModelsReady(true);
      setSrStatus("AI systems are ready.");
    };
    initVision();
  }, []);

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    mutedRef.current = isMuted;
  }, [isMuted]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea" || (target as any)?.isContentEditable) return;

      const key = e.key.toLowerCase();
      if (key === "m") {
        e.preventDefault();
        setIsMuted((m) => !m);
      } else if (key === "s") {
        e.preventDefault();
        if (modeRef.current) startCamera();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, modelsReady]);

  const classifyGesture = (landmarks: Array<{ x: number; y: number; z?: number }> | undefined) => {
    if (!landmarks || landmarks.length < 21) return { translation: "Waiting for gesture...", confidence: 0, handDetected: false };

    const p = (i: number) => landmarks[i];
    const idxTip = p(8);
    const midTip = p(12);
    const wrist = p(0);
    if (!idxTip || !midTip || !wrist) return { translation: "Waiting for gesture...", confidence: 0, handDetected: false };

    // Hand scale (normalized coords): use bounding box diagonal for robust thresholds.
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const pt of landmarks) {
      if (!pt) continue;
      if (pt.x < minX) minX = pt.x;
      if (pt.x > maxX) maxX = pt.x;
      if (pt.y < minY) minY = pt.y;
      if (pt.y > maxY) maxY = pt.y;
    }
    const dx = Math.max(maxX - minX, 1e-6);
    const dy = Math.max(maxY - minY, 1e-6);
    const diag = Math.sqrt(dx * dx + dy * dy);

    const dist = (a: { x: number; y: number }, b: { x: number; y: number }) => {
      const ddx = a.x - b.x;
      const ddy = a.y - b.y;
      return Math.sqrt(ddx * ddx + ddy * ddy);
    };

    const higherThan = (tipIdx: number, baseIdx: number, t: number) => {
      const tip = p(tipIdx);
      const base = p(baseIdx);
      if (!tip || !base) return false;
      return base.y - tip.y > t; // smaller y = higher
    };
    const folded = (tipIdx: number, baseIdx: number, t: number) => {
      const tip = p(tipIdx);
      const base = p(baseIdx);
      if (!tip || !base) return false;
      return tip.y - base.y > t;
    };

    // Common thresholds
    const upT = 0.06 * dy;
    const foldT = 0.04 * dy;

    // Helper groups
    const indexUp = higherThan(8, 6, upT);
    const middleUp = higherThan(12, 10, upT);
    const ringUp = higherThan(16, 14, upT);
    const pinkyUp = higherThan(20, 18, upT);

    const indexFolded = folded(8, 6, foldT);
    const middleFolded = folded(12, 10, foldT);
    const ringFolded = folded(16, 14, foldT);
    const pinkyFolded = folded(20, 18, foldT);

    const thumbTip = p(4);
    const thumbIp = p(3);
    const thumbMcp = p(2);
    const indexKnuckle = p(5);
    const indexTip = p(8);

    // --- ASL alphabet (simple heuristics) ---
    // A: all fingers folded + thumb tip to the right of index knuckle (5).
    if (thumbTip && indexKnuckle && indexFolded && middleFolded && ringFolded && pinkyFolded && thumbTip.x > indexKnuckle.x) {
      return { translation: "A", confidence: 0.82, handDetected: true };
    }

    // B: four fingers up + thumb folded across palm.
    // Heuristic: index/middle/ring/pinky up AND thumb tip sits left of index knuckle (across palm) and near palm center.
    const palmCenter = p(9);
    const thumbAcross =
      Boolean(thumbTip && indexKnuckle && palmCenter) &&
      thumbTip.x < indexKnuckle.x &&
      dist(thumbTip!, palmCenter!) < 0.35 * diag;
    if (indexUp && middleUp && ringUp && pinkyUp && thumbAcross) {
      return { translation: "B", confidence: 0.83, handDetected: true };
    }

    // L: Index up and thumb extended sideways (forming an L).
    // Heuristic: index up, middle/ring/pinky folded, and thumb far from palm center horizontally.
    const thumbExtendedSideways =
      Boolean(thumbTip && thumbMcp && palmCenter) &&
      Math.abs(thumbTip.x - thumbMcp!.x) > 0.18 * dx &&
      Math.abs(thumbTip.y - thumbMcp!.y) < 0.22 * dy &&
      dist(thumbTip!, palmCenter!) > 0.25 * diag;
    if (indexUp && middleFolded && ringFolded && pinkyFolded && thumbExtendedSideways) {
      return { translation: "L", confidence: 0.84, handDetected: true };
    }

    // C: "curved" fingers forming a C shape.
    // Heuristic: index/middle/ring/pinky tips lie between their knuckles (MCP) and bases (PIP) in y (neither fully up nor fully folded),
    // and fingertips are spread (not touching like O).
    const between = (tipIdx: number, knuckleIdx: number, baseIdx: number) => {
      const tip = p(tipIdx);
      const kn = p(knuckleIdx);
      const base = p(baseIdx);
      if (!tip || !kn || !base) return false;
      const lo = Math.min(kn.y, base.y);
      const hi = Math.max(kn.y, base.y);
      return tip.y > lo + 0.01 * dy && tip.y < hi - 0.01 * dy;
    };
    const cLike =
      between(8, 5, 6) &&
      between(12, 9, 10) &&
      between(16, 13, 14) &&
      between(20, 17, 18) &&
      Boolean(thumbTip && indexTip) &&
      dist(thumbTip!, indexTip!) > 0.16 * diag;
    if (cLike) return { translation: "C", confidence: 0.78, handDetected: true };

    // O (optional, per your note): thumb touching index/middle tip using distance formula.
    // This helps later letters too; keep it above PEACE/HELLO so "O" doesn't get misread.
    if (thumbTip) {
      const tToIndex = indexTip ? dist(thumbTip, indexTip) : Infinity;
      const tToMiddle = midTip ? dist(thumbTip, midTip) : Infinity;
      const touchT = 0.14 * diag;
      if (Math.min(tToIndex, tToMiddle) < touchT) {
        const c = Math.min(1, Math.max(0, 1 - Math.min(tToIndex, tToMiddle) / touchT));
        return { translation: "O", confidence: 0.7 + 0.25 * c, handDetected: true };
      }
    }

    // Rule 1: PEACE if index tip and middle tip are close together.
    const tipDist = dist(idxTip, midTip);
    const peaceThreshold = 0.18 * diag;
    if (tipDist < peaceThreshold) {
      const c = Math.min(1, Math.max(0, 1 - tipDist / peaceThreshold));
      return { translation: "PEACE", confidence: 0.65 + 0.35 * c, handDetected: true };
    }

    // Rule 2: HELLO if all tips are above their bases.
    // Tips: 4, 8, 12, 16, 20; Bases (per your rule style): 2, 6, 10, 14, 18
    const higher = (tipIdx: number, baseIdx: number) => {
      const tip = p(tipIdx);
      const base = p(baseIdx);
      if (!tip || !base) return false;
      return base.y - tip.y > upT; // smaller y = higher
    };

    const allUp =
      higher(4, 2) &&
      higher(8, 6) &&
      higher(12, 10) &&
      higher(16, 14) &&
      higher(20, 18);

    if (allUp) return { translation: "HELLO", confidence: 0.85, handDetected: true };

    return { translation: "Waiting for gesture...", confidence: 0.35, handDetected: true };
  };

  const commitUiFromPending = () => {
    const now = performance.now();
    // Avoid excessive React state churn; update at most ~60fps, but allow responsive feel.
    if (now - lastUiUpdateMsRef.current < 16) return;
    lastUiUpdateMsRef.current = now;

    const next = pendingUiRef.current;
    setTranslation(next.translation);
    setConfidence(next.confidence);
    setHandDetected(next.handDetected);
  };

  const speakObject = (name: string) => {
    if (modeRef.current !== "blind") return;
    if (mutedRef.current) return;
    if (!name) return;
    try {
      // If something is already speaking, don't interrupt (prevents chattiness).
      if (window.speechSynthesis.speaking) return;
      window.speechSynthesis.speak(new SpeechSynthesisUtterance(name));
    } catch {
      // ignore TTS failures
    }
  };

  const setCaptionGated = (caption: string, now: number) => {
    // Gate caption updates so they don't spam React.
    const minMs = 100;
    if (caption === lastCaptionRef.current && now - lastCaptionAtMsRef.current < 1000) return;
    if (now - lastCaptionAtMsRef.current < minMs) return;
    lastCaptionRef.current = caption;
    lastCaptionAtMsRef.current = now;
    setLiveCaption(caption);
  };

  // 3. The Vision Loop
  const predict = () => {
    if (!runningRef.current) return;

    const now = performance.now();
    // FPS sampling (cheap, no allocations). Update overlay a few times per second.
    if (lastFpsSampleMsRef.current === 0) lastFpsSampleMsRef.current = now;
    frameCountRef.current += 1;
    const sampleWindowMs = 500;
    if (now - lastFpsSampleMsRef.current >= sampleWindowMs) {
      const fpsNow = Math.round((frameCountRef.current * 1000) / (now - lastFpsSampleMsRef.current));
      frameCountRef.current = 0;
      lastFpsSampleMsRef.current = now;
      // Gate FPS state too so React doesn't churn.
      if (now - lastFpsUiUpdateMsRef.current >= 250) {
        lastFpsUiUpdateMsRef.current = now;
        setFps(fpsNow);
      }
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const currentMode = modeRef.current;
    const hl = landmarkerRef.current;
    const od = objectDetectorRef.current;
    if (video && canvas) {
      const results = currentMode === "deaf" && hl ? hl.detectForVideo(video, now) : null;
      const ctx = canvas.getContext("2d");

      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const currentLandmarks = results?.landmarks?.[0];

        // Object detection (throttled to keep 30-60 FPS)
        if (currentMode === "blind" && od && video.videoWidth > 0 && video.videoHeight > 0) {
          const detectEveryMs = 120; // ~8fps for objects; adjust as needed
          if (now - lastObjectDetectMsRef.current >= detectEveryMs) {
            lastObjectDetectMsRef.current = now;
            try {
              const objectResults = od.detectForVideo(video, now);
              lastObjectResultsRef.current = objectResults;

              // Smart TTS: only announce high-confidence objects, debounced + cooldown.
              const dets = (objectResults as any)?.detections as any[] | undefined;
              const top = dets?.[0];
              const cat = top?.categories?.[0];
              const name =
                typeof cat?.categoryName === "string"
                  ? cat.categoryName
                  : typeof cat?.displayName === "string"
                    ? cat.displayName
                    : "";
              const score = typeof cat?.score === "number" ? cat.score : 0;

              if (name && score > 0.7) {
                setCaptionGated(name, now);
                const cooldownMs = 5000;
                const lastName = lastSpokenObjectRef.current;
                const lastAt = lastSpokenObjectAtMsRef.current;
                const nameChanged = name !== lastName;
                const cooledDown = now - lastAt >= cooldownMs;

                if (nameChanged || cooledDown) {
                  lastSpokenObjectRef.current = name;
                  lastSpokenObjectAtMsRef.current = now;
                  speakObject(name);
                }
              }
            } catch {
              // ignore transient detector errors
            }
          }
        }

        // Draw object boxes (neon green)
        const objectResults = lastObjectResultsRef.current;
        const detections = currentMode === "blind" ? (objectResults?.detections as any[] | undefined) : undefined;
        if (detections && detections.length && video.videoWidth > 0 && video.videoHeight > 0) {
          const sx = canvas.width / video.videoWidth;
          const sy = canvas.height / video.videoHeight;

          ctx.save();
          ctx.strokeStyle = "#00ff88";
          ctx.lineWidth = 3;
          ctx.shadowColor = "rgba(0,255,136,0.55)";
          ctx.shadowBlur = 16;
          ctx.font = "14px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";

          for (const det of detections) {
            const bb = det?.boundingBox;
            if (!bb) continue;

            const x = (bb.originX ?? 0) * sx;
            const y = (bb.originY ?? 0) * sy;
            const w = (bb.width ?? 0) * sx;
            const h = (bb.height ?? 0) * sy;
            if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(w) || !Number.isFinite(h)) continue;
            if (w <= 1 || h <= 1) continue;

            ctx.strokeRect(x, y, w, h);

            const topCat = det?.categories?.[0];
            const label = typeof topCat?.categoryName === "string" ? topCat.categoryName : "object";
            const score = typeof topCat?.score === "number" ? topCat.score : undefined;
            const text = score != null ? `${label} ${(score * 100).toFixed(0)}%` : label;

            const padX = 8;
            const padY = 6;
            const textW = ctx.measureText(text).width;
            const boxW = textW + padX * 2;
            const boxH = 22;
            const tx = x;
            const ty = Math.max(0, y - boxH - 6);

            ctx.fillStyle = "rgba(0,255,136,0.14)";
            ctx.fillRect(tx, ty, boxW, boxH);
            ctx.strokeRect(tx, ty, boxW, boxH);
            ctx.fillStyle = "rgba(231,255,245,0.95)";
            ctx.shadowBlur = 0;
            ctx.fillText(text, tx + padX, ty + 16);
            ctx.shadowBlur = 16;
          }
          ctx.restore();
        }

        if (currentMode === "deaf" && currentLandmarks && currentLandmarks.length >= 21) {
          // Draw the dots (lightweight).
          ctx.fillStyle = "#60a5fa";
          for (const point of currentLandmarks) {
            ctx.beginPath();
            ctx.arc(point.x * canvas.width, point.y * canvas.height, 4, 0, 2 * Math.PI);
            ctx.fill();
          }

          const classified = classifyGesture(currentLandmarks);
          pendingUiRef.current = classified;
        } else {
          // In blind mode, we keep captions for objects; in deaf mode we reset.
          if (currentMode === "deaf") {
            pendingUiRef.current = { translation: "Waiting for gesture...", confidence: 0, handDetected: false };
          } else {
            pendingUiRef.current = { translation, confidence, handDetected };
          }
          if (currentMode !== "blind") setCaptionGated("", now);
        }

        commitUiFromPending();
      }
    }

    // Optimized: exactly one rAF loop, capped by display refresh (~60fps).
    rafIdRef.current = requestAnimationFrame(predict);
  };

  const startCamera = async () => {
    if (runningRef.current) return;
    if (!modelsReady) {
      setSrStatus("Loading AI systems…");
      return;
    }
    runningRef.current = true;

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", frameRate: { ideal: 60 } },
      audio: false,
    });
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.onloadeddata = () => {
        if (rafIdRef.current == null) rafIdRef.current = requestAnimationFrame(predict);
      };
    }
  };

  // If user picks a mode before models finish loading, start camera once ready.
  useEffect(() => {
    if (!modelsReady) return;
    if (!modeRef.current) return;
    if (runningRef.current) return;
    startCamera();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelsReady, mode]);

  useEffect(() => {
    return () => {
      runningRef.current = false;
      if (rafIdRef.current != null) cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
      if (videoRef.current) videoRef.current.onloadeddata = null;

      const video = videoRef.current;
      const stream = video?.srcObject as MediaStream | null;
      if (stream) {
        for (const t of stream.getTracks()) t.stop();
      }
      if (video) video.srcObject = null;
    };
  }, []);

  return (
    <main className="min-h-screen text-white bg-[radial-gradient(1200px_circle_at_20%_10%,rgba(59,130,246,0.25),transparent_55%),radial-gradient(900px_circle_at_90%_20%,rgba(168,85,247,0.25),transparent_55%),radial-gradient(800px_circle_at_40%_90%,rgba(16,185,129,0.18),transparent_55%),linear-gradient(to_bottom,#020617,#030712)]">
      {/* Screen-reader announcements */}
      <div className="sr-only" aria-live="polite">
        {srStatus}
      </div>

      {mode === null && (
        <div className="fixed inset-0 z-50 grid place-items-center bg-slate-950/70 backdrop-blur-xl">
          <div className="mx-auto w-full max-w-lg rounded-3xl border border-white/10 bg-white/5 p-8 text-center shadow-2xl">
            <h2 className="text-3xl font-black tracking-tight">
              <span className="bg-gradient-to-r from-sky-300 via-blue-400 to-violet-400 bg-clip-text text-transparent">
                Choose Mode
              </span>
            </h2>
            <p className="mt-3 text-slate-200/70">
              Sign Language mode is optimized for text. Object Detection mode is optimized for voice.
            </p>

            <div className="mt-6 grid grid-cols-1 gap-3">
              <button
                type="button"
                onClick={() => {
                  setMode("deaf");
                  setIsMuted(true); // disable Speech API in deaf mode
                  setSrStatus(modelsReady ? "Sign Language mode selected." : "Loading AI systems…");
                  startCamera();
                }}
                className="w-full rounded-2xl bg-gradient-to-r from-sky-500 to-blue-600 px-6 py-4 text-lg font-black shadow-lg shadow-sky-500/20 transition hover:brightness-110 active:scale-[0.99]"
                aria-describedby="ai-ready-hint"
              >
                Sign Language Mode
              </button>

              <button
                type="button"
                onClick={() => {
                  setMode("blind");
                  setIsMuted(false); // enable Speech API in blind mode
                  setSrStatus(modelsReady ? "Object Detection mode selected." : "Loading AI systems…");
                  startCamera();
                }}
                onMouseEnter={() => {
                  const now = performance.now();
                  if (now - lastHoverPromptAtMsRef.current < 2000) return;
                  lastHoverPromptAtMsRef.current = now;
                  try {
                    window.speechSynthesis.cancel();
                    window.speechSynthesis.speak(new SpeechSynthesisUtterance("Object Detection Mode"));
                  } catch {
                    // ignore TTS failures
                  }
                }}
                onFocus={() => {
                  const now = performance.now();
                  if (now - lastHoverPromptAtMsRef.current < 2000) return;
                  lastHoverPromptAtMsRef.current = now;
                  try {
                    window.speechSynthesis.cancel();
                    window.speechSynthesis.speak(new SpeechSynthesisUtterance("Object Detection Mode"));
                  } catch {
                    // ignore TTS failures
                  }
                }}
                className="w-full rounded-2xl bg-gradient-to-r from-violet-500 to-emerald-500 px-6 py-4 text-lg font-black shadow-lg shadow-violet-500/20 transition hover:brightness-110 active:scale-[0.99]"
                aria-describedby="ai-ready-hint"
              >
                Object Detection Mode
              </button>
            </div>

            <p id="ai-ready-hint" className="mt-3 text-sm text-slate-200/60">
              {modelsReady ? "AI systems are ready." : "Loading AI systems…"}
            </p>
          </div>
        </div>
      )}

      <div className="mx-auto max-w-6xl px-6 py-10">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h1 className="text-4xl sm:text-5xl font-black tracking-tight">
              <span className="bg-gradient-to-r from-sky-300 via-blue-400 to-violet-400 bg-clip-text text-transparent">
                Sight2Sound
              </span>
            </h1>
            <p className="mt-2 text-slate-300/80">
              Real-time sign-to-speech with on-device landmark logic.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2 backdrop-blur-xl">
              <p className="text-xs font-mono tracking-widest text-slate-200/80">
                STATUS: <span className={handDetected ? "text-emerald-300" : "text-slate-300"}>{handDetected ? "HAND DETECTED" : "IDLE"}</span>
              </p>
            </div>
            <button
              onClick={startCamera}
              className="rounded-full bg-gradient-to-r from-sky-500 to-violet-500 px-6 py-3 font-bold shadow-lg shadow-sky-500/20 transition hover:brightness-110 active:scale-[0.98]"
            >
              Start Camera
            </button>
          </div>
        </div>

        <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-12">
          <div className="lg:col-span-7">
            <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-white/5 backdrop-blur-xl shadow-2xl">
              <div className="pointer-events-none absolute inset-0 rounded-3xl ring-2 ring-sky-400/30 shadow-[0_0_40px_rgba(56,189,248,0.25)]" />
              <div className="relative aspect-video">
                <video ref={videoRef} autoPlay playsInline className="absolute inset-0 h-full w-full object-cover mirror" />
                <canvas ref={canvasRef} width={640} height={360} className="absolute inset-0 h-full w-full pointer-events-none" />

                {/* Performance Overlay */}
                <div className="absolute right-3 top-3 flex items-start gap-2">
                  <div className="rounded-xl border border-white/10 bg-black/40 px-3 py-2 backdrop-blur-md">
                    <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-200/70">Performance</p>
                    <p className="mt-0.5 text-sm font-mono text-slate-100">{fps} FPS</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setIsMuted((m) => !m)}
                    className="rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-xs font-semibold text-slate-100 backdrop-blur-md hover:bg-black/50"
                    aria-pressed={isMuted}
                    aria-label={isMuted ? "Unmute voice guidance" : "Mute voice guidance"}
                  >
                    {isMuted ? "Unmute" : "Mute"}
                  </button>
                </div>
              </div>

              {/* Live Captions */}
              <div
                className="border-t border-white/10 bg-black/20 px-5 py-4"
                aria-live="polite"
                aria-label="Live captions"
              >
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-300/70">Live Captions</p>
                <p className="mt-2 text-2xl font-black tracking-tight text-slate-50">
                  {mode === "blind" ? (liveCaption || "—") : (translation || "—")}
                </p>
              </div>

              <div className="border-t border-white/10 p-5">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-300/70">Confidence Meter</p>
                    <p className="mt-1 text-sm text-slate-200/80">
                      {handDetected ? "Tracking landmarks" : "No hand detected"}
                    </p>
                  </div>
                  <p className="text-sm font-mono text-slate-200/70">{Math.round(confidence * 100)}%</p>
                </div>
                <div className="mt-3 h-3 w-full overflow-hidden rounded-full bg-white/10">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-emerald-400 via-sky-400 to-violet-400 transition-[width] duration-100"
                    style={{ width: `${Math.round(Math.min(1, Math.max(0, confidence)) * 100)}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="lg:col-span-5">
            <div className="rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur-xl shadow-2xl">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-300/70">Translation Card</p>
              <div className="mt-4 rounded-2xl bg-gradient-to-br from-sky-500/20 via-violet-500/15 to-emerald-500/15 p-6 ring-1 ring-white/10">
                <p className="text-4xl font-black tracking-tight">
                  <span className="bg-gradient-to-r from-white via-slate-100 to-slate-200 bg-clip-text text-transparent">
                    {translation}
                  </span>
                </p>
                <p className="mt-3 text-sm text-slate-200/70">
                  Zero-latency: gesture classification happens inside the video results loop (no API call).
                </p>
              </div>

              <div className="mt-6 grid grid-cols-2 gap-3 text-sm">
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <p className="text-slate-200/80 font-semibold">PEACE</p>
                  <p className="mt-1 text-slate-200/60">Index & middle tips close</p>
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <p className="text-slate-200/80 font-semibold">HELLO</p>
                  <p className="mt-1 text-slate-200/60">All tips above bases</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}