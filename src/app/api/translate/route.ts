import { NextResponse } from 'next/server';

export async function POST(req: Request) {
  try {
    const body = await req.json().catch(() => ({}));
    const landmarksUnknown = (body as any)?.landmarks;

    type Point = { x: number; y: number; z?: number };

    const toPoint = (v: unknown): Point | null => {
      if (!v) return null;
      if (Array.isArray(v)) {
        const [x, y, z] = v as unknown[];
        if (typeof x !== 'number' || typeof y !== 'number') return null;
        if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
        if (typeof z === 'number' && Number.isFinite(z)) return { x, y, z };
        return { x, y };
      }
      if (typeof v === 'object') {
        const obj = v as Record<string, unknown>;
        const x = obj.x;
        const y = obj.y;
        const z = obj.z;
        if (typeof x !== 'number' || typeof y !== 'number') return null;
        if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
        if (typeof z === 'number' && Number.isFinite(z)) return { x, y, z };
        return { x, y };
      }
      return null;
    };

    const lm: Array<Point | null> = Array.isArray(landmarksUnknown)
      ? (landmarksUnknown as unknown[]).slice(0, 21).map(toPoint)
      : [];

    const p = (idx: number): Point | null => (idx >= 0 && idx < lm.length ? lm[idx] : null);

    // If Mediapipe-style normalized coords: smaller y means "higher" on the image.
    const points = lm.filter((pt): pt is Point => Boolean(pt));
    let translation = 'UNKNOWN';

    if (points.length >= 8) {
      let minX = Infinity,
        maxX = -Infinity,
        minY = Infinity,
        maxY = -Infinity;
      for (const pt of points) {
        if (pt.x < minX) minX = pt.x;
        if (pt.x > maxX) maxX = pt.x;
        if (pt.y < minY) minY = pt.y;
        if (pt.y > maxY) maxY = pt.y;
      }
      const scale = Math.max(maxX - minX, maxY - minY, 1e-6);

      const isHigherThan = (tipIdx: number, baseIdx: number, threshold: number): boolean => {
        const tip = p(tipIdx);
        const base = p(baseIdx);
        if (!tip || !base) return false;
        return base.y - tip.y > threshold;
      };

      const isFolded = (tipIdx: number, baseIdx: number, threshold: number): boolean => {
        const tip = p(tipIdx);
        const base = p(baseIdx);
        if (!tip || !base) return false;
        return tip.y - base.y > threshold;
      };

      const higher = 0.08 * scale;
      const significantHigher = 0.18 * scale;
      const folded = 0.04 * scale;

      // Indices per your rules: Index tip=8/base=6, Middle tip=12/base=10
      // Ring tip=16/base=14, Pinky tip=20/base=18, Thumb tip=4, Wrist=0
      const indexUp = isHigherThan(8, 6, higher);
      const middleUp = isHigherThan(12, 10, higher);
      const ringFolded = isFolded(16, 14, folded);
      const pinkyFolded = isFolded(20, 18, folded);

      const indexSignificantlyUp = isHigherThan(8, 6, significantHigher);
      const middleSignificantlyUp = isHigherThan(12, 10, significantHigher);

      const wrist = p(0);
      const thumbTip = p(4);
      const thumbUp = Boolean(wrist && thumbTip && wrist.y - thumbTip.y > higher);

      const middleFolded = isFolded(12, 10, folded);
      const indexFolded = isFolded(8, 6, folded);

      // Rule priority: more specific gestures first.
      if (indexSignificantlyUp && middleSignificantlyUp && ringFolded && pinkyFolded) {
        translation = 'PEACE';
      } else if (
        thumbUp &&
        isFolded(8, 6, folded) &&
        isFolded(12, 10, folded) &&
        ringFolded &&
        pinkyFolded
      ) {
        translation = 'THUMBS_UP';
      } else if (
        indexUp &&
        !middleUp &&
        !isHigherThan(16, 14, higher) &&
        !isHigherThan(20, 18, higher)
      ) {
        translation = 'POINTING';
      } else if (
        isHigherThan(8, 6, higher) &&
        isHigherThan(12, 10, higher) &&
        isHigherThan(16, 14, higher) &&
        isHigherThan(20, 18, higher)
      ) {
        translation = 'OPEN PALM';
      } else if (indexFolded && middleFolded && ringFolded && pinkyFolded) {
        translation = 'UNKNOWN';
      }
    }

    return NextResponse.json({ translation });

  } catch (error: any) {
    return NextResponse.json({ translation: 'UNKNOWN' }, { status: 200 });
  }
}