import { NextResponse } from 'next/server';

export async function POST(req: Request) {
  try {
    const { landmarks } = await req.json();

    // ISE Logic: We simulate the AI by checking the y-coordinate of the index finger tip (8) 
    // vs the index finger base (5). If tip is higher (smaller Y), it's a "POINT".
    const indexTipY = landmarks[8].y;
    const indexBaseY = landmarks[5].y;
    const thumbTipY = landmarks[4].y;

    let gesture = "SCANNING...";

    if (indexTipY < indexBaseY && thumbTipY > indexTipY) {
      gesture = "POINTING UP";
    } else if (indexTipY > indexBaseY) {
      gesture = "FIST / CLOSED";
    } else {
      gesture = "OPEN PALM";
    }

    // We return this exactly like Claude would, so your frontend doesn't break!
    return NextResponse.json({ translation: gesture });

  } catch (error: any) {
    return NextResponse.json({ error: "Mock logic failed" }, { status: 500 });
  }
}