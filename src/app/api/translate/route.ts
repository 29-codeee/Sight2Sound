import { Anthropic } from '@anthropic-ai/sdk';
import { NextResponse } from 'next/server';

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY || '',
});

export async function POST(req: Request) {
  try {
    const { landmarks } = await req.json();
    
    // MOCK RESPONSE: This bypasses the Anthropic 500 error
    const mockGestures = ["PEACE", "THUMBS_UP", "HELLO", "FIST"];
    const randomGesture = mockGestures[Math.floor(Math.random() * mockGestures.length)];
    
    return NextResponse.json({ translation: `MOCK: ${randomGesture}` });

} catch (error: any)  {
    console.error("DETAILED_ERROR:", error.message); // THIS PRINTS TO YOUR VS CODE TERMINAL
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}