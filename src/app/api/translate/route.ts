import { Anthropic } from '@anthropic-ai/sdk';
import { NextResponse } from 'next/server';

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY || '',
});

export async function POST(req: Request) {
  try {
    const { landmarks } = await req.json();
    
    const response = await anthropic.messages.create({
      model: "claude-3-5-sonnet-20240620",
      max_tokens: 10,
      messages: [{ role: "user", content: `Identify this gesture: ${JSON.stringify(landmarks)}` }],
    });

    // @ts-ignore
    return NextResponse.json({ translation: response.content[0].text });
  } catch (error: any) {
    console.error("DETAILED_ERROR:", error.message); // THIS PRINTS TO YOUR VS CODE TERMINAL
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}