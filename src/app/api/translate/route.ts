import { Anthropic } from '@anthropic-ai/sdk';

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY, 
});

export async function POST(req: Request) {
  try {
    const { landmarks } = await req.json();

    const response = await anthropic.messages.create({
      model: "claude-3-5-sonnet-20240620",
      max_tokens: 100,
      messages: [
        {
          role: "user",
          content: `You are the brain of Sight2Sound-AI. I will give you 3D hand coordinates (MediaPipe landmarks). 
          Tell me what sign or gesture this represents in ONE or TWO words max. 
          Data: ${JSON.stringify(landmarks)}`
        }
      ],
    });

    // @ts-ignore
    return Response.json({ translation: response.content[0].text });
  } catch (error) {
    return Response.json({ error: "Claude is tired!" }, { status: 500 });
  }
}