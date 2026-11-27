import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";
import readline from "readline/promises";

import { stdin, stdout } from "process";

dotenv.config();

const rl = readline.createInterface({ input: stdin, output: stdout });

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
});

const messages = [];

async function bot() {
  while (true) {
    const userInput = await rl.question("\nUser: ");

    if (userInput.toLowerCase() === "exit") {
      break;
    }

    messages.push({ role: "user", parts: [{ text: userInput }] });

    const response = await ai.models.generateContentStream({
      model: "gemini-3.5-flash-lite",
      contents: messages,
      config: {
        systemInstruction:
          "Your role is to assist with database questions only. nothing else",
      },
    });

    let fullResponse = "";

    process.stdout.write("\nAI: ");
    for await (const chunk of response) {
      const text = chunk.text || "";
      process.stdout.write(text);
      fullResponse += text;
    }

    console.log();

    messages.push({ role: "model", parts: [{ text: fullResponse }] });
  }

  rl.close();
}

bot();
