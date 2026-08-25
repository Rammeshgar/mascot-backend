import { GoogleGenAI } from "@google/genai";

const ALLOWED_ORIGINS = new Set([
    "https://rammeshgar.github.io",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]);

const apiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY || "";
const model = process.env.GEMINI_TRANSCRIBE_MODEL || process.env.GEMINI_MODEL || "gemini-3.6-flash";
const ai = apiKey ? new GoogleGenAI({ apiKey }) : null;
const ALLOWED_AUDIO_TYPES = new Set([
    "audio/webm",
    "audio/ogg",
    "audio/mp4",
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
]);

function requestOrigin(event) {
    return String(event.headers?.origin || event.headers?.Origin || "").trim();
}

function corsHeaders(origin = "") {
    const headers = {
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": "no-store",
        Vary: "Origin",
    };
    if (ALLOWED_ORIGINS.has(origin)) headers["Access-Control-Allow-Origin"] = origin;
    return headers;
}

function response(statusCode, body, origin) {
    return { statusCode, headers: corsHeaders(origin), body: JSON.stringify(body) };
}

export async function handler(event) {
    const origin = requestOrigin(event);
    const method = String(event.httpMethod || "GET").toUpperCase();

    if (method === "OPTIONS") return { statusCode: 204, headers: corsHeaders(origin), body: "" };
    if (origin && !ALLOWED_ORIGINS.has(origin)) return response(403, { error: "Origin not allowed." }, origin);
    if (method !== "POST") return response(405, { error: "Method not allowed." }, origin);
    if (!ai) return response(500, { error: "The server is missing GEMINI_API_KEY." }, origin);

    let body;
    try {
        body = JSON.parse(event.body || "{}");
    } catch {
        return response(400, { error: "Invalid JSON request." }, origin);
    }

    const audioBase64 = typeof body.audioBase64 === "string" ? body.audioBase64.trim() : "";
    const mimeType = String(body.mimeType || "audio/webm").split(";")[0].toLowerCase();
    if (!audioBase64) return response(400, { error: "No audio was received." }, origin);
    if (audioBase64.length > 5_500_000) return response(413, { error: "The recording is too large." }, origin);
    if (!ALLOWED_AUDIO_TYPES.has(mimeType)) return response(415, { error: "Unsupported audio format." }, origin);

    try {
        const result = await ai.models.generateContent({
            model,
            contents: [
                {
                    text: "Transcribe this spoken question accurately. Return only the words that were spoken, with no label, commentary, quotation marks, or markdown. Preserve the speaker's language.",
                },
                { inlineData: { data: audioBase64, mimeType } },
            ],
            config: {
                temperature: 0,
                maxOutputTokens: 300,
            },
        });

        const transcript = String(result.text || "")
            .trim()
            .replace(/^(["'`])([\s\S]*)\1$/, "$2")
            .trim();
        if (!transcript) return response(422, { error: "No speech was detected." }, origin);
        return response(200, { transcript }, origin);
    } catch (error) {
        console.error("Audio transcription error:", error);
        return response(502, { error: "Voice transcription is temporarily unavailable." }, origin);
    }
}
