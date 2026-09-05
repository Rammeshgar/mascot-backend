const ALLOWED_ORIGINS = new Set([
    "https://rammeshgar.github.io",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:4877",
    "http://127.0.0.1:4877",
    "http://localhost:8774",
    "http://127.0.0.1:8774",
]);

const openaiApiKey = process.env.OPENAI_API_KEY || "";
const TRANSCRIPTION_MODELS = ["gpt-4o-mini-transcribe", "whisper-1"];
const ALLOWED_AUDIO_TYPES = new Set(["audio/webm", "audio/ogg", "audio/mp4", "audio/mpeg", "audio/wav", "audio/x-wav"]);
const EXTENSIONS = { "audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "m4a", "audio/mpeg": "mp3", "audio/wav": "wav", "audio/x-wav": "wav" };

function requestOrigin(event) {
    return String(event.headers?.origin || event.headers?.Origin || "").trim();
}

function corsHeaders(origin = "") {
    const headers = {
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": "no-store",
        "Vary": "Origin",
    };
    if (ALLOWED_ORIGINS.has(origin)) headers["Access-Control-Allow-Origin"] = origin;
    return headers;
}

function jsonResponse(statusCode, body, origin = "") {
    return { statusCode, headers: corsHeaders(origin), body: JSON.stringify(body) };
}

async function requestTranscription(audio, mimeType, model) {
    const form = new FormData();
    form.append("model", model);
    form.append("response_format", "json");
    form.append("file", new Blob([audio], { type: mimeType }), `question.${EXTENSIONS[mimeType]}`);
    const response = await fetch("https://api.openai.com/v1/audio/transcriptions", {
        method: "POST",
        headers: { Authorization: `Bearer ${openaiApiKey}` },
        signal: AbortSignal.timeout(30000),
        body: form,
    });
    const data = await response.json().catch(() => ({}));
    return { response, data };
}

export const handler = async (event) => {
    const method = String(event.httpMethod || "GET").toUpperCase();
    const origin = requestOrigin(event);
    if (origin && !ALLOWED_ORIGINS.has(origin)) return jsonResponse(403, { error: "Origin not allowed." }, origin);
    if (method === "OPTIONS") return { statusCode: 204, headers: corsHeaders(origin), body: "" };
    if (method !== "POST") return jsonResponse(405, { error: "Method not allowed." }, origin);
    if (!openaiApiKey) return jsonResponse(500, { error: "The server is missing OPENAI_API_KEY." }, origin);

    let body;
    try { body = JSON.parse(event.body || "{}"); }
    catch { return jsonResponse(400, { error: "Invalid JSON request." }, origin); }

    const audioBase64 = typeof body.audioBase64 === "string" ? body.audioBase64.trim() : "";
    const mimeType = String(body.mimeType || "audio/webm").split(";")[0].toLowerCase();
    if (!audioBase64) return jsonResponse(400, { error: "No audio recording was provided." }, origin);
    if (!ALLOWED_AUDIO_TYPES.has(mimeType)) return jsonResponse(415, { error: "Unsupported audio format." }, origin);

    let audio;
    try { audio = Buffer.from(audioBase64, "base64"); }
    catch { return jsonResponse(400, { error: "The audio recording is invalid." }, origin); }
    if (!audio.length) return jsonResponse(400, { error: "The audio recording is empty." }, origin);
    if (audio.length > 4_500_000) return jsonResponse(413, { error: "The recording is too large. Keep it under 15 seconds." }, origin);

    try {
        let { response, data } = await requestTranscription(audio, mimeType, TRANSCRIPTION_MODELS[0]);

        if (!response.ok && [400, 403, 404, 500, 502, 503].includes(response.status)) {
            console.warn(
                "Primary transcription model unavailable; retrying with fallback:",
                response.status,
                data?.error?.code || data?.error?.type || "unknown"
            );
            ({ response, data } = await requestTranscription(audio, mimeType, TRANSCRIPTION_MODELS[1]));
        }

        if (!response.ok) {
            console.error(
                "OpenAI transcription error:",
                response.status,
                data?.error?.code || data?.error?.type || "unknown"
            );
            if (response.status === 429) return jsonResponse(429, { error: "Voice transcription is busy. Please retry shortly." }, origin);
            return jsonResponse(502, { error: "Voice transcription is temporarily unavailable." }, origin);
        }
        const transcript = String(data.text || "").trim();
        if (!transcript) return jsonResponse(422, { error: "No speech was detected." }, origin);
        return jsonResponse(200, { transcript }, origin);
    } catch (error) {
        console.error("Transcription request failed:", error?.name || "Error");
        return jsonResponse(500, { error: "Voice transcription could not be reached." }, origin);
    }
};
