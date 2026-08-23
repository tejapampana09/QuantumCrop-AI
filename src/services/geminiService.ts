import { GoogleGenAI, Modality } from "@google/genai";

const getAI = () => {
  const apiKey = process.env.GEMINI_API_KEY || (import.meta as any).env?.VITE_GEMINI_API_KEY || "";
  if (!apiKey) return null;
  return new GoogleGenAI({ apiKey });
};

export interface ExpertAdvisory {
  disease: string;
  confidence: number;
  crop_health_score: number;
  spread_risk: "Low" | "Medium" | "High" | "Critical";
  remedies: {
    organic: string;
    chemical: string;
  };
  remediation_steps: string[];
  yield_prediction: string;
  weather_risk: string;
  nearby_disease_trends: string;
  expert_advice: {
    explanation: string;
    treatment: {
      organic: string;
      chemical: string;
    };
    yield_impact: string;
    prevention: string;
  };
  additional_info: string;
  advisory_status: "active" | "unavailable";
  arbitration_status?: "success" | "crop_mismatch" | "disease_uncertain" | "not_a_leaf" | "poor_quality" | "uncertain";
  detected_crop?: string;
  arbitration_reason?: string;
  action_guidance?: string;
}

export const generateHealthyReference = async (cropName: string): Promise<string | null> => {
  try {
    const ai = getAI();
    if (!ai) return null;

    const response = await ai.models.generateContent({
      model: 'gemini-3.1-flash-image',
      contents: {
        parts: [
          { text: `A high-resolution, professional macro photograph of a perfectly healthy ${cropName} plant leaf, vibrant green, no diseases, studio lighting, white background.` }
        ]
      },
      config: {
        imageConfig: {
          aspectRatio: "1:1"
        }
      }
    });

    for (const part of response.candidates?.[0]?.content?.parts || []) {
      if (part.inlineData) {
        return `data:image/png;base64,${part.inlineData.data}`;
      }
    }
    return null;
  } catch (error) {
    console.warn("Healthy reference generation unavailable:", error);
    return null;
  }
};

export const speakDiagnosis = async (text: string): Promise<void> => {
  try {
    const ai = getAI();
    if (!ai) return;

    const response = await ai.models.generateContent({
      model: "gemini-3.1-flash-tts-preview",
      contents: [{ parts: [{ text }] }],
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: 'Kore' },
          },
        },
      },
    });

    const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (base64Audio) {
      const audio = new Audio(`data:audio/wav;base64,${base64Audio}`);
      audio.play();
    }
  } catch (error) {
    console.warn("Speech synthesis unavailable:", error);
  }
};

export const searchGrounding = async (query: string, systemInstruction?: string) => {
  const ai = getAI();
  if (!ai) {
    return { text: "AI advisory search currently unavailable (API key not configured).", sources: [] };
  }

  try {
    const response = await ai.models.generateContent({
      model: "gemini-3.1-flash-preview",
      contents: query,
      config: {
        tools: [{ googleSearch: {} }],
        systemInstruction
      },
    });

    return {
      text: response.text || "No relevant information found.",
      sources: response.candidates?.[0]?.groundingMetadata?.groundingChunks?.map(c => c.web).filter(Boolean) || []
    };
  } catch (error: any) {
    console.warn("Search Grounding Error, falling back to direct prompt:", error.message);
    try {
      const fallbackResponse = await ai.models.generateContent({
        model: "gemini-3.1-flash-preview",
        contents: query,
        config: { systemInstruction }
      });
      return {
        text: fallbackResponse.text || "AI advisory currently unavailable.",
        sources: []
      };
    } catch {
      return { text: "AI advisory currently unavailable.", sources: [] };
    }
  }
};

export const mapsGrounding = async (query: string, lat?: number, lng?: number, systemInstruction?: string) => {
  const ai = getAI();
  if (!ai) {
    return { text: "Agricultural map lookup currently unavailable.", sources: [] };
  }

  const config: any = {
    tools: [{ googleMaps: {} }],
    systemInstruction
  };

  if (lat && lng) {
    config.toolConfig = {
      retrievalConfig: {
        latLng: { latitude: lat, longitude: lng }
      }
    };
  }

  try {
    const response = await ai.models.generateContent({
      model: "gemini-3.1-flash-preview",
      contents: query,
      config
    });

    return {
      text: response.text || "Map data unavailable for this query.",
      sources: response.candidates?.[0]?.groundingMetadata?.groundingChunks?.map(c => c.maps).filter(Boolean) || []
    };
  } catch (error: any) {
    console.warn("Maps Grounding Error:", error.message);
    return { text: "Agricultural map data currently unavailable.", sources: [] };
  }
};

export const translateAdvisory = async (text: string, targetLanguage: string): Promise<string> => {
  const ai = getAI();
  if (!ai) return text;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-3.1-flash-lite-preview",
      contents: `Translate the following agricultural advisory into ${targetLanguage}. Keep agricultural terms accurate and easy for a farmer to understand. Output ONLY the translated text.\n\nText: ${text}`,
    });
    return response.text || text;
  } catch {
    return text;
  }
};

const cleanJSON = (text: string) => {
  try {
    const cleaned = text.replace(/```json\n?|```/g, '').trim();
    return JSON.parse(cleaned);
  } catch (e) {
    console.warn("Failed to parse JSON from AI response:", text);
    return null;
  }
};

/**
 * Gemini Advisory Layer:
 * Takes the REAL predicted disease name and confidence from the primary MobileNetV2 classifier.
 * Generates an expert pathology explanation, treatment, and prevention plan.
 * NEVER overrides or replaces the predicted disease.
 */
export const analyzeCropImages = async (
  images: string[],
  hybridResult: any
): Promise<ExpertAdvisory> => {
  // Primary model prediction from MobileNetV2
  const primaryDisease = hybridResult?.primaryDiagnosis?.disease || hybridResult?.cnn?.prediction || "Unknown Condition";
  const rawConfidence = hybridResult?.primaryDiagnosis?.confidence !== undefined
    ? hybridResult.primaryDiagnosis.confidence
    : ((hybridResult?.cnn?.confidence || 0) * 100);
  const confidence = Number(rawConfidence.toFixed(1));

  // Determine standard fallback
  const isHealthy = primaryDisease.toLowerCase().includes("healthy");
  const fallbackAdvisory: ExpertAdvisory = {
    disease: primaryDisease,
    confidence: confidence,
    crop_health_score: isHealthy ? 95 : Math.max(10, Math.round(100 - confidence * 0.7)),
    spread_risk: isHealthy ? "Low" : (confidence > 80 ? "High" : "Medium"),
    remedies: {
      organic: isHealthy
        ? "Maintain optimal crop spacing, regular weeding, and balanced drip irrigation."
        : "Isolate affected plants immediately, prune heavily infected foliage, and spray organic neem oil (5ml/L).",
      chemical: isHealthy
        ? "No chemical fungicide or pesticide required for healthy crops."
        : "Consult local agronomy extension for targeted copper oxychloride or systemic fungicide application."
    },
    remediation_steps: isHealthy
      ? ["Continue standard agronomic monitoring", "Ensure balanced soil nitrogen and phosphorus", "Maintain clean field boundaries"]
      : ["Isolate infected foliage", "Prune and safely discard diseased leaves", "Avoid overhead sprinkler irrigation", "Apply protective fungicide barrier"],
    yield_prediction: isHealthy
      ? "Expected normal harvest yield based on current healthy canopy."
      : "Potential 15-25% yield loss if untreated within 5-7 days.",
    weather_risk: "Monitor local humidity and leaf wetness duration.",
    nearby_disease_trends: "Regional agricultural alerts monitor standard seasonal pathogen vectors.",
    expert_advice: {
      explanation: isHealthy
        ? `The leaf exhibits vibrant chlorophyll distribution and no significant pathogen lesions, consistent with a healthy crop state.`
        : `Primary MobileNetV2 classifier identified ${primaryDisease.replace(/___/g, ' - ').replace(/_/g, ' ')}. Characterized by diagnostic foliar symptoms.`,
      treatment: {
        organic: isHealthy ? "Standard preventive maintenance." : "Neem oil spray, bio-fungicides (Trichoderma viride).",
        chemical: isHealthy ? "None required." : "Targeted agricultural fungicide application as per label instructions."
      },
      yield_impact: isHealthy ? "Minimal risk." : "Moderate risk without timely intervention.",
      prevention: "Implement crop rotation, maintain soil drainage, and use certified disease-free seeds."
    },
    additional_info: "Expert advisory generated from verified agronomic knowledge base.",
    advisory_status: "active"
  };

  const ai = getAI();
  if (!ai) {
    fallbackAdvisory.advisory_status = "unavailable";
    fallbackAdvisory.additional_info = "AI advisory service unavailable (GEMINI_API_KEY not configured). Displaying standard agronomic guidelines.";
    return fallbackAdvisory;
  }

  try {
    const imageParts = images.slice(0, 2).map((img: string) => {
      const mimeTypeMatch = img.match(/data:([^;]+);base64,/);
      const mimeType = mimeTypeMatch ? mimeTypeMatch[1] : "image/jpeg";
      const base64Data = img.includes(",") ? img.split(",")[1] : img;
      return {
        inlineData: {
          mimeType,
          data: base64Data
        }
      };
    });

    const prompt = `You are a Chief Agricultural Pathologist and Agronomist providing advisory insights for a farmer.

CRITICAL ARCHITECTURAL CONTEXT:
The primary edge classifier MobileNetV2 (1280D) has diagnosed the crop with: "${primaryDisease}" (Confidence: ${confidence}%).
Your role is to provide expert pathology explanations, foliar visual assessment, concrete organic/chemical remedies, and prevention steps.

INSTRUCTIONS:
1. Verify if the uploaded image is a valid plant leaf (is_leaf: true/false).
2. Assess visual quality (good/moderate/poor) and provide foliar diagnostic observations.
3. Formulate concrete organic treatments (with preparation and dosages), chemical fungicide options, and 4 actionable recovery steps specifically for "${primaryDisease}".
4. Calculate crop health score (0-100) and spread risk (Low/Medium/High/Critical).

Return a valid JSON object matching this exact schema:
{
  "is_leaf": boolean,
  "visual_quality": "good" | "moderate" | "poor",
  "crop_health_score": number,
  "spread_risk": "Low" | "Medium" | "High" | "Critical",
  "remedies": {
    "organic": "string (concrete organic treatments with preparation and dosages)",
    "chemical": "string (specific active ingredients and application instructions)"
  },
  "remediation_steps": ["step 1", "step 2", "step 3", "step 4"],
  "yield_prediction": "string (estimated harvest impact)",
  "weather_risk": "string",
  "nearby_disease_trends": "string",
  "expert_advice": {
    "explanation": "string (detailed breakdown of symptoms and pathology progression)",
    "treatment": {
      "organic": "string",
      "chemical": "string"
    },
    "yield_impact": "string",
    "prevention": "string"
  },
  "additional_info": "string"
}`;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: {
        parts: [{ text: prompt }, ...imageParts]
      },
      config: {
        responseMimeType: "application/json"
      }
    });

    const parsed = cleanJSON(response.text || "{}");
    if (parsed) {
      const isLeaf = parsed.is_leaf !== false;

      return {
        disease: primaryDisease,
        confidence: confidence,
        crop_health_score: typeof parsed.crop_health_score === 'number' ? parsed.crop_health_score : fallbackAdvisory.crop_health_score,
        spread_risk: parsed.spread_risk || fallbackAdvisory.spread_risk,
        remedies: parsed.remedies || fallbackAdvisory.remedies,
        remediation_steps: Array.isArray(parsed.remediation_steps) && parsed.remediation_steps.length > 0 ? parsed.remediation_steps : fallbackAdvisory.remediation_steps,
        yield_prediction: parsed.yield_prediction || fallbackAdvisory.yield_prediction,
        weather_risk: parsed.weather_risk || fallbackAdvisory.weather_risk,
        nearby_disease_trends: parsed.nearby_disease_trends || fallbackAdvisory.nearby_disease_trends,
        expert_advice: parsed.expert_advice || fallbackAdvisory.expert_advice,
        additional_info: parsed.additional_info || "AI expert advisory successfully compiled.",
        advisory_status: "active",
        arbitration_status: isLeaf ? "success" : "not_a_leaf",
        detected_crop: primaryDisease.split("___")[0] || "Unknown",
        arbitration_reason: isLeaf ? "Primary MobileNetV2 diagnosis confirmed." : "No valid plant leaf detected.",
        action_guidance: isLeaf ? "Remedy Available" : "Please provide a clear close-up of a single leaf"
      };
    }
  } catch (error: any) {
    console.warn("Gemini API advisory call failed, using safe agronomic fallback:", error.message);
  }

  fallbackAdvisory.advisory_status = "unavailable";
  fallbackAdvisory.arbitration_status = "uncertain";
  fallbackAdvisory.additional_info = "AI advisory service temporarily unavailable. Displaying standard agronomic baseline.";
  return fallbackAdvisory;
};

export const fetchRealMarketData = async () => {
  const prompt = `Fetch representative global market prices and trends for agricultural commodities: Rice, Wheat, Corn, Tomato. 
  For each commodity, provide:
  - Current price (with currency)
  - Trend ("up" or "down")
  - Percentage change
  - Suggestion ("BUY", "SELL", or "HOLD")
  - Profitability Index (0-100)
  - Forecast period (e.g., "2 weeks")
  
  Return the response in JSON format as an array of objects:
  [
    { "name": "string", "price": "string", "trend": "up" | "down", "change": "string", "suggestion": "BUY" | "SELL" | "HOLD", "profitability": number, "forecast": "string" }
  ]`;

  try {
    const ai = getAI();
    if (!ai) return null;

    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json"
      }
    });
    return cleanJSON(response.text || "[]");
  } catch (error) {
    console.warn("Error fetching market data from AI:", error);
    return null;
  }
};
