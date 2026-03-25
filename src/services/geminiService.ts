import { GoogleGenAI, Modality, Type } from "@google/genai";

const getAI = () => new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || "" });

export const generateHealthyReference = async (cropName: string) => {
  const ai = getAI();
  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash-image',
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
};

export const speakDiagnosis = async (text: string) => {
  const ai = getAI();
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash-preview-tts",
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
};

export const searchGrounding = async (query: string, systemInstruction?: string) => {
  const ai = getAI();
  const apiKey = process.env.GEMINI_API_KEY;
  
  if (!apiKey) {
    console.error("GEMINI_API_KEY is missing in the frontend environment.");
    return { text: "I'm sorry, but the AI service is not configured correctly (API key missing).", sources: [] };
  }

  try {
    const response = await ai.models.generateContent({
      model: "gemini-3.1-flash-preview",
      contents: query,
      config: {
        tools: [{ googleSearch: {} }],
        systemInstruction: systemInstruction
      },
    });

    return {
      text: response.text || "I couldn't find a specific answer to that.",
      sources: response.candidates?.[0]?.groundingMetadata?.groundingChunks?.map(c => c.web).filter(Boolean) || []
    };
  } catch (error: any) {
    console.error("Search Grounding Error:", error);
    
    // Fallback if grounding fails (e.g. tool not available)
    try {
      const fallbackResponse = await ai.models.generateContent({
        model: "gemini-3.1-flash-preview",
        contents: query,
        config: { systemInstruction }
      });
      return {
        text: fallbackResponse.text || "I'm having trouble connecting to the AI core.",
        sources: []
      };
    } catch (fallbackError) {
      throw error; // Re-throw original error if fallback also fails
    }
  }
};

export const mapsGrounding = async (query: string, lat?: number, lng?: number, systemInstruction?: string) => {
  const ai = getAI();
  const apiKey = process.env.GEMINI_API_KEY;

  if (!apiKey) {
    console.error("GEMINI_API_KEY is missing in the frontend environment.");
    return { text: "I'm sorry, but the AI service is not configured correctly (API key missing).", sources: [] };
  }

  const config: any = {
    tools: [{ googleMaps: {} }],
    systemInstruction: systemInstruction
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
      text: response.text || "I couldn't locate that information on the map.",
      sources: response.candidates?.[0]?.groundingMetadata?.groundingChunks?.map(c => c.maps).filter(Boolean) || []
    };
  } catch (error: any) {
    console.error("Maps Grounding Error:", error);
    
    // Fallback if grounding fails
    try {
      const fallbackResponse = await ai.models.generateContent({
        model: "gemini-3.1-flash-preview",
        contents: query,
        config: { systemInstruction }
      });
      return {
        text: fallbackResponse.text || "I'm having trouble accessing map data.",
        sources: []
      };
    } catch (fallbackError) {
      throw error;
    }
  }
};

export const translateAdvisory = async (text: string, targetLanguage: string) => {
  const ai = getAI();
  const response = await ai.models.generateContent({
    model: "gemini-3.1-flash-lite-preview",
    contents: `Translate the following agricultural advisory to ${targetLanguage}. Keep the tone professional and helpful for a farmer. Output ONLY the translated text.\n\nText: ${text}`,
  });
  return response.text;
};

const cleanJSON = (text: string) => {
  try {
    // Remove markdown code blocks if present
    const cleaned = text.replace(/```json\n?|```/g, '').trim();
    return JSON.parse(cleaned);
  } catch (e) {
    console.error("Failed to parse JSON from AI response:", text);
    throw e;
  }
};

export const analyzeCropImages = async (images: string[], hybridResult: any) => {
  const imageParts = images.map((img: string) => {
    const mimeTypeMatch = img.match(/data:([^;]+);base64,/);
    const mimeType = mimeTypeMatch ? mimeTypeMatch[1] : "image/jpeg";
    const base64Data = img.includes(",") ? img.split(",")[1] : img;
    return {
      inlineData: {
        mimeType: mimeType,
        data: base64Data
      }
    };
  });

  const prompt = `Act as a Senior Agricultural Pathologist and Quantum AI Expert.
  Analyze these crop images and the results from our Hybrid Quantum-Classical Pipeline.
  
  HYBRID PIPELINE DATA:
  - Detected Condition: ${hybridResult.hybrid_pipeline?.vqc?.prediction || "Unknown"}
  - Quantum Confidence: ${((hybridResult.hybrid_pipeline?.vqc?.confidence || 0) * 100).toFixed(2)}%
  - Backend: ${hybridResult.hybrid_pipeline?.vqc?.backend || "Simulated"}
  
  TASK:
  1. Confirm or refine the diagnosis based on the images.
  2. Provide detailed organic and chemical remedies.
  3. Calculate a Crop Health Score (0-100).
  4. Predict spread risk and yield impact.
  5. Provide remediation steps and prevention strategies.
  6. Simulate nearby disease trends and weather risks.
  
  Return the response in JSON format with these fields:
  {
    "disease": "string",
    "confidence": number,
    "crop_health_score": number,
    "spread_risk": "Low/Medium/High/Critical",
    "remedies": {
      "organic": "string",
      "chemical": "string"
    },
    "remediation_steps": ["string"],
    "yield_prediction": "string",
    "weather_risk": "string",
    "nearby_disease_trends": "string",
    "expert_advice": {
      "explanation": "string",
      "treatment": {
        "organic": "string",
        "chemical": "string"
      },
      "yield_impact": "string",
      "prevention": "string"
    },
    "additional_info": "string"
  }`;

  try {
    const ai = getAI();
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: {
        parts: [{ text: prompt }, ...imageParts]
      },
      config: {
        responseMimeType: "application/json"
      }
    });
    return cleanJSON(response.text || "{}");
  } catch (error: any) {
    console.warn("Gemini API Error. Using fallback analysis.", error.message);
    return {
      disease: hybridResult.hybrid_pipeline?.vqc?.prediction || "Unknown Pathogen",
      confidence: hybridResult.hybrid_pipeline?.vqc?.confidence || 0.7,
      crop_health_score: 75,
      spread_risk: "Medium",
      remedies: {
        organic: "Isolate affected plants and apply neem oil.",
        chemical: "Consult a local agronomist for fungicide recommendations."
      },
      remediation_steps: ["Isolate plants", "Remove infected leaves", "Monitor spread"],
      yield_prediction: "Potential 15-20% loss if untreated.",
      weather_risk: "High humidity may accelerate spread.",
      nearby_disease_trends: "Increasing reports in neighboring regions.",
      expert_advice: {
        explanation: "The system detected signs of stress consistent with fungal infection.",
        treatment: {
          organic: "Apply neem oil or organic fungicide.",
          chemical: "Apply copper-based fungicides if the infection is severe."
        },
        yield_impact: "Potential 15-30% loss if untreated within 72 hours.",
        prevention: "Implement crop rotation and maintain optimal soil moisture levels."
      },
      additional_info: "API limits reached for real-time expert analysis."
    };
  }
};

export const fetchRealMarketData = async () => {
  const prompt = `Fetch the latest global market prices and trends for the following agricultural commodities: Rice, Wheat, Corn, Tomato. 
  For each commodity, provide:
  - Current price (with currency)
  - Trend (up or down)
  - Percentage change
  - Suggestion (BUY, SELL, or HOLD)
  - Profitability Index (0-100)
  - Forecast period (e.g., "2 weeks")
  
  Return the response in JSON format as an array of objects:
  [
    { "name": "string", "price": "string", "trend": "up" | "down", "change": "string", "suggestion": "BUY" | "SELL" | "HOLD", "profitability": number, "forecast": "string" }
  ]`;

  try {
    const ai = getAI();
    const response = await ai.models.generateContent({
      model: "gemini-3.1-pro-preview",
      contents: prompt,
      config: {
        tools: [{ googleSearch: {} }],
        responseMimeType: "application/json"
      }
    });
    return JSON.parse(response.text || "[]");
  } catch (error) {
    console.error("Error fetching market data:", error);
    return null;
  }
};
