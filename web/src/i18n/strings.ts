import type { Language, QuestionId } from "@/api/types";

/**
 * UI strings for the citizen report screen. Question wording and the s1-s4 status lines are copied from the
 * repo (backend/dialogue/lexicon.py, frontend/report.html), where hi/kn were reviewed by native speakers.
 * The other hi/kn strings below were written for this UI. [TODO-VERIFY] native-speaker review before the viva.
 */
export const LANGS: { code: Language; label: string; font: string }[] = [
  { code: "en", label: "EN", font: "font-sans" }, { code: "hi", label: "हिन्दी", font: "font-deva" }, { code: "kn", label: "ಕನ್ನಡ", font: "font-kann" },
];

export const QUESTIONS: Record<QuestionId, Record<Language, string>> = {
  Q_CONSCIOUS: { en: "Is the person awake and responding to you?", hi: "क्या व्यक्ति होश में है और जवाब दे रहा है?", kn: "ವ್ಯಕ್ತಿಗೆ ಪ್ರಜ್ಞೆ ಇದೆಯಾ ಮತ್ತು ನೀವು ಮಾತಾಡಿಸಿದಾಗ ಪ್ರತಿಕ್ರಿಯೆ ಕೊಡ್ತಿದ್ದಾರಾ?" },
  Q_BREATHING: { en: "Is the person breathing normally?", hi: "क्या व्यक्ति सामान्य तरीके से साँस ले रहा है?", kn: "ವ್ಯಕ್ತಿ ಸಾಮಾನ್ಯವಾಗಿ ಉಸಿರಾಡ್ತಿದ್ದಾರಾ?" },
  Q_BLEEDING: { en: "Is there visible bleeding? If yes, is it heavy?", hi: "क्या खून बह रहा है? अगर हाँ, तो क्या बहुत ज़्यादा खून बह रहा है?", kn: "ರಕ್ತ ಬರ್ತಿದೆಯಾ? ಬಂದರೆ, ತುಂಬಾ ರಕ್ತ ಬರ್ತಿದೆಯಾ?" },
  Q_SPEAK: { en: "Can the person speak clearly?", hi: "क्या वो ठीक से बोल पा रहे हैं?", kn: "ವ್ಯಕ್ತಿ ಸ್ಪಷ್ಟವಾಗಿ ಮಾತಾಡ್ತಿದ್ದಾರಾ?" },
  Q_NUM_INJURED: { en: "How many people are injured?", hi: "कितने लोग घायल हुए हैं?", kn: "ಎಷ್ಟು ಜನರಿಗೆ ಗಾಯವಾಗಿದೆ?" },
  Q_MECHANISM: { en: "Was the person hit by a fast vehicle, or did they fall from a height?", hi: "क्या व्यक्ति को किसी तेज़ गाड़ी ने टक्कर मारी है, या वो ऊँचाई से गिरे हैं?", kn: "ವ್ಯಕ್ತಿಗೆ ವೇಗವಾಗಿ ಬಂದ ಗಾಡಿ ಡಿಕ್ಕಿ ಹೊಡೀತಾ, ಅಥವಾ ಎತ್ತರದಿಂದ ಬಿದ್ರಾ?" },
};
export const QUESTION_ORDER: QuestionId[] = ["Q_CONSCIOUS", "Q_BREATHING", "Q_BLEEDING", "Q_SPEAK", "Q_NUM_INJURED", "Q_MECHANISM"];
export const QKEY: Record<QuestionId, string> = { Q_CONSCIOUS: "q1", Q_BREATHING: "q2", Q_BLEEDING: "q3", Q_SPEAK: "q4", Q_NUM_INJURED: "q5", Q_MECHANISM: "q6" };

export const STATUS: Record<Language, Record<"s1" | "s2" | "s3" | "s4", string>> = {
  en: { s1: "Help is on the way. Stay with the person if it is safe.", s2: "An ambulance has been sent.", s3: "Please answer a few quick questions.", s4: "Follow these first-aid steps until help arrives." },
  hi: { s1: "मदद आ रही है। अगर सुरक्षित है, तो उस व्यक्ति के पास ही रहें।", s2: "एम्बुलेंस भेज दी गई है।", s3: "बस कुछ छोटे-छोटे सवालों के जवाब दे दीजिए।", s4: "मदद पहुँचने तक ये first-aid के तरीके अपनाइए।" },
  kn: { s1: "ಸಹಾಯ ಬರುತ್ತಿದೆ. ನಿಮಗೆ ಸುರಕ್ಷಿತವಾಗಿದ್ದರೆ, ಆ ವ್ಯಕ್ತಿಯ ಜೊತೆಯಲ್ಲೇ ಇರಿ.", s2: "ಆಂಬ್ಯುಲೆನ್ಸ್ ಕಳಿಸಲಾಗಿದೆ.", s3: "ದಯವಿಟ್ಟು ಕೆಲವು ಚಿಕ್ಕ ಪ್ರಶ್ನೆಗಳಿಗೆ ಉತ್ತರ ಕೊಡಿ.", s4: "ಸಹಾಯ ಬರುವವರೆಗೂ ಈ first-aid ಕ್ರಮಗಳನ್ನು ಅನುಸರಿಸಿ." },
};

type K = "title" | "sub" | "tap" | "listening" | "stop" | "type" | "placeholder" | "send" | "processing" | "locating" | "gps" | "gpsApprox" | "gpsFail" | "demoLoc"
  | "qOf" | "yes" | "no" | "notSure" | "playQ" | "source" | "hideSource" | "notCovered" | "noMeds" | "wait" | "again" | "number" | "micDenied" | "steps" | "verbatim" | "reported" | "listen";
export const UI: Record<Language, Record<K, string>> = {
  en: { title: "Report an emergency", sub: "Describe what happened. Help is coordinated as you answer.", tap: "Tap and speak", listening: "Listening…", stop: "Tap to finish", type: "Type instead", placeholder: "What happened? Where is the person?", send: "Send report", processing: "Understanding your report…", locating: "Finding your location…", gps: "Device GPS", gpsApprox: "Approximate location", gpsFail: "Location unavailable", demoLoc: "Use demo location", qOf: "Question {n} of 6", yes: "Yes", no: "No", notSure: "Not sure", playQ: "Play question", source: "Source", hideSource: "Hide source", notCovered: "We can’t give specific steps for this situation.", noMeds: "We can’t advise on medicines or doses.", wait: "Keep the person safe and wait for the professionals.", again: "New report", number: "Number of people", micDenied: "Microphone unavailable. Please type your report.", steps: "First-aid steps", verbatim: "Quoted from the guideline", reported: "You reported", listen: "Listen again" },
  hi: { title: "आपातकाल की रिपोर्ट करें", sub: "बताइए क्या हुआ। आपके जवाब देते ही मदद का इंतज़ाम होता है।", tap: "दबाकर बोलिए", listening: "सुन रहे हैं…", stop: "खत्म करने के लिए दबाएँ", type: "लिखकर बताएँ", placeholder: "क्या हुआ? व्यक्ति कहाँ है?", send: "रिपोर्ट भेजें", processing: "आपकी रिपोर्ट समझ रहे हैं…", locating: "आपकी लोकेशन ढूँढ रहे हैं…", gps: "डिवाइस GPS", gpsApprox: "अनुमानित लोकेशन", gpsFail: "लोकेशन नहीं मिली", demoLoc: "डेमो लोकेशन इस्तेमाल करें", qOf: "सवाल {n} / 6", yes: "हाँ", no: "नहीं", notSure: "पता नहीं", playQ: "सवाल सुनें", source: "स्रोत", hideSource: "स्रोत छिपाएँ", notCovered: "इस स्थिति के लिए हम खास कदम नहीं बता सकते।", noMeds: "हम दवा या खुराक की सलाह नहीं दे सकते।", wait: "व्यक्ति को सुरक्षित रखें और पेशेवरों का इंतज़ार करें।", again: "नई रिपोर्ट", number: "लोगों की संख्या", micDenied: "माइक्रोफ़ोन उपलब्ध नहीं है। कृपया रिपोर्ट लिखें।", steps: "प्राथमिक उपचार के कदम", verbatim: "दिशानिर्देश से उद्धृत", reported: "आपने बताया", listen: "फिर से सुनें" },
  kn: { title: "ತುರ್ತು ಪರಿಸ್ಥಿತಿ ವರದಿ ಮಾಡಿ", sub: "ಏನಾಯಿತು ಎಂದು ಹೇಳಿ. ನೀವು ಉತ್ತರಿಸುತ್ತಿದ್ದಂತೆ ಸಹಾಯ ಏರ್ಪಡುತ್ತದೆ.", tap: "ಒತ್ತಿ ಮಾತಾಡಿ", listening: "ಕೇಳುತ್ತಿದ್ದೇವೆ…", stop: "ಮುಗಿಸಲು ಒತ್ತಿ", type: "ಬರೆದು ತಿಳಿಸಿ", placeholder: "ಏನಾಯಿತು? ವ್ಯಕ್ತಿ ಎಲ್ಲಿದ್ದಾರೆ?", send: "ವರದಿ ಕಳಿಸಿ", processing: "ನಿಮ್ಮ ವರದಿಯನ್ನು ಅರ್ಥಮಾಡಿಕೊಳ್ಳುತ್ತಿದ್ದೇವೆ…", locating: "ನಿಮ್ಮ ಸ್ಥಳ ಹುಡುಕುತ್ತಿದ್ದೇವೆ…", gps: "ಸಾಧನದ GPS", gpsApprox: "ಅಂದಾಜು ಸ್ಥಳ", gpsFail: "ಸ್ಥಳ ಸಿಗಲಿಲ್ಲ", demoLoc: "ಡೆಮೊ ಸ್ಥಳ ಬಳಸಿ", qOf: "ಪ್ರಶ್ನೆ {n} / 6", yes: "ಹೌದು", no: "ಇಲ್ಲ", notSure: "ಗೊತ್ತಿಲ್ಲ", playQ: "ಪ್ರಶ್ನೆ ಕೇಳಿ", source: "ಮೂಲ", hideSource: "ಮೂಲ ಮರೆಮಾಡಿ", notCovered: "ಈ ಸ್ಥಿತಿಗೆ ನಿರ್ದಿಷ್ಟ ಹಂತಗಳನ್ನು ಹೇಳಲು ಸಾಧ್ಯವಿಲ್ಲ.", noMeds: "ಔಷಧ ಅಥವಾ ಡೋಸ್ ಬಗ್ಗೆ ಸಲಹೆ ನೀಡಲು ಸಾಧ್ಯವಿಲ್ಲ.", wait: "ವ್ಯಕ್ತಿಯನ್ನು ಸುರಕ್ಷಿತವಾಗಿಡಿ ಮತ್ತು ವೃತ್ತಿಪರರಿಗಾಗಿ ಕಾಯಿರಿ.", again: "ಹೊಸ ವರದಿ", number: "ಜನರ ಸಂಖ್ಯೆ", micDenied: "ಮೈಕ್ರೊಫೋನ್ ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ವರದಿಯನ್ನು ಬರೆಯಿರಿ.", steps: "ಪ್ರಥಮ ಚಿಕಿತ್ಸೆಯ ಹಂತಗಳು", verbatim: "ಮಾರ್ಗಸೂಚಿಯಿಂದ ಉಲ್ಲೇಖ", reported: "ನೀವು ತಿಳಿಸಿದ್ದು", listen: "ಮತ್ತೆ ಕೇಳಿ" },
};
export const ANSWER_TOKEN: Record<Language, { yes: string; no: string; unknown: string }> = {
  en: { yes: "yes", no: "no", unknown: "not sure" }, hi: { yes: "हां", no: "नहीं", unknown: "पता नहीं" }, kn: { yes: "ಹೌದು", no: "ಇಲ್ಲ", unknown: "ಗೊತ್ತಿಲ್ಲ" },
};
