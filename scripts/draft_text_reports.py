"""Draft the text-severity report set (docs/08 §4): template grid -> data/text_reports/reports.csv.

    python -m scripts.draft_text_reports [--per-language 150] [--force]

Grid: injury type (5, as schemas.INJURY_TYPES) x severity (4) x core phrase x opener x suffix,
sampled with a fixed seed, balanced across the 20 (type, severity) cells. Hindi/Kannada get
native-script reports plus romanised (typing/ASR style) reports. Rows are written with
reviewed=0: they are DRAFTS. Labels follow the rubric in README.md; the human reviewer must
read every row, fix wording/label, then set reviewed=1. Hindi/Kannada wording is machine-drafted
[TODO-VERIFY native review]. No locations in the text (GPS only, AGENTS rule 5).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

OUT = Path("data/text_reports/reports.csv")
SEED = 42
TYPES = ("road_accident", "fall", "burn", "cardiac_medical", "other")

# NATIVE[lang][type] = [severity 0, 1, 2, 3], each a list of core-phrase variants.
EN = {
    "road_accident": [
        ["two bikes touched, nobody is hurt, just scratches on the bike", "car hit a divider, everyone is out and walking, no injury"],
        ["bike skid, rider has a sprained wrist and scraped knees, he is talking fine", "auto and car collided, passenger has pain in the shoulder but is stable"],
        ["bike hit a bus, rider's leg looks broken and bleeding a lot but he is awake", "car accident, driver is conscious but has a deep cut on the head bleeding heavily"],
        ["truck hit a bike, rider is lying on the road not moving not breathing", "car crash, driver unconscious, blood everywhere, not responding"]],
    "fall": [
        ["child fell from a bench, crying, no cut, playing again", "old man slipped, got up himself, says he is fine"],
        ["woman fell on stairs, ankle swollen and she cannot put weight, awake and talking", "worker slipped from a low stool, bruise on the back, pain but stable"],
        ["man fell from the first floor, awake but cannot move his legs, bleeding from the head", "fell from a ladder, bone sticking out of the arm, bleeding heavily"],
        ["worker fell from the scaffolding, unconscious, not breathing properly", "fell from the third floor, no response, blood from the ears"]],
    "burn": [
        ["small splash of hot tea on the hand, red but no blister, she is fine", "touched a hot pan, mild redness, no pain now"],
        ["cooking oil burn on the forearm, blisters, in pain but talking normally", "kid got hot water on the leg, blisters, crying but alert"],
        ["gas cylinder flash, burns on face and arms, he is awake but in severe pain and struggling to breathe", "fire in the kitchen, large burns on the legs, skin peeling, conscious"],
        ["house fire, person pulled out unconscious, burns over the body, not breathing", "electric shock and burns, collapsed, no response, no pulse"]],
    "cardiac_medical": [
        ["man felt dizzy for a minute, now sitting and drinking water, normal", "woman feeling weak from the heat, talking fine, no pain"],
        ["elderly man has chest discomfort but is sitting and talking, sweating a bit", "girl fainted briefly, woke up, pale but responding"],
        ["man has severe chest pain, sweating a lot, breathing hard, still conscious", "seizure just stopped, he is drowsy, confused and breathing fast"],
        ["man collapsed suddenly, unconscious, not breathing, no pulse", "woman fell down, lips blue, not responding, no breathing"]],
    "other": [
        ["I saw a crowd, not sure anything happened, nobody looks hurt", "there was a fight earlier but everyone left, no one injured"],
        ["boy hit by a cricket ball, nose bleeding a little, stopped now, he is alert", "dog bite on the leg, small wound, bleeding slightly, walking"],
        ["stabbed in the stomach, bleeding a lot, awake but weak", "man hit on the head with a rod, heavy bleeding, confused"],
        ["person found on the road, unconscious, not breathing, a lot of blood", "pulled out of water, unconscious, not breathing, blue face"]],
}
HI = {
    "road_accident": [
        ["दो बाइक आपस में छू गईं, किसी को चोट नहीं लगी, बस खरोंच है", "कार डिवाइडर से टकराई, सब बाहर निकलकर चल रहे हैं, कोई घायल नहीं"],
        ["बाइक फिसल गई, चालक की कलाई में मोच है और घुटने छिल गए, वो ठीक से बात कर रहा है", "ऑटो और कार की टक्कर हुई, सवारी के कंधे में दर्द है लेकिन होश में है"],
        ["बाइक बस से टकरा गई, चालक का पैर टूटा लगता है और बहुत खून बह रहा है, पर वो होश में है", "कार एक्सीडेंट हुआ, ड्राइवर होश में है पर सिर में गहरा कट है और खून बहुत बह रहा है"],
        ["ट्रक ने बाइक को टक्कर मारी, सवार सड़क पर पड़ा है, हिल नहीं रहा, सांस नहीं ले रहा", "कार क्रैश हो गई, ड्राइवर बेहोश है, हर तरफ खून है, कोई जवाब नहीं दे रहा"]],
    "fall": [
        ["बच्चा बेंच से गिर गया, रो रहा था, अब फिर खेल रहा है, कोई कट नहीं", "बुज़ुर्ग फिसल गए, खुद उठ गए, कह रहे हैं ठीक हूँ"],
        ["औरत सीढ़ियों से गिर गई, टखना सूज गया है, पैर नहीं रख पा रही, होश में है", "मज़दूर स्टूल से फिसला, पीठ में चोट और दर्द है, पर स्थिर है"],
        ["आदमी पहली मंज़िल से गिरा, होश में है पर पैर हिला नहीं पा रहा, सिर से खून बह रहा है", "सीढ़ी से गिरा, हाथ की हड्डी बाहर दिख रही है, बहुत खून बह रहा है"],
        ["मज़दूर मचान से गिर गया, बेहोश है, ठीक से सांस नहीं ले रहा", "तीसरी मंज़िल से गिरा, कोई जवाब नहीं, कान से खून आ रहा है"]],
    "burn": [
        ["हाथ पर गर्म चाय के छींटे पड़े, थोड़ी लाली है, छाला नहीं, वो ठीक है", "गर्म तवे को छू लिया, हल्की लाली है, अब दर्द नहीं"],
        ["बांह पर गर्म तेल से जल गया, छाले हैं, दर्द है पर सामान्य बात कर रहा है", "बच्चे के पैर पर गर्म पानी गिरा, छाले हैं, रो रहा है पर सतर्क है"],
        ["गैस सिलेंडर में आग लगी, चेहरे और बाजुओं पर जलन, होश में है पर बहुत दर्द में है और सांस लेने में तकलीफ है", "रसोई में आग लगी, पैरों पर बड़ी जलन, चमड़ी उतर रही है, होश में है"],
        ["घर में आग लगी, आदमी को बेहोशी में बाहर निकाला, पूरे शरीर पर जलन, सांस नहीं ले रहा", "करंट लगा और जल गया, गिर पड़ा, कोई जवाब नहीं, नब्ज़ नहीं मिल रही"]],
    "cardiac_medical": [
        ["आदमी को एक मिनट चक्कर आया, अब बैठकर पानी पी रहा है, सामान्य है", "औरत को गर्मी से कमज़ोरी लग रही है, ठीक से बोल रही है, दर्द नहीं"],
        ["बुज़ुर्ग को सीने में हल्की बेचैनी है, बैठे हैं और बात कर रहे हैं, थोड़ा पसीना आ रहा है", "लड़की थोड़ी देर के लिए बेहोश हुई, होश आ गया, पीली है पर जवाब दे रही है"],
        ["आदमी को सीने में तेज़ दर्द है, बहुत पसीना आ रहा है, सांस फूल रही है, होश में है", "दौरा अभी रुका है, वो सुस्त और उलझन में है, तेज़ सांस ले रहा है"],
        ["आदमी अचानक गिर पड़ा, बेहोश है, सांस नहीं ले रहा, नब्ज़ नहीं है", "औरत गिर गई, होंठ नीले हैं, कोई जवाब नहीं, सांस नहीं"]],
    "other": [
        ["भीड़ दिखी, पता नहीं कुछ हुआ या नहीं, कोई घायल नहीं दिख रहा", "पहले झगड़ा हुआ था पर सब चले गए, कोई घायल नहीं"],
        ["लड़के को क्रिकेट की गेंद लगी, नाक से थोड़ा खून आया, अब रुक गया, वो सतर्क है", "कुत्ते ने पैर पर काटा, छोटा घाव है, थोड़ा खून है, चल रहा है"],
        ["पेट में चाकू लगा है, बहुत खून बह रहा है, होश में है पर कमज़ोर है", "आदमी के सिर पर रॉड मारी, बहुत खून बह रहा है, उलझन में है"],
        ["सड़क पर एक आदमी मिला, बेहोश है, सांस नहीं ले रहा, बहुत खून है", "आदमी को पानी से निकाला, बेहोश है, सांस नहीं ले रहा, चेहरा नीला है"]],
}
KN = {
    "road_accident": [
        ["ಎರಡು ಬೈಕ್‌ಗಳು ಸ್ವಲ್ಪ ತಾಗಿದವು, ಯಾರಿಗೂ ಗಾಯವಾಗಿಲ್ಲ, ಸ್ವಲ್ಪ ತರಚಿದೆ ಅಷ್ಟೇ", "ಕಾರು ಡಿವೈಡರ್‌ಗೆ ಡಿಕ್ಕಿ ಹೊಡೆಯಿತು, ಎಲ್ಲರೂ ಹೊರಬಂದು ನಡೆಯುತ್ತಿದ್ದಾರೆ, ಗಾಯವಿಲ್ಲ"],
        ["ಬೈಕ್ ಜಾರಿ ಬಿತ್ತು, ಸವಾರನ ಮಣಿಕಟ್ಟಿಗೆ ಉಳುಕಾಗಿದೆ, ಮೊಣಕಾಲು ತರಚಿದೆ, ಚೆನ್ನಾಗಿ ಮಾತನಾಡುತ್ತಿದ್ದಾನೆ", "ಆಟೋ ಮತ್ತು ಕಾರು ಡಿಕ್ಕಿಯಾಯಿತು, ಪ್ರಯಾಣಿಕರ ಭುಜದಲ್ಲಿ ನೋವಿದೆ ಆದರೆ ಎಚ್ಚರದಲ್ಲಿದ್ದಾರೆ"],
        ["ಬೈಕ್ ಬಸ್‌ಗೆ ಡಿಕ್ಕಿ ಹೊಡೆಯಿತು, ಸವಾರನ ಕಾಲು ಮುರಿದಂತಿದೆ, ತುಂಬಾ ರಕ್ತ ಹೋಗುತ್ತಿದೆ, ಆದರೆ ಎಚ್ಚರವಿದ್ದಾನೆ", "ಕಾರು ಅಪಘಾತ, ಚಾಲಕ ಎಚ್ಚರವಿದ್ದಾನೆ ಆದರೆ ತಲೆಗೆ ಆಳವಾದ ಗಾಯ, ತುಂಬಾ ರಕ್ತ ಸುರಿಯುತ್ತಿದೆ"],
        ["ಲಾರಿ ಬೈಕ್‌ಗೆ ಡಿಕ್ಕಿ ಹೊಡೆಯಿತು, ಸವಾರ ರಸ್ತೆಯಲ್ಲಿ ಬಿದ್ದಿದ್ದಾನೆ, ಅಲುಗಾಡುತ್ತಿಲ್ಲ, ಉಸಿರಾಡುತ್ತಿಲ್ಲ", "ಕಾರು ಅಪಘಾತ, ಚಾಲಕ ಪ್ರಜ್ಞಾಹೀನ, ಎಲ್ಲೆಡೆ ರಕ್ತ, ಕರೆದರೂ ಉತ್ತರಿಸುತ್ತಿಲ್ಲ"]],
    "fall": [
        ["ಮಗು ಬೆಂಚಿನಿಂದ ಬಿತ್ತು, ಅತ್ತಿತು, ಈಗ ಮತ್ತೆ ಆಡುತ್ತಿದೆ, ಗಾಯವಿಲ್ಲ", "ವೃದ್ಧರು ಜಾರಿ ಬಿದ್ದರು, ತಾವೇ ಎದ್ದರು, ಚೆನ್ನಾಗಿದ್ದೇನೆ ಎನ್ನುತ್ತಿದ್ದಾರೆ"],
        ["ಮಹಿಳೆ ಮೆಟ್ಟಿಲಿನಿಂದ ಬಿದ್ದರು, ಕಾಲಿನ ಹಿಮ್ಮಡಿ ಊದಿಕೊಂಡಿದೆ, ಕಾಲೂರಲು ಆಗುತ್ತಿಲ್ಲ, ಎಚ್ಚರದಲ್ಲಿದ್ದಾರೆ", "ಕೆಲಸಗಾರ ಸ್ಟೂಲ್‌ನಿಂದ ಜಾರಿದ, ಬೆನ್ನಿಗೆ ಏಟಾಗಿ ನೋವಿದೆ, ಆದರೆ ಸ್ಥಿರವಾಗಿದ್ದಾನೆ"],
        ["ವ್ಯಕ್ತಿ ಮೊದಲ ಮಹಡಿಯಿಂದ ಬಿದ್ದ, ಎಚ್ಚರವಿದೆ ಆದರೆ ಕಾಲು ಅಲ್ಲಾಡಿಸಲು ಆಗುತ್ತಿಲ್ಲ, ತಲೆಯಿಂದ ರಕ್ತ ಸುರಿಯುತ್ತಿದೆ", "ಏಣಿಯಿಂದ ಬಿದ್ದ, ಕೈಯ ಮೂಳೆ ಹೊರಗೆ ಕಾಣುತ್ತಿದೆ, ತುಂಬಾ ರಕ್ತ ಹೋಗುತ್ತಿದೆ"],
        ["ಕೆಲಸಗಾರ ಕಟ್ಟಡದ ಮೇಲಿಂದ ಬಿದ್ದ, ಪ್ರಜ್ಞೆ ಇಲ್ಲ, ಸರಿಯಾಗಿ ಉಸಿರಾಡುತ್ತಿಲ್ಲ", "ಮೂರನೇ ಮಹಡಿಯಿಂದ ಬಿದ್ದ, ಉತ್ತರಿಸುತ್ತಿಲ್ಲ, ಕಿವಿಯಿಂದ ರಕ್ತ ಬರುತ್ತಿದೆ"]],
    "burn": [
        ["ಕೈಗೆ ಬಿಸಿ ಚಹಾ ಚೆಲ್ಲಿತು, ಸ್ವಲ್ಪ ಕೆಂಪಾಗಿದೆ, ಗುಳ್ಳೆ ಇಲ್ಲ, ಚೆನ್ನಾಗಿದ್ದಾರೆ", "ಬಿಸಿ ಕಾವಲಿ ಮುಟ್ಟಿದರು, ಸ್ವಲ್ಪ ಕೆಂಪಾಗಿದೆ, ಈಗ ನೋವಿಲ್ಲ"],
        ["ಕೈಗೆ ಬಿಸಿ ಎಣ್ಣೆ ಬಿದ್ದು ಸುಟ್ಟಿದೆ, ಗುಳ್ಳೆಗಳಿವೆ, ನೋವಿದೆ ಆದರೆ ಸಾಮಾನ್ಯವಾಗಿ ಮಾತನಾಡುತ್ತಿದ್ದಾರೆ", "ಮಗುವಿನ ಕಾಲಿಗೆ ಬಿಸಿ ನೀರು ಬಿತ್ತು, ಗುಳ್ಳೆಗಳಿವೆ, ಅಳುತ್ತಿದೆ ಆದರೆ ಎಚ್ಚರವಿದೆ"],
        ["ಗ್ಯಾಸ್ ಸಿಲಿಂಡರ್ ಬೆಂಕಿ ಹೊತ್ತಿತು, ಮುಖ ಮತ್ತು ತೋಳುಗಳಿಗೆ ಸುಟ್ಟಿದೆ, ಎಚ್ಚರವಿದೆ ಆದರೆ ತುಂಬಾ ನೋವು, ಉಸಿರಾಡಲು ಕಷ್ಟ", "ಅಡುಗೆಮನೆಯಲ್ಲಿ ಬೆಂಕಿ, ಕಾಲುಗಳಿಗೆ ದೊಡ್ಡ ಸುಟ್ಟ ಗಾಯ, ಚರ್ಮ ಎದ್ದು ಬರುತ್ತಿದೆ, ಎಚ್ಚರವಿದ್ದಾರೆ"],
        ["ಮನೆಗೆ ಬೆಂಕಿ ಬಿತ್ತು, ವ್ಯಕ್ತಿಯನ್ನು ಪ್ರಜ್ಞೆ ಇಲ್ಲದೆ ಹೊರಗೆ ತೆಗೆದರು, ಇಡೀ ದೇಹ ಸುಟ್ಟಿದೆ, ಉಸಿರಾಡುತ್ತಿಲ್ಲ", "ವಿದ್ಯುತ್ ಶಾಕ್ ಹೊಡೆದು ಸುಟ್ಟಿದೆ, ಕುಸಿದು ಬಿದ್ದ, ಉತ್ತರವಿಲ್ಲ, ನಾಡಿ ಸಿಗುತ್ತಿಲ್ಲ"]],
    "cardiac_medical": [
        ["ವ್ಯಕ್ತಿಗೆ ಒಂದು ನಿಮಿಷ ತಲೆ ಸುತ್ತು ಬಂತು, ಈಗ ಕುಳಿತು ನೀರು ಕುಡಿಯುತ್ತಿದ್ದಾರೆ, ಸಹಜವಾಗಿದ್ದಾರೆ", "ಮಹಿಳೆಗೆ ಬಿಸಿಲಿನಿಂದ ಸುಸ್ತಾಗಿದೆ, ಚೆನ್ನಾಗಿ ಮಾತನಾಡುತ್ತಿದ್ದಾರೆ, ನೋವಿಲ್ಲ"],
        ["ವೃದ್ಧರಿಗೆ ಎದೆಯಲ್ಲಿ ಸ್ವಲ್ಪ ಕಿರಿಕಿರಿ ಇದೆ, ಕುಳಿತು ಮಾತನಾಡುತ್ತಿದ್ದಾರೆ, ಸ್ವಲ್ಪ ಬೆವರುತ್ತಿದ್ದಾರೆ", "ಹುಡುಗಿ ಸ್ವಲ್ಪ ಹೊತ್ತು ಪ್ರಜ್ಞೆ ತಪ್ಪಿದಳು, ಈಗ ಎಚ್ಚರವಾಗಿದೆ, ಬಿಳಿಚಿಕೊಂಡಿದ್ದಾಳೆ ಆದರೆ ಉತ್ತರಿಸುತ್ತಿದ್ದಾಳೆ"],
        ["ವ್ಯಕ್ತಿಗೆ ಎದೆಯಲ್ಲಿ ತೀವ್ರ ನೋವು, ತುಂಬಾ ಬೆವರುತ್ತಿದ್ದಾರೆ, ಉಸಿರಾಟ ಕಷ್ಟ, ಎಚ್ಚರವಿದ್ದಾರೆ", "ಈಗಷ್ಟೇ ಫಿಟ್ಸ್ ನಿಂತಿದೆ, ಅವನು ಮಂಪರಿನಲ್ಲಿದ್ದಾನೆ, ಗೊಂದಲದಲ್ಲಿದ್ದಾನೆ, ವೇಗವಾಗಿ ಉಸಿರಾಡುತ್ತಿದ್ದಾನೆ"],
        ["ವ್ಯಕ್ತಿ ಒಮ್ಮೆಗೇ ಕುಸಿದು ಬಿದ್ದ, ಪ್ರಜ್ಞೆ ಇಲ್ಲ, ಉಸಿರಾಡುತ್ತಿಲ್ಲ, ನಾಡಿ ಇಲ್ಲ", "ಮಹಿಳೆ ಕುಸಿದು ಬಿದ್ದರು, ತುಟಿ ನೀಲಿಯಾಗಿದೆ, ಉತ್ತರಿಸುತ್ತಿಲ್ಲ, ಉಸಿರಿಲ್ಲ"]],
    "other": [
        ["ಜನರ ಗುಂಪು ಕಂಡಿತು, ಏನಾದರೂ ಆಯಿತೋ ಗೊತ್ತಿಲ್ಲ, ಯಾರಿಗೂ ಗಾಯವಾದಂತೆ ಕಾಣುತ್ತಿಲ್ಲ", "ಮೊದಲು ಜಗಳವಾಗಿತ್ತು ಆದರೆ ಎಲ್ಲರೂ ಹೊರಟುಹೋದರು, ಯಾರಿಗೂ ಗಾಯವಿಲ್ಲ"],
        ["ಹುಡುಗನಿಗೆ ಕ್ರಿಕೆಟ್ ಚೆಂಡು ತಗುಲಿತು, ಮೂಗಿನಿಂದ ಸ್ವಲ್ಪ ರಕ್ತ ಬಂತು, ಈಗ ನಿಂತಿದೆ, ಎಚ್ಚರವಾಗಿದ್ದಾನೆ", "ನಾಯಿ ಕಾಲಿಗೆ ಕಚ್ಚಿತು, ಚಿಕ್ಕ ಗಾಯ, ಸ್ವಲ್ಪ ರಕ್ತ ಬರುತ್ತಿದೆ, ನಡೆಯುತ್ತಿದ್ದಾರೆ"],
        ["ಹೊಟ್ಟೆಗೆ ಚಾಕು ಇರಿದಿದೆ, ತುಂಬಾ ರಕ್ತ ಹೋಗುತ್ತಿದೆ, ಎಚ್ಚರವಿದೆ ಆದರೆ ನಿಶ್ಶಕ್ತನಾಗಿದ್ದಾನೆ", "ತಲೆಗೆ ರಾಡ್‌ನಿಂದ ಹೊಡೆದಿದ್ದಾರೆ, ತುಂಬಾ ರಕ್ತ ಹೋಗುತ್ತಿದೆ, ಗೊಂದಲದಲ್ಲಿದ್ದಾನೆ"],
        ["ರಸ್ತೆಯಲ್ಲಿ ಒಬ್ಬ ವ್ಯಕ್ತಿ ಸಿಕ್ಕಿದ, ಪ್ರಜ್ಞೆ ಇಲ್ಲ, ಉಸಿರಾಡುತ್ತಿಲ್ಲ, ತುಂಬಾ ರಕ್ತವಿದೆ", "ನೀರಿನಿಂದ ಹೊರತೆಗೆದರು, ಪ್ರಜ್ಞೆ ಇಲ್ಲ, ಉಸಿರಾಡುತ್ತಿಲ್ಲ, ಮುಖ ನೀಲಿಯಾಗಿದೆ"]],
}
# ROMAN[lang][type] = one phrase per severity (typing / ASR-transliteration style).
HI_ROMAN = {
    "road_accident": ["bike aapas me touch hui, kisi ko chot nahi lagi, bas scratch hai", "bike slip ho gayi, uske haath me moch hai, baat kar raha hai", "bike bus se takrayi, pair toota lag raha hai, bahut khoon beh raha hai, hosh me hai", "truck ne bike ko mara, banda sadak pe pada hai, hil nahi raha, saans nahi le raha"],
    "fall": ["bachcha bench se gir gaya, ro raha tha, ab theek hai", "aunty seedhi se gir gayi, pair sooj gaya hai, hosh me hai", "aadmi pehli manzil se gira, hosh me hai par pair nahi hil rahe, sar se khoon beh raha hai", "mazdoor upar se gir gaya, behosh hai, saans theek se nahi aa rahi"],
    "burn": ["chai ke chheente haath pe pade, thodi lali hai, theek hai", "tel se haath jal gaya, chhale hain, dard hai par baat kar raha hai", "cylinder me aag lagi, chehre aur haathon pe jalan, hosh me hai par bahut dard hai", "ghar me aag lagi, bande ko behosh nikala, poora badan jala hai, saans nahi aa rahi"],
    "cardiac_medical": ["chakkar aaya tha ek minute, ab baith ke paani pi raha hai, theek hai", "uncle ko seene me halka dard hai, baith ke baat kar rahe hain, pasina aa raha hai", "seene me tez dard hai, bahut pasina, saans phool rahi hai, hosh me hai", "aadmi achanak gir gaya, behosh hai, saans nahi aa rahi, nabz nahi mil rahi"],
    "other": ["bheed dikhi, pata nahi kuch hua ya nahi, koi ghayal nahi dikh raha", "ladke ko cricket ball lagi, naak se thoda khoon aaya, ab ruk gaya, alert hai", "pet me chaku laga hai, bahut khoon beh raha hai, hosh me hai par kamzor hai", "sadak pe aadmi mila, behosh hai, saans nahi le raha, bahut khoon hai"],
}
KN_ROMAN = {
    "road_accident": ["eradu bike swalpa thagalitu, yarigu gaaya aagilla, swalpa tharachide ashte", "bike jaari bittu, savaarana kai uLukide, matadtha idhane", "bike bus ge dikki hodeyitu, kaalu muridide ansutte, tumba raktha hogtha ide, eccharadalli idhane", "lorry bike ge dikki hodeyitu, savaara rasteyalli bidhidhane, alugaadtilla, usiraadtilla"],
    "fall": ["magu bench inda bittu, atthu, ega matte aadtha ide, gaaya illa", "mahile metlinda bidru, kaalu uditu, eccharadalli idhare", "vyakthi modala mahadiyinda bidda, eccharavide aadre kaalu alugaadsakke aagtilla, thaleyinda raktha suriyuttide", "kelasagaara kattadada melinda bidda, pragne illa, sariyaagi usiraadtilla"],
    "burn": ["kaige bisi chaha chellitu, swalpa kempaagide, ega novilla", "bisi enne bidhu kai sutthide, gulle ide, novide aadre maatadtha idhare", "gas cylinder benki hottitu, mukha mattu tholu sutthide, eccharavide aadre tumba novu", "mane ge benki bittu, pragne illade horage tegedaru, idi deha sutthide, usiraadtilla"],
    "cardiac_medical": ["ondu nimisha thale sutthu banthu, ega kuthu neeru kudithidhare, sahajavidhare", "vrudharige edeyalli swalpa kirikiri ide, kuthu maatadtha idhare, swalpa bevarthidhare", "edeyalli theevra novu, tumba bevarthidhare, usiraata kashta, eccharavidhare", "vyakthi omme kusidhu bidda, pragne illa, usiraadtilla, naadi illa"],
    "other": ["jana gumpu kandithu, enaadru aayitho gothilla, yarigu gaaya kaanthilla", "huduganige cricket chendu thagulitu, mooginiinda swalpa raktha banthu, ega nindide, eccharavide", "hotte ge chaaku irididhe, tumba raktha hogtha ide, eccharavide aadre nishakta", "rasteyalli obba vyakthi sikkida, pragne illa, usiraadtilla, tumba raktha ide"],
}

OPEN = {
    "en": ["Please send an ambulance, ", "Help, ", "There is an accident near the signal, ", "Hello, emergency, ", "", "Sir please come fast, "],
    "hi": ["कृपया एम्बुलेंस भेजिए, ", "मदद कीजिए, ", "सिग्नल के पास हादसा हुआ है, ", "हेलो, इमरजेंसी है, ", "", "जल्दी आइए, "],
    "kn": ["ದಯವಿಟ್ಟು ಆಂಬ್ಯುಲೆನ್ಸ್ ಕಳುಹಿಸಿ, ", "ಸಹಾಯ ಮಾಡಿ, ", "ಸಿಗ್ನಲ್ ಹತ್ತಿರ ಅಪಘಾತವಾಗಿದೆ, ", "ಹಲೋ, ತುರ್ತು ಪರಿಸ್ಥಿತಿ, ", "", "ಬೇಗ ಬನ್ನಿ, "],
    "hi_r": ["please ambulance bhejiye, ", "madad kijiye, ", "jaldi aaiye, ", ""],
    "kn_r": ["dayavittu ambulance kaluhisi, ", "sahaya maadi, ", "begane banni, ", ""],
}
SUFFIX = {
    "en": ["", " Please hurry.", " I am at the roadside.", " People are gathering around.", " Please come fast!", " I don't know what to do."],
    "hi": ["", " जल्दी कीजिए।", " मैं सड़क किनारे हूँ।", " लोग इकट्ठा हो गए हैं।", " कृपया जल्दी आइए!", " मुझे समझ नहीं आ रहा क्या करूँ।"],
    "kn": ["", " ಬೇಗ ಬನ್ನಿ.", " ನಾನು ರಸ್ತೆ ಬದಿಯಲ್ಲಿದ್ದೇನೆ.", " ಜನ ಸೇರಿದ್ದಾರೆ.", " ದಯವಿಟ್ಟು ಬೇಗ ಬನ್ನಿ!", " ಏನು ಮಾಡಬೇಕೋ ಗೊತ್ತಾಗುತ್ತಿಲ್ಲ."],
    "hi_r": ["", " jaldi kijiye.", " log jama ho gaye hain."],
    "kn_r": ["", " begane banni.", " jana serideare."],
}
NATIVE = {"en": EN, "hi": HI, "kn": KN}
ROMAN = {"hi": HI_ROMAN, "kn": KN_ROMAN}
N_ROMAN_PER_CELL = 2     # hi/kn only: 40 of 150 reports in romanised typing style


def _draft(rng: random.Random, lang: str, style: str, typ: str, sev: int, seen: set) -> str:
    for _ in range(200):
        if style == "native":
            core, key = rng.choice(NATIVE[lang][typ][sev]), lang
        else:
            core, key = ROMAN[lang][typ][sev], f"{lang}_r"
        suffix = rng.choice(SUFFIX[key])
        stop = "।" if key == "hi" else "."
        text = (rng.choice(OPEN[key]) + core + (stop + suffix if suffix else "")).strip()
        if lang == "en":
            text = text[0].upper() + text[1:]
        if text not in seen:
            seen.add(text)
            return text
    raise RuntimeError("template grid exhausted; lower --per-language")


def build(per_language: int = 150, seed: int = SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    cells = [(t, s) for t in TYPES for s in range(4)]
    rows = []
    for lang in ("en", "hi", "kn"):
        seen: set[str] = set()
        roman = N_ROMAN_PER_CELL if lang in ROMAN else 0
        plan = [(t, s, "native") for t, s in cells for _ in range(per_language // 20 - roman)]
        plan += [(t, s, "roman") for t, s in cells for _ in range(roman)]
        plan += [(t, s, "native") for t, s in rng.sample(cells, per_language - len(plan))]
        for typ, sev, style in plan:
            rows.append({"language": lang, "text": _draft(rng, lang, style, typ, sev, seen),
                         "label": sev, "reviewed": 0, "split": "",
                         "notes": f"draft-v1 type={typ} script={style}"})
    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df.insert(0, "report_id", df.language + "-" + (df.groupby("language").cumcount() + 1)
              .map(lambda n: f"{n:04d}"))
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Draft the text-severity report CSV (reviewed=0)")
    p.add_argument("--per-language", type=int, default=150)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--force", action="store_true", help="overwrite (destroys any review work!)")
    a = p.parse_args()
    if a.out.exists() and not a.force:
        raise SystemExit(f"{a.out} exists (may hold reviewed rows); use --force to overwrite")
    df = build(a.per_language)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False, encoding="utf-8-sig")   # BOM so Excel shows Devanagari/Kannada
    print(f"wrote {len(df)} draft rows to {a.out} (all reviewed=0)")
    print(df.groupby(["language", "label"]).size().unstack().to_string())


if __name__ == "__main__":
    main()
