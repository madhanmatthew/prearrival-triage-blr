# Held-out ASR/triage clips (10 per language)

Purpose: a clean `eval_triage` number. The existing 90 clips were used to tune the hi/kn cue
lexicon, so these must be NEW recordings.

How to record
- Different speaker(s) from the first 90 if possible. Phone, 5-15 s each, quiet room (a few in noise is fine).
- Read the scenario FACTS, then say it in your OWN natural words (do not read the sample line
  word for word). Do not look at `backend/dialogue/lexicon.py` first.
- Save as `heldout/en_h01.wav` ... `hi_h01.m4a` ... `kn_h01.m4a` (h01-h10, same scenario per number).
- Then edit `heldout_reference.csv`: set `reference` to the exact words you said (native script).
  Get consent from each speaker.

| # | Facts (say these in your own words) | Sample en | Sample hi | Sample kn |
|---|---|---|---|---|
| h01 | Auto hit a bike at Hebbal flyover; rider lying still, not responding; 2 injured | A speeding auto hit a bike on Hebbal flyover, the rider is lying still and not responding, two people are hurt. | हेब्बाल फ्लाईओवर पर ऑटो ने बाइक को टक्कर मार दी, बाइक वाला हिल नहीं रहा, कुछ बोल नहीं रहा, दो लोग घायल हैं। | ಹೆಬ್ಬಾಳ ಫ್ಲೈಓವರ್ ಹತ್ರ ಆಟೋ ಬೈಕ್‌ಗೆ ಗುದ್ದಿದೆ, ಬೈಕ್ ಸವಾರ ಅಲ್ಲಾಡ್ತಿಲ್ಲ, ಮಾತಾಡ್ತಿಲ್ಲ, ಇಬ್ರಿಗೆ ಪೆಟ್ಟಾಗಿದೆ. |
| h02 | Old man slipped in bathroom at home (Jayanagar); hip pain, cannot stand up | My grandfather slipped in the bathroom, his hip hurts a lot and he cannot stand up. | मेरे दादाजी बाथरूम में फिसल गए, कूल्हे में बहुत दर्द है और वो खड़े नहीं हो पा रहे। | ನಮ್ ತಾತ ಬಾತ್‌ರೂಮ್‌ನಲ್ಲಿ ಜಾರಿ ಬಿದ್ರು, ಸೊಂಟ ತುಂಬಾ ನೋವು, ಎದ್ದು ನಿಲ್ಲೋಕೆ ಆಗ್ತಿಲ್ಲ. |
| h03 | Toddler burned by boiling water spilled on arm and chest (Koramangala) | Boiling water fell on my little girl, her arm and chest are burnt and she is screaming. | मेरी छोटी बेटी पर गरम पानी गिर गया, उसका हाथ और छाती जल गई है और वो बहुत रो रही है। | ನಮ್ ಪುಟ್ಟ ಮಗಳ ಮೇಲೆ ಬಿಸಿ ನೀರು ಬಿದ್ದಿದೆ, ಕೈ ಮತ್ತೆ ಎದೆ ಸುಟ್ಟಿದೆ, ತುಂಬಾ ಅಳ್ತಿದ್ದಾಳೆ. |
| h04 | Man at Majestic bus stand: chest pressure, pain down left arm, sweating | A man at Majestic bus stand has pain in his chest going down his left arm and he is sweating. | मेजेस्टिक बस स्टैंड पर एक आदमी के सीने में दर्द है, बायां हाथ भी दर्द कर रहा है और पसीना आ रहा है। | ಮೆಜೆಸ್ಟಿಕ್ ಬಸ್ ಸ್ಟ್ಯಾಂಡ್‌ನಲ್ಲಿ ಒಬ್ರಿಗೆ ಎದೆ ನೋವು, ಎಡ ಕೈ ಕೂಡ ನೋವು, ಬೆವರ್ತಿದಾರೆ. |
| h05 | Student having a seizure in college canteen (Yelahanka), body jerking | A girl is having a seizure in our college canteen, her body is jerking and her eyes are rolled back. | हमारे कॉलेज कैंटीन में एक लड़की को दौरा पड़ा है, उसका पूरा शरीर झटके खा रहा है। | ನಮ್ ಕಾಲೇಜ್ ಕ್ಯಾಂಟೀನ್‌ನಲ್ಲಿ ಒಂದು ಹುಡುಗಿಗೆ ಫಿಟ್ಸ್ ಬಂದಿದೆ, ಮೈ ಎಲ್ಲಾ ಜರ್ಕ್ ಆಗ್ತಿದೆ. |
| h06 | Man choking on food in a restaurant (Indiranagar), cannot speak or cough | A man is choking on food in a restaurant, he cannot speak and he is holding his throat. | रेस्टोरेंट में एक आदमी के गले में खाना फंस गया है, वो बोल नहीं पा रहा और गला पकड़े हुए है। | ರೆಸ್ಟೋರೆಂಟ್‌ನಲ್ಲಿ ಒಬ್ರ ಗಂಟಲಲ್ಲಿ ಊಟ ಸಿಕ್ಕಿಕೊಂಡಿದೆ, ಮಾತಾಡೋಕೆ ಆಗ್ತಿಲ್ಲ, ಗಂಟಲು ಹಿಡ್ಕೊಂಡಿದಾರೆ. |
| h07 | Woman's face drooping on one side, speech slurred, arm weak (stroke) | My aunt's face is drooping on one side, her speech is slurred and she cannot lift her right arm. | मेरी आंटी का चेहरा एक तरफ टेढ़ा हो गया है, बोलने में दिक्कत है और दायां हाथ उठा नहीं पा रहीं। | ನಮ್ ಆಂಟಿ ಮುಖ ಒಂದು ಕಡೆ ಸೊಟ್ಟಗಾಗಿದೆ, ಮಾತು ಥೊದಲ್ತಿದೆ, ಬಲ ಕೈ ಎತ್ತೋಕೆ ಆಗ್ತಿಲ್ಲ. |
| h08 | Factory worker (Peenya) hand caught in machine, blood spurting | A worker's hand got caught in a machine at Peenya and blood is pouring out, it will not stop. | पीन्या की फैक्ट्री में मशीन में एक वर्कर का हाथ फंस गया है, बहुत खून बह रहा है, रुक नहीं रहा। | ಪೀಣ್ಯದ ಫ್ಯಾಕ್ಟರಿಯಲ್ಲಿ ಮಷೀನ್‌ಗೆ ಒಬ್ಬ ವರ್ಕರ್ ಕೈ ಸಿಕ್ಕಿಕೊಂಡಿದೆ, ತುಂಬಾ ರಕ್ತ ಬರ್ತಿದೆ, ನಿಲ್ತಿಲ್ಲ. |
| h09 | Lorry overturned on Tumkur Road; 3 injured, one unconscious | A lorry overturned on Tumkur Road, three people are injured and one of them is unconscious. | तुमकुर रोड पर लॉरी पलट गई है, तीन लोग घायल हैं और एक बेहोश है। | ತುಮಕೂರು ರೋಡ್‌ನಲ್ಲಿ ಲಾರಿ ಬಿದ್ದಿದೆ, ಮೂರು ಜನಕ್ಕೆ ಪೆಟ್ಟಾಗಿದೆ, ಒಬ್ರು ಪ್ರಜ್ಞೆ ಇಲ್ಲ. |
| h10 | Asthma patient (Whitefield) wheezing, cannot finish sentences | My brother has asthma, he is wheezing and cannot speak a full sentence, he is struggling to breathe. | मेरे भाई को अस्थमा है, उसे सांस लेने में बहुत तकलीफ हो रही है, पूरा वाक्य भी नहीं बोल पा रहा। | ನಮ್ ತಮ್ಮಂಗೆ ಅಸ್ತಮಾ ಇದೆ, ಉಸಿರಾಡೋಕೆ ತುಂಬಾ ಕಷ್ಟ ಆಗ್ತಿದೆ, ಪೂರ್ತಿ ವಾಕ್ಯ ಮಾತಾಡೋಕೆ ಆಗ್ತಿಲ್ಲ. |
