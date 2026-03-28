"""
Advanced Sentiment & Intent Engine for Smart DentalOps
Multi-layer analysis combining VADER + dental domain lexicon + contextual rules

Layers:
  1. VADER sentiment scoring (much better than TextBlob for short texts)
  2. Dental domain lexicon (custom weights for medical/dental terms)
  3. Negation handling ("not happy", "no pain", "didn't hurt")
  4. Emotion classification (anxiety, satisfaction, frustration, trust, urgency)
  5. Intent detection with confidence scoring
  6. Risk adjustment with weighted multi-signal fusion
"""

import re
from typing import Optional

# ── Try to import VADER, fall back to TextBlob if not installed ───────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    from textblob import TextBlob
    VADER_AVAILABLE = False

# ── Dental domain lexicon ─────────────────────────────────────────────────────
# Positive words in dental context → lower no-show risk
DENTAL_POSITIVE = {
    "excited": 0.8, "looking forward": 0.9, "ready": 0.7, "confirmed": 0.9,
    "see you": 0.8, "on my way": 0.95, "be there": 0.9, "will come": 0.9,
    "no pain": 0.7, "feeling better": 0.8, "healed": 0.8, "recovered": 0.8,
    "great": 0.7, "wonderful": 0.8, "excellent": 0.8, "happy": 0.7,
    "thank you": 0.6, "thanks": 0.6, "appreciate": 0.7, "helpful": 0.7,
    "comfortable": 0.7, "painless": 0.8, "smooth": 0.6, "easy": 0.6,
    "perfect": 0.8, "amazing": 0.8, "love": 0.7, "satisfied": 0.8,
    "relief": 0.7, "relieved": 0.7, "better": 0.6, "improved": 0.7,
    "clean": 0.6, "fresh": 0.6, "good": 0.5, "fine": 0.4, "okay": 0.3,
}

# Negative words in dental context → higher no-show risk
DENTAL_NEGATIVE = {
    "cancel": -0.95, "cancelling": -0.95, "cancelled": -0.95,
    "reschedule": -0.7, "rescheduling": -0.7, "postpone": -0.7,
    "can't make": -0.9, "cannot make": -0.9, "won't be": -0.9,
    "not coming": -0.9, "skip": -0.8, "miss": -0.7, "missed": -0.7,
    "afraid": -0.6, "scared": -0.7, "terrified": -0.8, "fear": -0.6,
    "anxious": -0.5, "anxiety": -0.5, "nervous": -0.5, "worried": -0.5,
    "pain": -0.4, "hurt": -0.4, "hurts": -0.4, "painful": -0.5,
    "terrible": -0.7, "awful": -0.7, "horrible": -0.8, "bad": -0.5,
    "hate": -0.8, "dislike": -0.6, "unhappy": -0.6, "disappointed": -0.6,
    "expensive": -0.4, "costly": -0.4, "unaffordable": -0.6,
    "insurance": -0.2, "billing": -0.2, "bill": -0.2, "cost": -0.2,
    "late": -0.3, "running late": -0.4, "delay": -0.4, "delayed": -0.4,
    "busy": -0.3, "emergency": -0.3, "sick": -0.4, "ill": -0.4,
    "problem": -0.4, "issue": -0.3, "complaint": -0.5, "wrong": -0.4,
    "swelling": -0.5, "bleeding": -0.5, "infection": -0.6, "abscess": -0.7,
}

# Negation words that flip sentiment
NEGATION_WORDS = {
    "not", "no", "never", "don't", "doesn't", "didn't", "won't", "can't",
    "cannot", "isn't", "aren't", "wasn't", "weren't", "hardly", "barely",
    "scarcely", "nothing", "nobody", "nowhere", "neither", "nor"
}

# Intensifiers that amplify sentiment
INTENSIFIERS = {
    "very": 1.3, "really": 1.3, "extremely": 1.5, "absolutely": 1.4,
    "totally": 1.3, "completely": 1.4, "so": 1.2, "quite": 1.1,
    "super": 1.3, "incredibly": 1.5, "terribly": 1.4, "awfully": 1.4,
}

# Emotion categories with keyword triggers
EMOTION_KEYWORDS = {
    "anxiety": ["scared", "afraid", "terrified", "nervous", "anxious", "fear", "phobia", "dread", "worried", "panic"],
    "satisfaction": ["happy", "satisfied", "pleased", "great", "excellent", "wonderful", "love", "perfect", "amazing", "fantastic"],
    "frustration": ["frustrated", "annoyed", "irritated", "upset", "angry", "mad", "furious", "disappointed", "fed up"],
    "trust": ["trust", "confident", "comfortable", "safe", "reliable", "professional", "caring", "gentle", "kind"],
    "urgency": ["urgent", "emergency", "asap", "immediately", "right away", "severe", "extreme", "critical", "serious"],
    "pain": ["pain", "hurt", "ache", "sore", "throbbing", "sharp", "burning", "swelling", "bleeding", "infection"],
}

# Intent patterns with confidence weights
INTENT_PATTERNS = {
    "Cancellation": {
        "high": ["cancel", "cancelling", "cancelled", "can't make it", "cannot make it", "won't be there",
                 "not coming", "have to cancel", "need to cancel", "want to cancel", "unable to attend",
                 "won't make", "can't attend", "not able to come"],
        "medium": ["skip", "miss", "might not", "probably won't", "don't think i can"],
        "low": ["maybe not", "not sure if", "might cancel", "thinking of cancelling"],
    },
    "Delay": {
        "high": ["running late", "will be late", "reschedule", "change the time", "move appointment",
                 "postpone", "push back", "different time", "another day", "different day"],
        "medium": ["late", "delay", "behind schedule", "stuck in traffic", "held up", "bit late"],
        "low": ["might be late", "could be late", "possibly late", "running a bit behind"],
    },
    "Confirmation": {
        "high": ["confirm", "confirmed", "will be there", "see you", "on my way", "be there",
                 "looking forward", "ready for", "all set", "definitely coming", "yes i'll be there"],
        "medium": ["yes", "okay", "ok", "sure", "fine", "will come", "coming", "attending"],
        "low": ["probably", "should be there", "planning to come", "hope to make it"],
    },
    "Inquiry": {
        "high": ["what time", "where is", "how long", "what should i", "do i need to", "can i",
                 "is it okay", "what happens", "will there be", "how much"],
        "medium": ["question", "wondering", "want to know", "need to know", "asking about"],
        "low": ["curious", "just checking", "wanted to ask"],
    },
    "Complaint": {
        "high": ["not happy", "very disappointed", "terrible experience", "worst", "unacceptable",
                 "this is wrong", "this is bad", "completely wrong"],
        "medium": ["disappointed", "unhappy", "not satisfied", "problem with", "issue with", "complaint"],
        "low": ["not great", "could be better", "wasn't ideal"],
    },
}


# ── Core analysis functions ───────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Clean and normalize text"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s\'\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def detect_negation_windows(tokens: list) -> set:
    """Return indices of tokens that are negated (within 3 words of a negation word)"""
    negated = set()
    for i, token in enumerate(tokens):
        if token in NEGATION_WORDS:
            for j in range(i + 1, min(i + 4, len(tokens))):
                negated.add(j)
    return negated

def get_vader_score(text: str) -> dict:
    """Get VADER compound score and components"""
    if VADER_AVAILABLE:
        scores = _vader.polarity_scores(text)
        return {
            "compound": scores["compound"],
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
        }
    else:
        from textblob import TextBlob
        blob = TextBlob(text)
        pol = blob.sentiment.polarity
        return {
            "compound": pol,
            "positive": max(0, pol),
            "negative": max(0, -pol),
            "neutral": 1 - abs(pol),
        }

def get_domain_score(text: str) -> float:
    """Score text using dental domain lexicon with negation and intensifier handling"""
    tokens = preprocess(text).split()
    negated_indices = detect_negation_windows(tokens)
    
    domain_score = 0.0
    matched_count = 0
    
    # Check multi-word phrases first
    for phrase, weight in {**DENTAL_POSITIVE, **DENTAL_NEGATIVE}.items():
        if ' ' in phrase and phrase in text.lower():
            domain_score += weight
            matched_count += 1
    
    # Check single words with negation and intensifier handling
    for i, token in enumerate(tokens):
        # Check intensifier before this token
        intensifier = 1.0
        if i > 0 and tokens[i - 1] in INTENSIFIERS:
            intensifier = INTENSIFIERS[tokens[i - 1]]
        
        if token in DENTAL_POSITIVE:
            score = DENTAL_POSITIVE[token] * intensifier
            if i in negated_indices:
                score = -score * 0.8  # negated positive → negative
            domain_score += score
            matched_count += 1
        elif token in DENTAL_NEGATIVE:
            score = DENTAL_NEGATIVE[token] * intensifier
            if i in negated_indices:
                score = -score * 0.6  # negated negative → slightly positive
            domain_score += score
            matched_count += 1
    
    # Normalize
    if matched_count > 0:
        domain_score = max(-1.0, min(1.0, domain_score / max(matched_count, 1)))
    
    return round(domain_score, 3)

def classify_emotions(text: str) -> dict:
    """Detect emotions present in the text"""
    text_lower = text.lower()
    detected = {}
    
    for emotion, keywords in EMOTION_KEYWORDS.items():
        matches = [kw for kw in keywords if kw in text_lower]
        if matches:
            detected[emotion] = {
                "detected": True,
                "keywords": matches,
                "intensity": min(1.0, len(matches) * 0.3 + 0.4),
            }
    
    return detected

def detect_intent_advanced(text: str) -> dict:
    """Multi-confidence intent detection"""
    text_lower = text.lower()
    
    best_intent = "Unknown"
    best_confidence = "Low"
    best_score = 0
    matched_keywords = []
    all_intents = {}
    
    confidence_weights = {"high": 3, "medium": 2, "low": 1}
    
    for intent, levels in INTENT_PATTERNS.items():
        intent_score = 0
        intent_keywords = []
        intent_confidence = "Low"
        
        for level, patterns in levels.items():
            for pattern in patterns:
                if pattern in text_lower:
                    intent_score += confidence_weights[level]
                    intent_keywords.append(pattern)
                    if level == "high":
                        intent_confidence = "High"
                    elif level == "medium" and intent_confidence != "High":
                        intent_confidence = "Medium"
        
        if intent_score > 0:
            all_intents[intent] = {
                "score": intent_score,
                "confidence": intent_confidence,
                "keywords": intent_keywords,
            }
            if intent_score > best_score:
                best_score = intent_score
                best_intent = intent
                best_confidence = intent_confidence
                matched_keywords = intent_keywords
    
    return {
        "intent": best_intent,
        "confidence": best_confidence,
        "keywords": matched_keywords,
        "all_detected": all_intents,
        "score": best_score,
    }

def fuse_scores(vader_compound: float, domain_score: float) -> float:
    """
    Weighted fusion of VADER and domain scores.
    VADER is better for general sentiment, domain is better for dental context.
    """
    # If domain has strong signal, weight it more
    domain_weight = 0.6 if abs(domain_score) > 0.3 else 0.3
    vader_weight = 1.0 - domain_weight
    
    fused = (vader_compound * vader_weight) + (domain_score * domain_weight)
    return round(max(-1.0, min(1.0, fused)), 3)

def categorize_sentiment(score: float) -> str:
    """Convert numeric score to category with finer granularity"""
    if score >= 0.5:
        return "very_positive"
    elif score >= 0.15:
        return "positive"
    elif score >= -0.15:
        return "neutral"
    elif score >= -0.5:
        return "negative"
    else:
        return "very_negative"

def calculate_risk_adjustment(
    sentiment_score: float,
    intent: str,
    intent_confidence: str,
    emotions: dict,
    current_risk: float
) -> dict:
    """
    Multi-signal risk adjustment with weighted fusion.
    
    Signals:
    - Sentiment score (continuous -1 to +1)
    - Intent (categorical with confidence)
    - Emotions (anxiety, frustration, urgency)
    """
    adjustments = []
    explanations = []
    
    # 1. Sentiment-based adjustment
    if sentiment_score >= 0.5:
        adj = -12.0
        explanations.append("Very positive sentiment detected — patient highly engaged")
    elif sentiment_score >= 0.15:
        adj = -5.0
        explanations.append("Positive sentiment — patient appears satisfied")
    elif sentiment_score >= -0.15:
        adj = 0.0
        explanations.append("Neutral sentiment — standard communication")
    elif sentiment_score >= -0.5:
        adj = +12.0
        explanations.append("Negative sentiment — patient may be dissatisfied")
    else:
        adj = +20.0
        explanations.append("Very negative sentiment — high dissatisfaction risk")
    adjustments.append(adj)
    
    # 2. Intent-based adjustment
    confidence_multipliers = {"High": 1.0, "Medium": 0.6, "Low": 0.3}
    mult = confidence_multipliers.get(intent_confidence, 0.5)
    
    intent_adjustments = {
        "Cancellation": +30.0,
        "Delay": +8.0,
        "Confirmation": -15.0,
        "Inquiry": +2.0,
        "Complaint": +15.0,
        "Unknown": 0.0,
    }
    intent_adj = intent_adjustments.get(intent, 0.0) * mult
    if intent != "Unknown":
        explanations.append(f"{intent_confidence} confidence {intent} intent detected ({intent_adj:+.1f}% risk)")
    adjustments.append(intent_adj)
    
    # 3. Emotion-based adjustments
    if "anxiety" in emotions:
        intensity = emotions["anxiety"]["intensity"]
        adj = +8.0 * intensity
        adjustments.append(adj)
        explanations.append(f"Dental anxiety detected ({adj:+.1f}% risk)")
    
    if "frustration" in emotions:
        intensity = emotions["frustration"]["intensity"]
        adj = +10.0 * intensity
        adjustments.append(adj)
        explanations.append(f"Frustration detected ({adj:+.1f}% risk)")
    
    if "urgency" in emotions:
        adj = +5.0
        adjustments.append(adj)
        explanations.append("Urgency signals detected")
    
    if "satisfaction" in emotions:
        intensity = emotions["satisfaction"]["intensity"]
        adj = -8.0 * intensity
        adjustments.append(adj)
        explanations.append(f"Satisfaction signals detected ({adj:+.1f}% risk)")
    
    if "trust" in emotions:
        adj = -6.0
        adjustments.append(adj)
        explanations.append("Trust/comfort signals detected (-6% risk)")
    
    # Total adjustment (cap at ±40)
    total_adj = max(-40.0, min(40.0, sum(adjustments)))
    adjusted_risk = max(0.0, min(100.0, current_risk + total_adj))
    
    return {
        "adjustedRisk": round(adjusted_risk, 1),
        "riskChange": round(total_adj, 1),
        "explanation": " | ".join(explanations) if explanations else "No significant signals detected.",
        "breakdown": {
            "sentimentAdjustment": adjustments[0] if adjustments else 0,
            "intentAdjustment": adjustments[1] if len(adjustments) > 1 else 0,
            "emotionAdjustments": adjustments[2:] if len(adjustments) > 2 else [],
        }
    }


# ── Main public API ───────────────────────────────────────────────────────────

def analyze_full(message: str, current_risk: float = 50.0) -> dict:
    """
    Full multi-layer sentiment and intent analysis.
    
    Returns comprehensive analysis including:
    - VADER scores
    - Domain-specific scores
    - Fused sentiment score
    - Emotion classification
    - Intent detection
    - Risk adjustment
    """
    # Layer 1: VADER
    vader = get_vader_score(message)
    
    # Layer 2: Domain lexicon
    domain_score = get_domain_score(message)
    
    # Layer 3: Fused score
    fused_score = fuse_scores(vader["compound"], domain_score)
    
    # Layer 4: Sentiment category
    sentiment_category = categorize_sentiment(fused_score)
    # Map to simple 3-way for backward compatibility
    simple_sentiment = "positive" if fused_score > 0.1 else ("negative" if fused_score < -0.1 else "neutral")
    
    # Layer 5: Emotion classification
    emotions = classify_emotions(message)
    
    # Layer 6: Intent detection
    intent_result = detect_intent_advanced(message)
    
    # Layer 7: Risk adjustment
    risk_result = calculate_risk_adjustment(
        fused_score,
        intent_result["intent"],
        intent_result["confidence"],
        emotions,
        current_risk
    )
    
    # Subjectivity from VADER (ratio of non-neutral)
    subjectivity = round(1.0 - vader["neutral"], 3)
    
    return {
        # Backward-compatible fields
        "sentiment": simple_sentiment,
        "polarity": fused_score,
        "subjectivity": subjectivity,
        "adjustedRisk": risk_result["adjustedRisk"],
        "riskChange": risk_result["riskChange"],
        "explanation": risk_result["explanation"],
        
        # Enhanced fields
        "sentimentCategory": sentiment_category,
        "vaderScore": vader["compound"],
        "domainScore": domain_score,
        "emotions": emotions,
        "emotionList": list(emotions.keys()),
        "intent": intent_result["intent"],
        "intentConfidence": intent_result["confidence"],
        "intentKeywords": intent_result["keywords"],
        "allIntents": intent_result["all_detected"],
        "riskBreakdown": risk_result["breakdown"],
        "analysisEngine": "VADER+Domain" if VADER_AVAILABLE else "TextBlob+Domain",
    }
