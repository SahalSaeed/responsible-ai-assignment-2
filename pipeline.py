"""
pipeline.py — Production Guardrail Pipeline
Assignment 2: Responsible & Explainable AI
FAST-NUCES

Three-layer content moderation pipeline:
  Layer 1: Regex-based pre-filter (fast, rule-based)
  Layer 2: Calibrated DistilBERT model (probabilistic)
  Layer 3: Human review queue (uncertainty escalation)
"""

import re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Optional


# =============================================================================
# LAYER 1: Regex Blocklist
# =============================================================================

BLOCKLIST = {
    "direct_threat": [
        # Pattern 1: "I will/gonna/going to [kill|murder|shoot|stab|hurt] you"
        re.compile(
            r"\b(i|we)\s+(will|gonna|am going to|are going to|'ll|shall)\s+"
            r"(?:kill|murder|shoot|stab|hurt|attack|destroy|end)\s+(you|u|yr|your|them|him|her)\b",
            re.IGNORECASE
        ),
        # Pattern 2: "you're going to die / you will die"
        re.compile(
            r"\b(you|u|you're|ur)\s+(are\s+going\s+to|will|gonna)\s+(die|get\s+hurt|get\s+killed|be\s+killed)\b",
            re.IGNORECASE
        ),
        # Pattern 3: "I'll find where you live"
        re.compile(
            r"\bi'?ll?\s+(find|track|locate|come\s+to)\s+(where\s+you\s+live|your\s+house|your\s+address|you)\b",
            re.IGNORECASE
        ),
        # Pattern 4: "someone should [kill|shoot|murder] you" — capturing group for threat verb
        re.compile(
            r"\b(someone|everybody|anyone|they)\s+should\s+(?P<threat_verb>kill|murder|shoot|stab|hurt|beat)\s+(you|him|her|them|u)\b",
            re.IGNORECASE
        ),
        # Pattern 5: "dead [name/you] walking" or "you are dead"
        re.compile(
            r"\b(you\s+are|you're|ur)\s+(dead|as\s+good\s+as\s+dead|going\s+to\s+die)\b",
            re.IGNORECASE
        ),
        # Pattern 6: threats with weapon nouns
        re.compile(
            r"\b(gun|knife|weapon|bomb)\s+(pointed\s+at|aimed\s+at|for)\s+(you|your\s+head|u)\b",
            re.IGNORECASE
        ),
    ],

    "self_harm_directed": [
        # Pattern 1: "you should kill yourself"
        re.compile(
            r"\byou\s+(should|need\s+to|ought\s+to|must)\s+kill\s+yourself\b",
            re.IGNORECASE
        ),
        # Pattern 2: "go kill yourself / go hang yourself"
        re.compile(
            r"\bgo\s+(kill|hang|hurt|harm|shoot|stab)\s+yourself\b",
            re.IGNORECASE
        ),
        # Pattern 3: "nobody would miss you if you died / disappeared"
        re.compile(
            r"\bnobody\s+(would|will|cares|'d)\s+(miss|care\s+about|mourn)\s+(you|u)\b",
            re.IGNORECASE
        ),
        # Pattern 4: "do everyone a favour and disappear/die"
        re.compile(
            r"\bdo\s+(everyone|the\s+world|us\s+all|us)\s+a\s+favo(?:u)?r\s+and\s+(disappear|die|end\s+it|kill\s+yourself)\b",
            re.IGNORECASE
        ),
        # Pattern 5: directed self-harm instruction
        re.compile(
            r"\b(just|please|why\s+don'?t\s+you)\s+(die|kill\s+yourself|end\s+it\s+all|disappear)\b",
            re.IGNORECASE
        ),
    ],

    "doxxing_stalking": [
        # Pattern 1: "I know where you live"
        re.compile(
            r"\bi\s+(know|found|have)\s+(where\s+you\s+live|your\s+address|where\s+you\s+are)\b",
            re.IGNORECASE
        ),
        # Pattern 2: "I'll post/share your address/number/info"
        re.compile(
            r"\bi'?(?:ll|will|am\s+going\s+to|'m\s+gonna)\s+(post|share|publish|leak|send)\s+(your\s+(?:address|phone|number|info|photos|nudes|location))\b",
            re.IGNORECASE
        ),
        # Pattern 3: "I found your real name / I know your real name"
        re.compile(
            r"\bi\s+(found|know|have)\s+(your\s+real\s+name|who\s+you\s+really\s+are|your\s+identity)\b",
            re.IGNORECASE
        ),
        # Pattern 4: "everyone will know who you really are"
        re.compile(
            r"\b(everyone|the\s+whole\s+internet|people)\s+will\s+know\s+(who\s+you\s+(really\s+)?are|your\s+real\s+identity)\b",
            re.IGNORECASE
        ),
        # Pattern 5: "I'm watching you / I've been following you"
        re.compile(
            r"\bi'?(?:ve\s+been|m|'m\s+always)\s+(watching|following|tracking|monitoring)\s+(you|your\s+every\s+move)\b",
            re.IGNORECASE
        ),
    ],

    "dehumanization": [
        # Pattern 1: "[group] are not human / are animals / are subhuman"
        re.compile(
            r"\b\w+\s+are\s+(?:not\s+)?(?:human|people|persons?)\b.{0,30}(?:animal|beast|subhuman|vermin|parasite)",
            re.IGNORECASE
        ),
        # Pattern 2: "[group] should be exterminated/eliminated/eradicated"
        re.compile(
            r"\b\w[\w\s]{1,30}\s+should\s+be\s+(?:exterminated|eliminated|eradicated|wiped\s+out|cleansed)\b",
            re.IGNORECASE
        ),
        # Pattern 3: "[group] are a disease/plague/cancer/infestation"
        re.compile(
            r"\b\w[\w\s]{1,30}\s+are\s+(?:a\s+)?(?:disease|plague|cancer|infestation|infection|virus|parasite|pest)\b",
            re.IGNORECASE
        ),
        # Pattern 4: non-human comparisons for groups
        re.compile(
            r"\b(?:those\s+)?(?:people|group|community|race|nation)\s+(?:are|is)\s+(?:like\s+)?(?:animals?|beasts?|monkeys?|rats?|roaches?|vermin)\b",
            re.IGNORECASE
        ),
        # Pattern 5: explicit dehumanization phrase
        re.compile(
            r"\b(?:not\s+(?:human|people|real\s+people)|less\s+than\s+human|not\s+even\s+(?:human|people))\b",
            re.IGNORECASE
        ),
    ],

    "coordinated_harassment": [
        # Pattern 1: "everyone report [username/account/this]"
        re.compile(
            r"\beveryone\s+(?:go\s+)?report\s+(?:@?\w+|this\s+(?:account|user|person|profile))\b",
            re.IGNORECASE
        ),
        # Pattern 2: "let's all go after / attack [target]"
        re.compile(
            r"\blet'?s\s+(?:all\s+)?(?:go\s+after|attack|target|spam|flood|raid|bombard)\s+(?:@?\w+|their\s+\w+)\b",
            re.IGNORECASE
        ),
        # Pattern 3: "raid their profile/channel/server" — lookahead version
        re.compile(
            r"\b(?:raid|mass\s+report|brigade|dogpile)\b(?=.{0,50}(?:profile|channel|server|account|page|post|tweet))",
            re.IGNORECASE
        ),
        # Pattern 4: "mass report this account"
        re.compile(
            r"\bmass\s+(?:report|flag|block|dm)\s+(?:this\s+)?(?:account|user|post|page)\b",
            re.IGNORECASE
        ),
    ]
}


def input_filter(text: str) -> Optional[dict]:
    """
    Layer 1: Regex pre-filter.
    Returns a block decision dict if a pattern matches, else None.
    The dict includes which category triggered the match for auditability.
    """
    for category, patterns in BLOCKLIST.items():
        for pattern in patterns:
            if pattern.search(text):
                return {
                    "decision": "block",
                    "layer": "input_filter",
                    "category": category,
                    "confidence": 1.0
                }
    return None


# =============================================================================
# LAYER 2: Calibrated DistilBERT Model
# =============================================================================

class DistilBERTWrapper(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for DistilBERT to use with CalibratedClassifierCV.
    """
    def __init__(self, model_path: str, max_length: int = 128):
        self.model_path = model_path
        self.max_length = max_length
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classes_ = np.array([0, 1])
        self._loaded = False

    def _load_model(self):
        if not self._loaded:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()
            self.model = self.model.to(self.device)
            self._loaded = True

    def _get_probs(self, texts):
        self._load_model()
        all_probs = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(
                batch, max_length=self.max_length,
                truncation=True, padding=True, return_tensors='pt'
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
        return np.array(all_probs)

    def fit(self, X, y):
        # X is assumed to be a list/array of texts
        return self

    def predict_proba(self, X):
        if hasattr(X, 'tolist'):
            X = X.tolist()
        probs = self._get_probs(X)
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


class ModerationPipeline:
    """
    Three-layer production content moderation pipeline.

    Layer 1: Fast regex pre-filter
    Layer 2: Calibrated DistilBERT classifier
    Layer 3: Human review queue (uncertainty escalation)

    Usage:
        pipeline = ModerationPipeline(model_path='./model_checkpoint_part1')
        pipeline.calibrate(texts, labels)  # fit calibration
        result = pipeline.predict("some comment text")
    """

    def __init__(
        self,
        model_path: str = './model_checkpoint_part1',
        block_threshold: float = 0.6,
        allow_threshold: float = 0.4,
    ):
        """
        Args:
            model_path: Path to the fine-tuned DistilBERT checkpoint
            block_threshold: Calibrated probability >= this -> block
            allow_threshold: Calibrated probability <= this -> allow
            Anything in between -> escalate to human review
        """
        self.model_path = model_path
        self.block_threshold = block_threshold
        self.allow_threshold = allow_threshold
        self.base_model = DistilBERTWrapper(model_path)
        self.calibrated_model = None
        self._calibrated = False

    def calibrate(self, texts, labels, cv: int = 3):
        """
        Fit isotonic calibration on a held-out set.

        Args:
            texts: List of comment strings (calibration set)
            labels: Corresponding binary labels (0/1)
            cv: Number of CV folds for CalibratedClassifierCV
        """
        print(f"Fitting calibration on {len(texts)} samples (cv={cv})...")
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.base_model,
            method='isotonic',
            cv=cv
        )
        self.calibrated_model.fit(np.array(texts), np.array(labels))
        self._calibrated = True
        print("Calibration complete.")

    def _model_predict_single(self, text: str) -> float:
        """Get calibrated probability for a single text."""
        if not self._calibrated:
            # Fall back to uncalibrated model
            proba = self.base_model.predict_proba([text])[0]
        else:
            proba = self.calibrated_model.predict_proba([text])[0]
        return float(proba[1])  # probability of toxic (class 1)

    def predict(self, text: str) -> dict:
        """
        Run the three-layer pipeline on a single text.

        Returns a dict with keys:
            - decision: 'block' | 'allow' | 'review'
            - layer: which layer made the decision
            - confidence: probability or 1.0 for regex matches
            - category: (Layer 1 only) which blocklist category matched
        """
        if not isinstance(text, str):
            text = str(text)

        # ---- Layer 1: Input filter ----
        filter_result = input_filter(text)
        if filter_result is not None:
            return filter_result

        # ---- Layer 2: Calibrated model ----
        try:
            confidence = self._model_predict_single(text)
        except Exception as e:
            # If model fails, escalate to review
            return {
                "decision": "review",
                "layer": "model_error",
                "confidence": -1.0,
                "error": str(e)
            }

        if confidence >= self.block_threshold:
            return {
                "decision": "block",
                "layer": "model",
                "confidence": confidence
            }
        elif confidence <= self.allow_threshold:
            return {
                "decision": "allow",
                "layer": "model",
                "confidence": confidence
            }

        # ---- Layer 3: Human review queue ----
        return {
            "decision": "review",
            "layer": "model",
            "confidence": confidence
        }

    def predict_batch(self, texts: list) -> list:
        """Run pipeline on a list of texts. Returns list of decision dicts."""
        return [self.predict(t) for t in texts]


# =============================================================================
# Convenience function for standalone testing
# =============================================================================

def load_pipeline(model_path: str = './model_checkpoint_part1') -> ModerationPipeline:
    """Load a pipeline instance. Call .calibrate() before use."""
    return ModerationPipeline(model_path=model_path)


if __name__ == '__main__':
    # Quick smoke test of the regex layer (no model needed)
    test_cases = [
        ("I will kill you tonight", "direct_threat"),
        ("go kill yourself loser", "self_harm_directed"),
        ("I know where you live and I'll post your address", "doxxing_stalking"),
        ("Those people are not human, they are animals", "dehumanization"),
        ("everyone report this account now", "coordinated_harassment"),
        ("I really enjoyed this movie", None),
        ("The weather is nice today", None),
    ]

    print("Layer 1 Regex Filter Smoke Test")
    print("=" * 60)
    for text, expected in test_cases:
        result = input_filter(text)
        decision = result['category'] if result else 'pass'
        status = "✓" if (result is None) == (expected is None) else "✗"
        print(f"{status} [{decision:25s}] {text[:55]}")
