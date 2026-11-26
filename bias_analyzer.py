"""
Multi-Provider Bias Analyzer Module

This module provides intelligent bias detection in academic papers using
multiple AI providers (Gemini and OpenRouter) with automatic fallback.
"""

import os
import json
import hashlib
import logging
import time
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from src import gemini_client
from src.gemini_client import GeminiError, is_real_key, parse_json_lenient

logger = logging.getLogger(__name__)


@dataclass
class BiasInstance:
    """Represents a single detected bias instance."""
    type: str
    severity: str  # "low", "moderate", "high"
    excerpt: str
    explanation: str
    suggestion: str
    section: str = ""


@dataclass
class BiasAnalysisResult:
    """Complete bias analysis result for a paper."""
    overall_score: int  # 0-100, lower is better (less biased)
    severity: str  # "low", "moderate", "high"
    summary: str
    biases: List[BiasInstance]
    strengths: List[str]
    error: Optional[str] = None
    provider: Optional[str] = None  # Track which API was used
    status: str = "ok"  # "ok" | "failed" | "disabled"
    model: Optional[str] = None


class BiasAnalysisCache:
    """Simple in-memory cache for bias analysis results."""
    
    def __init__(self, ttl_seconds: int = 3600):
        self._cache: Dict[str, tuple] = {}  # hash -> (result, timestamp)
        self._ttl = ttl_seconds
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for cache key."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[BiasAnalysisResult]:
        """Get cached result if available and not expired."""
        key = self._hash_text(text)
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                logger.info(f"Cache hit for text hash: {key[:8]}...")
                return result
            else:
                del self._cache[key]
        return None
    
    def set(self, text: str, result: BiasAnalysisResult) -> None:
        """Cache analysis result."""
        key = self._hash_text(text)
        self._cache[key] = (result, time.time())
        logger.info(f"Cached result for text hash: {key[:8]}...")


class BiasAnalyzer:
    """
    Analyzes academic papers for various types of biases using multiple AI providers.
    Supports Gemini and OpenRouter with automatic fallback.
    
    Supported bias types:
    - Confirmation Bias: Language assuming conclusions before evidence
    - Selection Bias: Indicators of non-representative sampling
    - Publication Bias: Overly positive framing, suppressing negative results
    - Funding Bias: Undisclosed conflicts of interest
    - Citation Bias: Selective citing supporting predetermined views
    - Methodology Bias: Flawed experimental design
    """
    
    SYSTEM_PROMPT = """You are an expert academic reviewer specializing in detecting biases in research papers. Your task is to analyze academic text for potential biases that could affect the validity or objectivity of the research.

Analyze the provided text for these bias types:

1. **Confirmation Bias**: Language that assumes conclusions before presenting evidence, or interprets results to fit predetermined beliefs. Look for phrases like "clearly proves", "obviously demonstrates", "as expected".

2. **Selection Bias**: Indicators of non-representative sampling, cherry-picked data, or exclusion criteria that might skew results. Look for limited samples, convenience sampling, or unjustified exclusions.

3. **Publication Bias**: Overly positive framing, suppression of negative or null results, exaggerated claims of novelty or importance. Look for "breakthrough", "revolutionary", "first ever" without justification.

4. **Funding Bias**: Potential conflicts of interest, industry-sponsored research without disclosure, or conclusions that suspiciously align with funder interests.

5. **Citation Bias**: Selective citing that only supports the authors' views, ignoring contradictory evidence, or over-reliance on self-citations.

6. **Methodology Bias**: Flawed experimental design, lack of controls, inappropriate statistical methods, or p-hacking indicators.

For each bias found:
- Quote the EXACT excerpt showing the bias (keep it brief, under 50 characters)
- Rate severity as "low", "moderate", or "high"
- Explain clearly WHY it's biased (1 sentence max)
- Provide a concrete suggestion for improvement (1 sentence max)

Also identify STRENGTHS - good practices the paper follows (e.g., "acknowledges limitations", "uses control group"). Keep strength descriptions brief (under 10 words each).

Calculate an overall bias score from 0-100 where:
- 0-25: Low bias (excellent objectivity)
- 26-50: Moderate bias (some concerns)
- 51-75: High bias (significant issues)
- 76-100: Severe bias (major credibility concerns)

IMPORTANT: 
- Respond ONLY with valid JSON matching this exact schema
- Keep all text fields CONCISE to avoid truncation
- Limit to maximum 5 biases and 5 strengths

{
  "biases": [
    {
      "type": "Confirmation Bias|Selection Bias|Publication Bias|Funding Bias|Citation Bias|Methodology Bias",
      "severity": "low|moderate|high",
      "excerpt": "brief quote",
      "explanation": "one sentence why biased",
      "suggestion": "one sentence fix"
    }
  ],
  "overall_score": 0-100,
  "summary": "1-2 sentence summary",
  "strengths": ["brief strength"]
}

If no biases are found, return an empty "biases" array with a low score and appropriate summary."""

    def __init__(self):
        """Initialize the Bias Analyzer with configured AI provider."""
        self.enabled = os.getenv("BIAS_ANALYSIS_ENABLED", "true").lower() == "true"
        self.provider = os.getenv("AI_PROVIDER", "gemini").lower()
        
        # Keys that are still .env.example placeholders count as "not configured"
        self.gemini_key = os.getenv("GEMINI_API_KEY") if is_real_key(os.getenv("GEMINI_API_KEY")) else None
        self.gemini_model = os.getenv("GEMINI_MODEL", gemini_client.DEFAULT_MODEL)

        self.openrouter_key = os.getenv("OPENROUTER_API_KEY") if is_real_key(os.getenv("OPENROUTER_API_KEY")) else None
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")

        # Initialize cache
        self._cache = BiasAnalysisCache(ttl_seconds=3600)

        if self.provider not in ("gemini", "openrouter"):
            logger.warning(f"Unknown AI_PROVIDER '{self.provider}', using gemini")
            self.provider = "gemini"

        if not self.gemini_key and not self.openrouter_key:
            logger.warning("No valid AI provider key configured. Bias analysis will be disabled.")
            self.enabled = False
        else:
            logger.info(
                f"BiasAnalyzer ready (primary={self.provider}, "
                f"gemini={'yes' if self.gemini_key else 'no'}, "
                f"openrouter={'yes' if self.openrouter_key else 'no'})"
            )

    def _build_prompt(self, text: str, section_name: Optional[str] = None) -> str:
        """Build the analysis prompt."""
        section_context = f"\n\nThis text is from the '{section_name}' section of an academic paper." if section_name else ""
        return f"{self.SYSTEM_PROMPT}{section_context}\n\nText to analyze:\n\"\"\"\n{text}\n\"\"\""

    @staticmethod
    def _validate(data: Any) -> Dict[str, Any]:
        """Check the model's JSON has the shape the frontend needs. Raises ValueError."""
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        if "overall_score" not in data and "biases" not in data:
            raise ValueError("JSON has neither 'overall_score' nor 'biases'")

        biases = []
        raw_biases = data.get("biases") or []
        if not isinstance(raw_biases, list):
            raise ValueError("'biases' is not a list")
        for b in raw_biases:
            if not isinstance(b, dict):
                continue
            severity = str(b.get("severity", "moderate")).strip().lower()
            if severity in ("medium", "mid"):
                severity = "moderate"
            if severity in ("severe", "critical"):
                severity = "high"
            if severity not in ("low", "moderate", "high"):
                severity = "moderate"
            biases.append({
                "type": str(b.get("type") or "Unknown"),
                "severity": severity,
                "excerpt": str(b.get("excerpt") or ""),
                "explanation": str(b.get("explanation") or ""),
                "suggestion": str(b.get("suggestion") or ""),
            })

        try:
            score = int(round(float(data.get("overall_score"))))
        except (TypeError, ValueError):
            raise ValueError("'overall_score' is missing or not a number")
        score = max(0, min(100, score))

        strengths = data.get("strengths") or []
        if not isinstance(strengths, list):
            strengths = []
        strengths = [str(x) for x in strengths if isinstance(x, (str, int, float)) and str(x).strip()]

        return {
            "biases": biases,
            "overall_score": score,
            "summary": str(data.get("summary") or "Analysis completed."),
            "strengths": strengths,
        }

    def _call_gemini(self, prompt: str) -> tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
        """
        Call Gemini (model fallback, timeout and retry are handled by gemini_client).
        Returns: (validated_data, model_used, error_message)
        """
        if not self.gemini_key:
            return None, None, "Gemini API key not configured"
        try:
            data, model = gemini_client.generate_json(prompt, max_output_tokens=8192, temperature=0.3)
            return self._validate(data), model, None
        except GeminiError as e:
            logger.warning(f"Gemini bias analysis failed: {e.message}")
            return None, None, e.message
        except ValueError as e:
            logger.warning(f"Gemini returned an unusable bias analysis: {e}")
            return None, None, f"Gemini returned an unusable response ({e})"
        except Exception as e:  # never let the provider crash the request
            logger.exception("Unexpected Gemini error")
            return None, None, str(e)[:200]

    def _call_openrouter(self, prompt: str) -> tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
        """
        Call OpenRouter API.
        Returns: (validated_data, model_used, error_message)
        """
        if not self.openrouter_key:
            return None, None, "OpenRouter API key not configured"

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "HTTP-Referer": "https://github.com/rk0802p/paper_bias_detection",
                    "X-Title": "Paper Bias Detection"
                },
                json={
                    "model": self.openrouter_model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 8192
                },
                timeout=(5, 60)
            )

            if response.status_code != 200:
                error_msg = f"OpenRouter API error {response.status_code}"
                logger.warning(f"{error_msg}: {response.text[:200]}")
                return None, None, error_msg

            body = response.json()
            content = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
            if not content:
                return None, None, "Empty response from OpenRouter"
            return self._validate(parse_json_lenient(content)), self.openrouter_model, None

        except ValueError as e:
            logger.warning(f"OpenRouter returned an unusable bias analysis: {e}")
            return None, None, f"OpenRouter returned an unusable response ({e})"
        except requests.exceptions.Timeout:
            return None, None, "OpenRouter request timed out"
        except Exception as e:
            logger.warning(f"OpenRouter API failed: {e}")
            return None, None, str(e)[:200]

    def _determine_severity(self, score: int) -> str:
        """Determine overall severity from score."""
        if score <= 25:
            return "low"
        elif score <= 50:
            return "moderate"
        else:
            return "high"

    def analyze_text(self, text: str, section_name: Optional[str] = None, use_cache: bool = True) -> BiasAnalysisResult:
        """
        Analyze text for biases using configured AI provider with automatic fallback.
        
        Args:
            text: The academic text to analyze
            section_name: Optional name of the section (e.g., "Abstract", "Methodology")
            use_cache: Whether to use cached results
            
        Returns:
            BiasAnalysisResult with detected biases and analysis
        """
        if not self.enabled:
            return BiasAnalysisResult(
                overall_score=0,
                severity="low",
                summary="Bias analysis is disabled. Set GEMINI_API_KEY in backend/.env and BIAS_ANALYSIS_ENABLED=true.",
                biases=[],
                strengths=[],
                error="Bias analysis disabled",
                provider=None,
                status="disabled"
            )
        
        # Check cache
        original_text = text
        if use_cache:
            cached = self._cache.get(text)
            if cached:
                return cached
        
        # Truncate very long texts
        max_chars = 30000
        if len(text) > max_chars:
            logger.warning(f"Text truncated from {len(text)} to {max_chars} characters")
            text = text[:max_chars] + "\n\n[Text truncated for analysis]"
        
        prompt = self._build_prompt(text, section_name)

        callers = {"gemini": self._call_gemini, "openrouter": self._call_openrouter}
        available = {"gemini": bool(self.gemini_key), "openrouter": bool(self.openrouter_key)}
        order = [self.provider] + [p for p in ("gemini", "openrouter") if p != self.provider]

        data, model_used, provider_used = None, None, None
        errors = []
        for name in order:
            if not available[name]:
                continue
            data, model_used, err = callers[name](prompt)
            if data is not None:
                provider_used = name
                break
            errors.append(f"{name}: {err}")
            logger.info(f"Bias provider '{name}' failed, trying next if available")

        # Handle complete failure - reported as a failure, never as a result
        if data is None:
            error_msg = "; ".join(errors) or "No AI provider configured"
            return BiasAnalysisResult(
                overall_score=0,
                severity="low",
                summary=f"Bias analysis could not be completed: {error_msg}",
                biases=[],
                strengths=[],
                error=error_msg,
                provider=None,
                status="failed",
            )

        biases = [
            BiasInstance(section=section_name or "", **b)
            for b in data["biases"]
        ]

        result = BiasAnalysisResult(
            overall_score=data["overall_score"],
            severity=self._determine_severity(data["overall_score"]),
            summary=data["summary"],
            biases=biases,
            strengths=data["strengths"],
            error=None,
            provider=provider_used,
            status="ok",
            model=model_used,
        )

        # Cache the result (keyed on the original, untruncated text)
        if use_cache:
            self._cache.set(original_text, result)

        logger.info(f"Bias analysis complete using {provider_used} ({model_used})")
        return result

    def analyze_paper_sections(self, sections: Dict[str, str]) -> BiasAnalysisResult:
        """
        Analyze multiple sections of a paper and combine results.
        
        Args:
            sections: Dict mapping section names to their text content
            
        Returns:
            Combined BiasAnalysisResult for the entire paper
        """
        if not self.enabled:
            return BiasAnalysisResult(
                overall_score=0,
                severity="low",
                summary="Bias analysis is disabled.",
                biases=[],
                strengths=[],
                error="Bias analysis disabled",
                provider=None,
                status="disabled"
            )
        
        # Combine all sections into one analysis for efficiency
        combined_text = ""
        for section_name, text in sections.items():
            if text and text.strip():
                combined_text += f"\n\n=== {section_name.upper()} ===\n{text}"
        
        if not combined_text.strip():
            return BiasAnalysisResult(
                overall_score=0,
                severity="low",
                summary="No text provided for analysis.",
                biases=[],
                strengths=[],
                error="Empty input",
                provider=None,
                status="failed"
            )
        
        return self.analyze_text(combined_text, section_name="Full Paper")


def result_to_dict(result: BiasAnalysisResult) -> Dict[str, Any]:
    """Convert BiasAnalysisResult to a JSON-serializable dictionary."""
    return {
        "overall_score": result.overall_score,
        "severity": result.severity,
        "summary": result.summary,
        "biases": [asdict(b) for b in result.biases],
        "strengths": result.strengths,
        "error": result.error,
        "provider": result.provider,
        "status": result.status,
        "model": result.model
    }


# Backward compatibility
GeminiBiasAnalyzer = BiasAnalyzer


# For testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    analyzer = BiasAnalyzer()
    
    test_text = """
    Our groundbreaking study clearly proves that our novel approach significantly 
    outperforms all existing methods. The results obviously demonstrate the 
    superiority of our technique. We selected participants who were likely to 
    respond well to our treatment. As expected, the outcomes confirmed our 
    hypothesis that this revolutionary method is the best solution available.
    The p-value was p=0.049, showing statistical significance.
    """
    
    print("Analyzing test text...")
    result = analyzer.analyze_text(test_text, section_name="Results")
    
    print(f"\nProvider Used: {result.provider}")
    print(f"Overall Score: {result.overall_score}/100")
    print(f"Severity: {result.severity}")
    print(f"Summary: {result.summary}")
    print(f"\nBiases Found: {len(result.biases)}")
    for bias in result.biases:
        print(f"  - {bias.type} ({bias.severity}): {bias.excerpt[:50]}...")
    print(f"\nStrengths: {result.strengths}")
