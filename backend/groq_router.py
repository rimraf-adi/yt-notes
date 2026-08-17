import os
import time
import logging
import threading
from typing import List, Dict, Any, Optional, Iterator, Callable, Union
from groq import Groq, RateLimitError, APIError, InternalServerError
from backend.config import GROQ_KEYS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GroqRouter")

# Model Definitions by Tier for Max Throughput
TIER_HEAVY = [
    "llama-3.3-70b-versatile",
    "deepseek-r1-distill-llama-70b",
    "llama-3.1-70b-versatile"
]

TIER_FAST = [
    "llama-3.1-8b-instant",
    "llama-3.2-3b-preview",
    "llama-3.2-1b-preview"
]

TIER_AUDIO = [
    "whisper-large-v3-turbo",
    "whisper-large-v3"
]

class KeyModelSlot:
    """
    Tracks state and metrics for a specific (Key, Model) pair.
    """
    def __init__(self, key_idx: int, api_key: str, model: str):
        self.key_idx = key_idx
        self.api_key = api_key
        self.model = model
        self.client = Groq(api_key=api_key)
        
        self.cooldown_until = 0.0
        self.inflight_requests = 0
        self.total_completions = 0
        self.total_transcriptions = 0
        self.total_errors = 0
        self.total_tokens_est = 0
        self.last_used_timestamp = 0.0

    @property
    def is_available(self) -> bool:
        return time.time() >= self.cooldown_until

    def mark_error(self, cooldown_duration: float = 60.0):
        self.cooldown_until = time.time() + cooldown_duration
        self.total_errors += 1
        logger.warning(
            f"⚠️ Cooldown triggered: Key #{self.key_idx + 1} | Model: {self.model} "
            f"for {cooldown_duration}s"
        )

    def mark_success(self, tokens_used: int = 0):
        self.last_used_timestamp = time.time()
        self.total_completions += 1
        self.total_tokens_est += tokens_used

class GroqKeyModelRouter:
    """
    High-Throughput, Matrix-Based Key and Model Router for Groq.
    
    Optimizes for:
    - Maximum parallel tokens / second across all 8 keys.
    - Independent rate-limit tracking per (Key, Model) combination.
    - Automatic fallbacks across keys and model tiers.
    - Zero-downtime round-robin and least-loaded routing.
    """
    def __init__(self, keys: Optional[List[str]] = None):
        self.keys = keys if keys is not None else GROQ_KEYS
        self.lock = threading.Lock()
        
        if not self.keys:
            logger.warning("No Groq keys provided to GroqKeyModelRouter!")

        # Build (Key, Model) Slot Matrix
        self.slots: Dict[str, List[KeyModelSlot]] = {}
        all_models = TIER_HEAVY + TIER_FAST + TIER_AUDIO
        
        for model in all_models:
            self.slots[model] = [
                KeyModelSlot(key_idx=i, api_key=k, model=model)
                for i, k in enumerate(self.keys)
            ]

        self.round_robin_pointers: Dict[str, int] = {m: 0 for m in all_models}

    def _select_best_slot(self, model: str) -> KeyModelSlot:
        """
        Selects the best available key for a given model:
        1. Filters out keys in cooldown.
        2. Picks the least-loaded (lowest inflight requests), using round-robin as tie-breaker.
        """
        with self.lock:
            if model not in self.slots:
                # Dynamically register model if new
                self.slots[model] = [
                    KeyModelSlot(key_idx=i, api_key=k, model=model)
                    for i, k in enumerate(self.keys)
                ]
                self.round_robin_pointers[model] = 0

            available_slots = [s for s in self.slots[model] if s.is_available]

            if not available_slots:
                # If all slots in cooldown, pick the one that expires soonest
                earliest_slot = min(self.slots[model], key=lambda s: s.cooldown_until)
                wait_sec = max(0.0, earliest_slot.cooldown_until - time.time())
                logger.info(f"All keys for {model} are busy. Waiting {wait_sec:.2f}s for Key #{earliest_slot.key_idx + 1}")
                if wait_sec > 0:
                    time.sleep(min(wait_sec, 4.0))
                return earliest_slot

            # Least-loaded key selection with round-robin priority
            start_idx = self.round_robin_pointers[model]
            n = len(available_slots)
            
            # Sort by inflight requests first, then round-robin offset
            available_slots.sort(key=lambda s: (s.inflight_requests, (s.key_idx - start_idx) % len(self.keys)))
            selected = available_slots[0]
            
            self.round_robin_pointers[model] = (selected.key_idx + 1) % len(self.keys)
            selected.inflight_requests += 1
            return selected

    def route_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        tier: str = "heavy", # 'heavy', 'fast', or 'auto'
        temperature: float = 0.3,
        max_tokens: int = 4096,
        max_retries: int = 8
    ) -> str:
        """
        Executes a chat completion with maximum throughput routing.
        Automatically cascades across model tiers and 8 keys on rate limits.
        """
        # Determine model cascade candidate list
        if model:
            candidate_models = [model]
            if tier == "heavy":
                candidate_models += [m for m in TIER_HEAVY if m != model] + TIER_FAST
            elif tier == "fast":
                candidate_models += [m for m in TIER_FAST if m != model]
            else:
                candidate_models += [m for m in (TIER_HEAVY + TIER_FAST) if m != model]
        else:
            candidate_models = TIER_HEAVY + TIER_FAST if tier != "fast" else TIER_FAST + TIER_HEAVY

        last_error = None
        attempt_count = 0

        for current_model in candidate_models:
            for _ in range(len(self.keys)):
                if attempt_count >= max_retries:
                    break
                attempt_count += 1

                slot = self._select_best_slot(current_model)
                try:
                    logger.info(f"⚡ [Throughput Router] Key #{slot.key_idx + 1} -> Model: {current_model}")
                    response = slot.client.chat.completions.create(
                        model=current_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    usage = getattr(response, "usage", None)
                    tokens = usage.total_tokens if usage else 1000
                    
                    with self.lock:
                        slot.inflight_requests = max(0, slot.inflight_requests - 1)
                        slot.mark_success(tokens)
                    
                    return response.choices[0].message.content or ""

                except RateLimitError as e:
                    with self.lock:
                        slot.inflight_requests = max(0, slot.inflight_requests - 1)
                        slot.mark_error(cooldown_duration=45.0)
                    last_error = e
                except (APIError, InternalServerError) as e:
                    with self.lock:
                        slot.inflight_requests = max(0, slot.inflight_requests - 1)
                        slot.mark_error(cooldown_duration=20.0)
                    last_error = e
                except Exception as e:
                    with self.lock:
                        slot.inflight_requests = max(0, slot.inflight_requests - 1)
                    logger.error(f"Error on Key #{slot.key_idx + 1} ({current_model}): {e}")
                    last_error = e

        raise RuntimeError(f"Chat failed across all {len(self.keys)} keys & model cascades: {last_error}")

    def route_chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        tier: str = "heavy",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        max_retries: int = 8
    ) -> Iterator[str]:
        """
        Streams chat tokens with key-model rotation and failover.
        """
        candidate_models = [model] if model else (TIER_HEAVY + TIER_FAST if tier != "fast" else TIER_FAST)
        last_error = None
        attempt_count = 0

        for current_model in candidate_models:
            for _ in range(len(self.keys)):
                if attempt_count >= max_retries:
                    break
                attempt_count += 1

                slot = self._select_best_slot(current_model)
                try:
                    response = slot.client.chat.completions.create(
                        model=current_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True
                    )
                    
                    with self.lock:
                        slot.inflight_requests = max(0, slot.inflight_requests - 1)
                        slot.mark_success()

                    for chunk in response:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content
                    return

                except RateLimitError as e:
                    with self.lock:
                        slot.inflight_requests = max(0, slot.inflight_requests - 1)
                        slot.mark_error(cooldown_duration=45.0)
                    last_error = e
                except Exception as e:
                    with self.lock:
                        slot.inflight_requests = max(0, slot.inflight_requests - 1)
                    last_error = e

        raise RuntimeError(f"Streaming failed across all keys: {last_error}")

    def route_transcription(
        self,
        audio_file_path: str,
        model: str = "whisper-large-v3-turbo",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        max_retries: int = 8
    ) -> Dict[str, Any]:
        """
        Routes Whisper transcription across keys and turbo/standard whisper models.
        """
        candidate_models = [model] + [m for m in TIER_AUDIO if m != model]
        last_error = None

        for current_model in candidate_models:
            for _ in range(len(self.keys)):
                slot = self._select_best_slot(current_model)
                try:
                    logger.info(f"🎙️ [Whisper Router] Key #{slot.key_idx + 1} -> Model: {current_model}")
                    with open(audio_file_path, "rb") as f:
                        kwargs = {
                            "file": (audio_file_path, f),
                            "model": current_model,
                            "response_format": "verbose_json",
                            "temperature": 0.0
                        }
                        if language:
                            kwargs["language"] = language
                        if prompt:
                            kwargs["prompt"] = prompt

                        res = slot.client.audio.transcriptions.create(**kwargs)
                        
                        with self.lock:
                            slot.inflight_requests = max(0, slot.inflight_requests - 1)
                            slot.total_transcriptions += 1
                        
                        if hasattr(res, "model_dump"):
                            return res.model_dump()
                        elif hasattr(res, "to_dict"):
                            return res.to_dict()
                        elif isinstance(res, dict):
                            return res
                        else:
                            return {
                                "text": getattr(res, "text", ""),
                                "segments": getattr(res, "segments", [])
                            }

                except RateLimitError as e:
                    with self.lock:
                        slot.inflight_requests = max(0, slot.inflight_requests - 1)
                        slot.mark_error(cooldown_duration=45.0)
                    last_error = e
                except Exception as e:
                    with self.lock:
                        slot.inflight_requests = max(0, slot.inflight_requests - 1)
                    last_error = e

        raise RuntimeError(f"Transcription failed across all keys: {last_error}")

    def get_router_matrix_stats(self) -> Dict[str, Any]:
        """
        Returns full telemetry across all 8 keys and model matrices.
        """
        now = time.time()
        with self.lock:
            key_summaries = []
            for i, key in enumerate(self.keys):
                masked = f"{key[:6]}...{key[-4:]}" if len(key) > 10 else f"Key #{i+1}"
                
                # Aggregate stats for this key across all models
                completions = sum(self.slots[m][i].total_completions for m in self.slots)
                transcriptions = sum(self.slots[m][i].total_transcriptions for m in self.slots)
                errors = sum(self.slots[m][i].total_errors for m in self.slots)
                inflight = sum(self.slots[m][i].inflight_requests for m in self.slots)
                tokens = sum(self.slots[m][i].total_tokens_est for m in self.slots)
                
                # Check active status
                cooldowns = [max(0, int(self.slots[m][i].cooldown_until - now)) for m in self.slots]
                max_cd = max(cooldowns)
                status = "Active (High Throughput)" if max_cd == 0 else f"Partial Cooldown ({max_cd}s)"

                key_summaries.append({
                    "key_index": i + 1,
                    "masked_key": masked,
                    "status": status,
                    "inflight_requests": inflight,
                    "completions": completions,
                    "transcriptions": transcriptions,
                    "errors": errors,
                    "estimated_tokens": tokens
                })

            return {
                "total_keys": len(self.keys),
                "active_keys": sum(1 for k in key_summaries if "Active" in k["status"]),
                "supported_models": list(self.slots.keys()),
                "keys": key_summaries
            }

# Global Router Singleton
groq_router = GroqKeyModelRouter()
