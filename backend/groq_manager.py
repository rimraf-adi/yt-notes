import time
import threading
import logging
from typing import List, Dict, Any, Optional, Iterator
from groq import Groq, RateLimitError, APIError, InternalServerError
from backend.config import GROQ_KEYS, WHISPER_MODEL, LLM_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroqKeyManager:
    """
    Manages rotation, rate-limit tracking, cooldowns, and automatic failover
    across 8 Groq API keys for both Whisper and LLM inference.
    """
    def __init__(self, keys: Optional[List[str]] = None):
        self.keys = keys if keys is not None else GROQ_KEYS
        if not self.keys:
            logger.warning("No Groq API keys found in environment! Please add GROQ_API_KEYS to .env")
        
        self.clients = [Groq(api_key=k) for k in self.keys]
        self.lock = threading.Lock()
        self.current_idx = 0
        # Track cooldown timestamp per key
        self.cooldowns: Dict[int, float] = {i: 0.0 for i in range(len(self.keys))}
        # Track usage stats per key
        self.usage_stats: Dict[int, Dict[str, int]] = {
            i: {"transcriptions": 0, "completions": 0, "errors": 0} 
            for i in range(len(self.keys))
        }

    def total_keys(self) -> int:
        return len(self.keys)

    def _get_next_available_key_index(self) -> int:
        """Returns the next key index that is not in cooldown, with round-robin priority."""
        with self.lock:
            if not self.keys:
                raise ValueError("No Groq API keys configured.")

            now = time.time()
            n = len(self.keys)
            
            # First attempt: find key not in cooldown starting from current_idx
            for offset in range(n):
                idx = (self.current_idx + offset) % n
                if self.cooldowns[idx] <= now:
                    self.current_idx = (idx + 1) % n
                    return idx

            # If all are in cooldown, pick the one with earliest cooldown expiry
            earliest_idx = min(self.cooldowns, key=self.cooldowns.get)
            sleep_needed = max(0.0, self.cooldowns[earliest_idx] - now)
            if sleep_needed > 0:
                logger.info(f"All Groq keys are rate-limited. Waiting {sleep_needed:.2f}s for key #{earliest_idx}...")
                time.sleep(min(sleep_needed, 5.0)) # Wait at most 5s then attempt
            self.current_idx = (earliest_idx + 1) % n
            return earliest_idx

    def _mark_key_rate_limited(self, idx: int, duration_sec: float = 60.0):
        with self.lock:
            self.cooldowns[idx] = time.time() + duration_sec
            self.usage_stats[idx]["errors"] += 1
            masked_key = f"{self.keys[idx][:6]}...{self.keys[idx][-4:]}" if len(self.keys[idx]) > 10 else f"Key #{idx}"
            logger.warning(f"Rate limit on Groq key #{idx} ({masked_key}). Cooldown for {duration_sec}s.")

    def get_stats(self) -> List[Dict[str, Any]]:
        now = time.time()
        with self.lock:
            stats = []
            for i, key in enumerate(self.keys):
                masked = f"{key[:6]}...{key[-4:]}" if len(key) > 10 else f"Key {i+1}"
                is_active = self.cooldowns[i] <= now
                stats.append({
                    "key_index": i + 1,
                    "masked_key": masked,
                    "status": "Active" if is_active else f"Cooldown ({max(0, int(self.cooldowns[i] - now))}s)",
                    "transcriptions": self.usage_stats[i]["transcriptions"],
                    "completions": self.usage_stats[i]["completions"],
                    "errors": self.usage_stats[i]["errors"]
                })
            return stats

    def transcribe_audio(
        self, 
        audio_file_path: str, 
        model: str = WHISPER_MODEL, 
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        max_retries: int = 5
    ) -> Dict[str, Any]:
        """
        Transcribes an audio chunk using Whisper Large on Groq with rotation & retry.
        Returns verbose JSON containing segments with timestamps.
        """
        last_exception = None
        tried_indices = set()

        for attempt in range(max_retries):
            idx = self._get_next_available_key_index()
            tried_indices.add(idx)
            client = self.clients[idx]

            try:
                with open(audio_file_path, "rb") as file:
                    kwargs = {
                        "file": (audio_file_path, file),
                        "model": model,
                        "response_format": "verbose_json",
                        "temperature": 0.0
                    }
                    if language:
                        kwargs["language"] = language
                    if prompt:
                        kwargs["prompt"] = prompt

                    logger.info(f"Transcribing {audio_file_path} using Groq Key #{idx + 1} ({model})...")
                    transcription = client.audio.transcriptions.create(**kwargs)
                    
                    with self.lock:
                        self.usage_stats[idx]["transcriptions"] += 1
                    
                    # Convert response to dictionary if needed
                    if hasattr(transcription, "model_dump"):
                        return transcription.model_dump()
                    elif hasattr(transcription, "to_dict"):
                        return transcription.to_dict()
                    elif isinstance(transcription, dict):
                        return transcription
                    else:
                        # Fallback parsing
                        return {
                            "text": getattr(transcription, "text", ""),
                            "segments": getattr(transcription, "segments", [])
                        }

            except RateLimitError as e:
                logger.warning(f"RateLimitError on key #{idx + 1}: {e}")
                self._mark_key_rate_limited(idx, duration_sec=60.0)
                last_exception = e
            except (APIError, InternalServerError) as e:
                logger.warning(f"APIError on key #{idx + 1}: {e}")
                self._mark_key_rate_limited(idx, duration_sec=30.0)
                last_exception = e
            except Exception as e:
                logger.error(f"Unexpected error on key #{idx + 1}: {e}")
                last_exception = e
                # Continue trying with other keys if network or API error
                time.sleep(1)

        raise RuntimeError(f"Whisper transcription failed after {max_retries} attempts: {last_exception}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = LLM_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        max_retries: int = 5
    ) -> str:
        """
        Executes an LLM chat completion with automatic key rotation and retry.
        """
        last_exception = None

        for attempt in range(max_retries):
            idx = self._get_next_available_key_index()
            client = self.clients[idx]

            try:
                logger.info(f"Generating LLM completion with key #{idx + 1} (model: {model})...")
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                with self.lock:
                    self.usage_stats[idx]["completions"] += 1
                return response.choices[0].message.content or ""

            except RateLimitError as e:
                logger.warning(f"RateLimitError on key #{idx + 1}: {e}")
                self._mark_key_rate_limited(idx, duration_sec=60.0)
                last_exception = e
            except (APIError, InternalServerError) as e:
                logger.warning(f"API error on key #{idx + 1}: {e}")
                self._mark_key_rate_limited(idx, duration_sec=30.0)
                last_exception = e
            except Exception as e:
                logger.error(f"Error on key #{idx + 1}: {e}")
                last_exception = e
                time.sleep(1)

        raise RuntimeError(f"Chat completion failed after {max_retries} attempts: {last_exception}")

    def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = LLM_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        max_retries: int = 5
    ) -> Iterator[str]:
        """
        Streams LLM chat completion tokens with key rotation.
        """
        last_exception = None

        for attempt in range(max_retries):
            idx = self._get_next_available_key_index()
            client = self.clients[idx]

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True
                )
                with self.lock:
                    self.usage_stats[idx]["completions"] += 1

                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
                return

            except RateLimitError as e:
                logger.warning(f"RateLimitError during stream on key #{idx + 1}: {e}")
                self._mark_key_rate_limited(idx, duration_sec=60.0)
                last_exception = e
            except Exception as e:
                logger.warning(f"Error during stream on key #{idx + 1}: {e}")
                last_exception = e
                time.sleep(1)

        raise RuntimeError(f"Chat streaming failed after {max_retries} attempts: {last_exception}")

# Singleton instance
groq_manager = GroqKeyManager()
