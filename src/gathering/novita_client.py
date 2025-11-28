import asyncio
import json
import time
from typing import Dict, List, Optional, Union, AsyncGenerator, Generator
import aiohttp
import requests
from openai import OpenAI
from dataclasses import dataclass
import logging

NOVITA_BACKEND_URL = "https://api.novita.ai/openai"

@dataclass
class LLMResponse:
    content: str
    model: str
    usage: Dict[str, int]
    response_time: float
    success: bool
    error: Optional[str] = None

class NovitaClient:
    temperature: float
    max_tokens: int
    client: OpenAI

    def __init__(self, api_key: Optional[str] = None, temperature: float = 1.0, max_tokens: int = 2048):
        with open('NOVITA_API_KEY', 'r') as f:
            novita_key = f.readline().strip()
        if not novita_key or novita_key == "":
            raise ValueError("NOVITA_API_KEY must be set in the NOVITA_API_KEY file")

        self.temperature = temperature
        self.max_tokens = max_tokens

        # Use the API key from file unless overridden
        use_key = api_key or novita_key
        self.client = OpenAI(
            base_url = NOVITA_BACKEND_URL,
            api_key = use_key,
            timeout=300.0  # 5 minutes timeout for Novita API calls
        )
        self.logger = logging.getLogger("NovitaClient")
    
    def complete(self, prompt: str, model: str):
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False
        )

        return response.choices[0].message.content

    def load_api_key(self):
        """Load API key from file if not provided during initialization."""
        try:
            with open('NOVITA_API_KEY', 'r') as f:
                novita_key = f.readline().strip()

            if not novita_key or novita_key == "":
                raise ValueError("NOVITA_API_KEY must be set in the NOVITA_API_KEY file")

            self.client = OpenAI(
                base_url = NOVITA_BACKEND_URL,
                api_key = novita_key,
                timeout=300.0  # 5 minutes timeout for Novita API calls
            )
        except Exception as e:
            raise RuntimeError(f"Could not load API key: {e}")

    async def complete_async(
        self,
        model: str,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retry_count: int = 3
    ) -> LLMResponse:
        """Async completion with retry logic."""
        if not self.client:
            return LLMResponse(
                content="",
                model=model,
                usage={},
                response_time=0.0,
                success=False,
                error="Client not initialized"
            )

        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        for attempt in range(retry_count):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    max_tokens=max_tok,
                    stream=False
                )
                response_time = time.time() - start_time
                content = response.choices[0].message.content

                # Log API response details
                self.logger.info(
                    "API response for model %s: %.1fs, %d tokens, content_length=%d",
                    model, response_time,
                    response.usage.total_tokens if response.usage else 0,
                    len(content) if content else 0
                )
                self.logger.debug("API response content preview: %s",
                                (content[:200] + "...") if content and len(content) > 200 else content)

                return LLMResponse(
                    content=content,
                    model=model,
                    usage=response.usage.model_dump() if response.usage else {},
                    response_time=response_time,
                    success=True
                )
            except Exception as e:
                error_msg = str(e)
                is_timeout_error = "524" in error_msg or "timeout" in error_msg.lower()

                # Use much longer backoff for 524 timeout errors since Novita servers need time to recover
                if is_timeout_error:
                    backoff = min(15 + (15 * attempt), 120)  # 15s, 30s, 45s... up to 2 minutes max
                    self.logger.warning("Server timeout (524) for model %s (attempt %d/%d): %s. Extended backoff %.1fs",
                                      model, attempt + 1, retry_count, error_msg, backoff)
                else:
                    backoff = min(2 ** attempt + (0.5 * attempt), 120)  # exponential with jitter, capped at 2 minutes
                    self.logger.warning("Completion error for model %s (attempt %d/%d): %s. Backing off %.1fs",
                                      model, attempt + 1, retry_count, error_msg, backoff)

                if attempt == retry_count - 1:
                    return LLMResponse(
                        content="",
                        model=model,
                        usage={},
                        response_time=0.0,
                        success=False,
                        error=str(e)
                    )
                await asyncio.sleep(backoff)

    async def close(self):
        """Close the client connection."""
        pass
