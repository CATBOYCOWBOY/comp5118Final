import asyncio
import json
import time
from typing import Dict, List, Optional, Union, AsyncGenerator, Generator
import aiohttp
import requests
from openai import OpenAI
from dataclasses import dataclass

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

    def __init__(self, api_key: Optional[str] = None, temperature: float = 1.0, max_tokens: int = 1024):
        self.temperature = temperature
        self.max_tokens = max_tokens

        if api_key:
            self.client = OpenAI(
                base_url = NOVITA_BACKEND_URL,
                api_key = api_key
            )
        else:
            self.client = None
    
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
                api_key = novita_key
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

                return LLMResponse(
                    content=response.choices[0].message.content,
                    model=model,
                    usage=response.usage.model_dump() if response.usage else {},
                    response_time=response_time,
                    success=True
                )
            except Exception as e:
                if attempt == retry_count - 1:
                    return LLMResponse(
                        content="",
                        model=model,
                        usage={},
                        response_time=0.0,
                        success=False,
                        error=str(e)
                    )
                await asyncio.sleep(1)

    async def close(self):
        """Close the client connection."""
        pass
