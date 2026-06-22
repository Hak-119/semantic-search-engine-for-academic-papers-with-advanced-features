"""
backend/llm.py
LLM integration for paper summarization and answer synthesis.

Primary: Groq (Llama 3.3 70B, Llama 3.1 8B Instant) - fast, free tier
Fallback: HuggingFace Inference Providers (Qwen 2.5 7B Instruct) - free, no Groq key needed

Architecture: every provider function shares the same signature
(prompt, model, max_tokens) -> str. Adding a new provider later
(Anthropic, OpenAI) means adding one function + one registry entry -
nothing else in the app needs to change.
"""

import os
from dotenv import load_dotenv
from groq import Groq
from huggingface_hub import InferenceClient

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")


#Groq Provider

def query_groq(prompt: str, model: str = "llama-3.3-70b-versatile", max_tokens: int = 1000) -> str:
    """
    Query a Groq-hosted model via chat completion.
    Models: llama-3.3-70b-versatile (quality), llama-3.1-8b-instant (speed)
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set. Add it to your .env file.")

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert academic research assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content


#HuggingFace Provider (fallback - no Groq key needed)

def query_huggingface(prompt: str, model: str = "Qwen/Qwen2.5-7B-Instruct", max_tokens: int = 1000) -> str:
    """
    Query a HuggingFace-hosted model via Inference Providers chat completion.
    Free tier, works without a Groq key.
    """
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY not set. Add it to your .env file.")

    client = InferenceClient(api_key=HF_API_KEY)
    response = client.chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert academic research assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content


#Model Registry - the modular swap point

MODEL_REGISTRY = {
    "Llama 3.3 70B (Groq)": {"provider": "groq", "model_id": "llama-3.3-70b-versatile"},
    "Llama 3.1 8B Instant (Groq)": {"provider": "groq", "model_id": "llama-3.1-8b-instant"},
    "Qwen 2.5 7B (HuggingFace)": {"provider": "huggingface", "model_id": "Qwen/Qwen2.5-7B-Instruct"},

    # Future paid providers slot in here with zero changes elsewhere:
    # "Claude Sonnet (Anthropic)": {"provider": "anthropic", "model_id": "claude-sonnet-4-5"},
    # "GPT-4o (OpenAI)": {"provider": "openai", "model_id": "gpt-4o"},
}


def generate_response(prompt: str, model_choice: str, max_tokens: int = 1000) -> str:
    """
    Single entry point for all LLM calls. Looks up model_choice in the
    registry and routes to the correct provider function. The rest of
    the app only ever calls this - it never needs to know which
    provider or model is actually running underneath.
    """
    if model_choice not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model choice: {model_choice}")

    config = MODEL_REGISTRY[model_choice]
    provider = config["provider"]
    model_id = config["model_id"]

    if provider == "groq":
        return query_groq(prompt, model=model_id, max_tokens=max_tokens)
    elif provider == "huggingface":
        return query_huggingface(prompt, model=model_id, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")