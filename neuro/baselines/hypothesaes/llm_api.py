"""LLM API utilities for HypotheSAEs."""

import os
import time
import openai
from azure.identity import ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential, get_bearer_token_provider
from openai import AzureOpenAI


def get_client():
    """Get the OpenAI client, initializing it if necessary."""
    
    client_id = os.environ.get("AZURE_CLIENT_ID")

    scope = "https://cognitiveservices.azure.com/.default"
    credential = get_bearer_token_provider(ChainedTokenCredential(
        AzureCliCredential(), # first check local
        ManagedIdentityCredential(client_id=client_id)
    ), scope)
    client = AzureOpenAI(
        api_version="2025-01-01-preview",
        azure_endpoint="https://dl-openai-1.openai.azure.com/",
        azure_ad_token_provider=credential,
    )
    return client

def get_completion(
    prompt: str,
    model: str = "gpt-4o",
    timeout: float = 15.0,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    **kwargs
) -> str:
    """
    Get completion from OpenAI API with retry logic and timeout.
    
    Args:
        prompt: The prompt to send
        model: Model to use
        max_retries: Maximum number of retries on rate limit
        backoff_factor: Factor to multiply backoff time by after each retry
        timeout: Timeout for the request
        **kwargs: Additional arguments to pass to the OpenAI API; max_tokens, temperature, etc.
    Returns:
        Generated completion text
    
    Raises:
        Exception: If all retries fail
    """
    client = get_client()
    model_abbrev_to_id = {}
    model_id = model_abbrev_to_id.get(model, model)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
                **kwargs
            )
            return response.choices[0].message.content
            
        except (openai.RateLimitError, openai.APITimeoutError) as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e
            
            wait_time = timeout * (backoff_factor ** attempt)
            if attempt > 0:
                print(f"API error: {e}; retrying in {wait_time:.1f}s... ({attempt + 1}/{max_retries})")
            time.sleep(wait_time)