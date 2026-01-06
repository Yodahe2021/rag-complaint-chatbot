# =============================================================================
# llm.py - Language Model Loading (HuggingFace FLAN-T5)
# =============================================================================
"""
This module handles loading the LLM (Large Language Model) for text generation.

KEY CONCEPTS FOR BEGINNERS:

WHAT IS AN LLM?
- A model that generates text based on input
- Takes a prompt (question + context) and produces an answer
- Examples: GPT, FLAN-T5, LLaMA, etc.

WHY FLAN-T5-SMALL?
- Small enough to run on CPU (~300MB)
- Instruction-tuned (good at following prompts)
- Free and open source
- Perfect for learning RAG concepts

LIMITATIONS:
- Small model = limited quality
- May produce short or incomplete answers
- Good for demos, not production

For better quality, you could use:
- Larger models (flan-t5-base, flan-t5-large)
- API-based models (OpenAI, Anthropic)
- Local models via Ollama (llama3, mistral)
"""

from typing import Optional
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from . import config


def get_llm(
    model_name: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.1
) -> HuggingFacePipeline:
    """
    Load the LLM for text generation.
    
    We use google/flan-t5-small:
    - Instruction-tuned T5 model
    - ~300MB download
    - Runs on CPU
    - Good for demos and learning
    
    Args:
        model_name: HuggingFace model name (default from config)
        max_new_tokens: Maximum tokens to generate (default 256)
        temperature: Randomness (0=deterministic, 1=creative)
        
    Returns:
        LangChain-compatible LLM object
        
    Example:
        >>> llm = get_llm()
        >>> response = llm.invoke("What is RAG?")
        >>> print(response)
    """
    if model_name is None:
        model_name = config.LLM_MODEL_NAME
    
    print(f"Loading LLM: {model_name}")
    print(f"  (First run will download ~300MB to {config.MODELS_DIR})")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Temperature: {temperature}")
    
    # Load tokenizer
    # The tokenizer converts text to tokens (numbers) and back
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(config.MODELS_DIR),
    )
    
    # Load model
    # AutoModelForSeq2SeqLM is for encoder-decoder models like T5
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        cache_dir=str(config.MODELS_DIR),
    )
    
    # Create a HuggingFace pipeline
    # This wraps the model for easy text generation
    pipe = pipeline(
        "text2text-generation",  # Task type for T5
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,  # Only sample if temperature > 0
        device=-1,  # -1 = CPU, 0 = first GPU
    )
    
    # Wrap in LangChain's HuggingFacePipeline for compatibility
    llm = HuggingFacePipeline(pipeline=pipe)
    
    print("✓ LLM loaded and ready")
    return llm


def test_llm(llm) -> str:
    """
    Quick test to verify the LLM is working.
    
    Args:
        llm: LangChain LLM object
        
    Returns:
        Generated response
    """
    test_prompt = "What is customer support? Answer in one sentence."
    
    print("Testing LLM with prompt:", test_prompt)
    response = llm.invoke(test_prompt)
    print(f"Response: {response}")
    
    return response


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

# The RAG prompt template
# This tells the LLM how to behave and what format to use
RAG_PROMPT_TEMPLATE = """You are a helpful customer support analytics assistant.
Your job is to answer questions about support tickets using ONLY the provided context.

INSTRUCTIONS:
1. Use ONLY the information from the context below
2. If the context doesn't contain enough information, say "I don't have enough information to answer this question."
3. Be concise and factual
4. Reference specific details from the tickets when possible

CONTEXT (Support Ticket Information):
{context}

QUESTION: {question}

ANSWER:"""


def format_docs_for_context(docs: list) -> str:
    """
    Format retrieved documents into a context string for the prompt.
    
    Args:
        docs: List of LangChain Documents
        
    Returns:
        Formatted string with all document contents
    """
    context_parts = []
    
    for i, doc in enumerate(docs, 1):
        # Include key metadata for context
        ticket_id = doc.metadata.get('ticket_id', 'Unknown')
        product = doc.metadata.get('product', 'Unknown')
        ticket_type = doc.metadata.get('ticket_type', 'Unknown')
        priority = doc.metadata.get('priority', 'Unknown')
        
        # Format this document's context
        context_parts.append(
            f"[Ticket {ticket_id}] Product: {product} | Type: {ticket_type} | Priority: {priority}\n"
            f"{doc.page_content}"
        )
    
    # Join all contexts with separators
    return "\n\n---\n\n".join(context_parts)
