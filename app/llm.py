"""Provider-agnostic LLM gateway with an 8-GB-M1-safe local default."""
import os, re, json
from functools import lru_cache
from dotenv import load_dotenv
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn): return fn
        return decorator

load_dotenv(override=True)
LLM_PROVIDER=os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_BASE_URL=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_FAST=os.getenv("OLLAMA_MODEL_FAST", os.getenv("LLM_MODEL", "llama3.2:3b"))
MODEL_REASONING=os.getenv("OLLAMA_MODEL_REASONING", MODEL_FAST)
MODEL_SYNTHESIS=os.getenv("OLLAMA_MODEL_SYNTHESIS", MODEL_REASONING)
OPENAI_MODEL=os.getenv("OPENAI_MODEL", "gpt-5.5")
LLM_TEMPERATURE=float(os.getenv("LLM_TEMPERATURE", "0.1"))
OLLAMA_KEEP_ALIVE=os.getenv("OLLAMA_KEEP_ALIVE", "0")
FORCE_MOCK_LLM=os.getenv("FORCE_MOCK_LLM", "false").lower()=="true"

MODEL_ROLES={"fast":MODEL_FAST,"reasoning":MODEL_REASONING,"synthesis":MODEL_SYNTHESIS}

def _provider_model(role:str)->str:
    if LLM_PROVIDER == "openai":
        return OPENAI_MODEL
    return MODEL_ROLES.get(role, MODEL_FAST)

@lru_cache(maxsize=8)
def _get_llm(model:str, temperature:float=0.1):
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)
    from langchain_ollama import ChatOllama
    # keep_alive=0 prevents a large local model from remaining resident on an 8-GB Mac.
    return ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=temperature, keep_alive=OLLAMA_KEEP_ALIVE)

def _extract_json(text:str)->dict:
    text=text.strip()
    text=re.sub(r"^```(?:json)?", "", text).strip()
    text=re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match=re.search(r"\{.*\}", text, flags=re.S)
        if match:
            return json.loads(match.group(0))
        raise

@traceable(name="underwriting.llm_json", run_type="chain")
def call_llm_json(prompt:str, mock_fallback:dict, role:str="fast", temperature:float=LLM_TEMPERATURE)->tuple[dict,bool]:
    if FORCE_MOCK_LLM:
        return mock_fallback, True
    model=_provider_model(role)
    try:
        llm=_get_llm(model, temperature)
        config={"tags":[f"provider:{LLM_PROVIDER}", f"model_role:{role}"], "metadata":{"provider":LLM_PROVIDER,"model_role":role,"model":model}}
        response=llm.invoke(prompt, config=config)
        content=response.content if hasattr(response,"content") else str(response)
        return _extract_json(content), False
    except Exception:
        return mock_fallback, True

def configured_models()->dict:
    if LLM_PROVIDER == "openai":
        return {"provider":"openai", "fast":OPENAI_MODEL, "reasoning":OPENAI_MODEL, "synthesis":OPENAI_MODEL}
    return {"provider":"ollama", **MODEL_ROLES}
