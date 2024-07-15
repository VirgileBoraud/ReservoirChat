# Chat with an intelligent assistant in your terminal
from openai import OpenAI
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

inference_server_url = "http://localhost:1234/v1"

# create vLLM Langchain instance
chat = ChatOpenAI(
    model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
    openai_api_key="lm-studio",
    openai_api_base="http://localhost:1234/v1",
    max_tokens=5,
    temperature=0,
)

# use the Ragas LangchainLLM wrapper to create a RagasLLM instance
vllm = LangchainLLMWrapper(chat)

from ragas.metrics import (
    context_precision,
    faithfulness,
    context_recall,
)
from ragas.metrics.critique import harmfulness

# change the LLM

faithfulness.llm = vllm
context_precision.llm = vllm
context_recall.llm = vllm
harmfulness.llm = vllm

# evaluate
from ragas import evaluate

from datasets import load_dataset

fiqa_eval = load_dataset("explodinggradients/fiqa", "ragas_eval")

result = evaluate(
    fiqa_eval["baseline"].select(range(5)),  # showing only 5 for demonstration
    metrics=[context_precision],
)

result
