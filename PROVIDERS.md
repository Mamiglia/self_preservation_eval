# Multi-Provider Support

This evaluation framework supports multiple AI model providers through [Inspect AI](https://inspect.ai-safety-institute.org.uk/). You can evaluate models from OpenAI, Anthropic, Google, HuggingFace, OpenRouter, and more.

## Quick Start

The framework uses a provider-agnostic model format: `provider/model-name`

```bash
# OpenAI
python run_eval.py --model openai/gpt-4o

# Anthropic
python run_eval.py --model anthropic/claude-3-5-sonnet-20241022

# Google Gemini
python run_eval.py --model google/gemini-1.5-pro

# OpenRouter (access to many models)
python run_eval.py --model openrouter/meta-llama/llama-3.1-70b-instruct

# HuggingFace (self-hosted or API)
python run_eval.py --model hf/meta-llama/Meta-Llama-3-8B-Instruct
```

## Installation

### Base Installation

The base installation includes only core dependencies:

```bash
pip install -e .
```

### Provider-Specific Installation

Install only the providers you need:

```bash
# OpenAI only
pip install -e ".[openai]"

# Anthropic only
pip install -e ".[anthropic]"

# Google Gemini only
pip install -e ".[google]"

# HuggingFace only
pip install -e ".[huggingface]"

# All providers
pip install -e ".[all-providers]"
```

## Provider Setup

### OpenAI

**Models:** GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, etc.

1. Get API key from https://platform.openai.com/api-keys
2. Add to `.env`:
   ```
   OPENAI_API_KEY=sk-...
   ```
3. Run evaluation:
   ```bash
   python run_eval.py --model openai/gpt-4o
   python run_eval.py --model openai/gpt-4o-mini
   python run_eval.py --model openai/gpt-3.5-turbo
   ```

### Anthropic

**Models:** Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku, etc.

1. Get API key from https://console.anthropic.com/
2. Add to `.env`:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```
3. Run evaluation:
   ```bash
   python run_eval.py --model anthropic/claude-3-5-sonnet-20241022
   python run_eval.py --model anthropic/claude-3-opus-20240229
   python run_eval.py --model anthropic/claude-3-haiku-20240307
   ```

### Google Gemini

**Models:** Gemini 1.5 Pro, Gemini 1.5 Flash, etc.

1. Get API key from https://aistudio.google.com/app/apikey
2. Add to `.env`:
   ```
   GOOGLE_API_KEY=...
   ```
3. Run evaluation:
   ```bash
   python run_eval.py --model google/gemini-1.5-pro
   python run_eval.py --model google/gemini-1.5-flash
   ```

### OpenRouter

**Models:** Access to 100+ models from various providers

OpenRouter provides a unified API to access models from OpenAI, Anthropic, Meta, Google, and many others.

1. Get API key from https://openrouter.ai/keys
2. Add to `.env`:
   ```
   OPENROUTER_API_KEY=sk-or-...
   ```
3. Run evaluation:
   ```bash
   # Meta Llama
   python run_eval.py --model openrouter/meta-llama/llama-3.1-70b-instruct
   
   # Mistral
   python run_eval.py --model openrouter/mistralai/mistral-large
   
   # Through OpenRouter, you can also access OpenAI/Anthropic models
   python run_eval.py --model openrouter/openai/gpt-4o
   python run_eval.py --model openrouter/anthropic/claude-3.5-sonnet
   ```

See available models at https://openrouter.ai/models

### HuggingFace

**Models:** Self-hosted or HuggingFace Inference API

#### Option 1: HuggingFace Inference API

1. Get token from https://huggingface.co/settings/tokens
2. Add to `.env`:
   ```
   HF_TOKEN=hf_...
   ```
3. Run evaluation:
   ```bash
   python run_eval.py --model hf/meta-llama/Meta-Llama-3-8B-Instruct
   python run_eval.py --model hf/mistralai/Mistral-7B-Instruct-v0.2
   ```

#### Option 2: Self-Hosted Models

For self-hosted models, you can use a local endpoint:

1. Start your model server (e.g., using vLLM, text-generation-inference, etc.)
2. Configure the endpoint:
   ```bash
   export HF_BASE_URL=http://localhost:8000/v1
   ```
3. Run evaluation:
   ```bash
   python run_eval.py --model hf/local-model
   ```

### Azure OpenAI

**Models:** GPT-4, GPT-3.5 via Azure

1. Get credentials from Azure portal
2. Add to `.env`:
   ```
   AZURE_OPENAI_API_KEY=...
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   ```
3. Run evaluation:
   ```bash
   python run_eval.py --model azure/your-deployment-name
   ```

## Environment Variables Reference

Create a `.env` file in the project root with the API keys you need:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini
GOOGLE_API_KEY=...

# HuggingFace
HF_TOKEN=hf_...

# OpenRouter
OPENROUTER_API_KEY=sk-or-...

# Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

You only need to set the keys for providers you plan to use.

## Cost Considerations

Different providers have different pricing models. Here are some cost-effective options for evaluation:

**Budget-friendly options:**
- `openai/gpt-4o-mini` - $0.15/$0.60 per 1M tokens (input/output)
- `anthropic/claude-3-haiku-20240307` - $0.25/$1.25 per 1M tokens
- `google/gemini-1.5-flash` - $0.075/$0.30 per 1M tokens
- OpenRouter models - varies by model, often competitive pricing

**Use `--limit N`** to test with a small sample first:
```bash
python run_eval.py --model openai/gpt-4o --limit 10
```

## Troubleshooting

### "Model not found" errors

Make sure you've installed the required provider package:
```bash
pip install -e ".[openai]"  # or anthropic, google, etc.
```

### Authentication errors

1. Check that your API key is set in `.env`
2. Verify the key is valid and has sufficient credits
3. Make sure `.env` is in the project root directory

### Rate limiting

If you hit rate limits:
1. Add delays between requests (Inspect AI handles this automatically)
2. Use `--limit` to run smaller batches
3. Consider using a provider with higher rate limits

## Advanced: Custom Providers

Inspect AI supports custom model providers. See the [Inspect AI documentation](https://inspect.ai-safety-institute.org.uk/models.html) for details on implementing custom providers.

## More Information

- [Inspect AI Model Providers Documentation](https://inspect.ai-safety-institute.org.uk/models.html)
- [OpenRouter Model List](https://openrouter.ai/models)
- [Anthropic Models](https://docs.anthropic.com/claude/docs/models-overview)
- [OpenAI Models](https://platform.openai.com/docs/models)
- [Google AI Models](https://ai.google.dev/models/gemini)
