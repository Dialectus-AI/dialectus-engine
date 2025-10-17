# Local Model Files Directory

This directory is for **local model files only** when testing the moderation system with custom or downloaded models.

## Purpose

Developers working on the Dialectus Engine can place local model files here (GGUF, safetensors, etc.) for testing with Ollama.

## Not Distributed

⚠️ **Important**:
- **Git-ignored** - `*.gguf`, `*.bin`, `*.safetensors` are excluded from version control
- **Never distributed** - Model files are NOT included in the PyPI package
- **Optional** - Most developers pull models directly from Ollama's registry

The `dialectus-engine` package is model-agnostic and ships without any AI models.

## Usage

### Option 1: Pull from Registry (Recommended)

Most developers should pull models directly from Ollama:

```bash
ollama pull <model-name>
```

Then reference it in your test config. No files needed in this directory.

### Option 2: Use Local Model Files

If you have a local GGUF file to test with:

1. Place the file here: `moderation/your-model.gguf`

2. Create an Ollama Modelfile in the repository root:

```bash
cat > Modelfile <<EOF
FROM ./moderation/your-model.gguf
TEMPLATE """{{ .System }}

{{ .Prompt }}"""
EOF
```

3. Import into Ollama:

```bash
ollama create my-test-model -f Modelfile
```

4. Reference in your test scripts:

```python
moderator = LLMModerator(
    base_url="http://localhost:11434/v1",
    model="my-test-model",
    timeout=15.0,
)
```

## Testing

See `MODERATION_TESTING.md` in the repository root for comprehensive testing instructions covering all supported providers (Ollama, OpenAI, OpenRouter, etc.).

## Notes

This directory will typically remain empty. It exists as a convention for developers who want to test with custom or unreleased model files.
