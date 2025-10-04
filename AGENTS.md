# Repository Guidelines

## Project Structure & Module Organization
- Core library lives in `debate_engine/` with orchestration (`core.py`), models, and shared enums.
- Providers and model plumbing are under `models/` (`providers/`, `manager.py`, `cache_manager.py`).
- Debate formats sit in `formats/`; judges in `judges/`; configuration helpers in `config/`.
- Example configs and requirements are at the repository root; keep generated artifacts out of source control.

## Build, Test, and Development Commands
```
pip install -e .              # Editable install of the library
pip install -e .[dev]         # Add linters, pyright, pytest
pyright .                     # Strict type check (no errors tolerated)
pytest                        # Run unit/integration tests (when present)
python -m build               # Build sdist and wheel
```
- Use `pip-compile requirements.in` when pin updates are needed.
- If `pyright` is missing on PATH, run `npx pyright .` (the CLI will auto-install the pinned version).

## Coding Style & Typing Rules
- Target Python 3.13+ with modern type hints (`list[str]`, `dict[str, ModelConfig]`, `X | None`).
- Pyright/Pylance run in strict mode: no implicit `Any`, no legacy `Dict[...]`/`List[...]`/`Optional[...]`, annotate every public function and return value.
- Prefer shared `TypeAlias` definitions for common shapes (`MessageList`, `ChunkCallback`), and type `**overrides` as `object`.
- Import protocols/iterables from `collections.abc`; avoid `typing.AsyncGenerator` etc.
- Follow PEP 8 with 4-space indentation, sorted imports, and zero unused symbols; keep logging structured (`logger.info("msg %s", value)` not f-strings).
- Pydantic v2 only: call `model_dump()` / `model_dump_json()`; avoid deprecated helpers.

## CLI Workflow Notes
## CLI Workflow Notes
- The Codex CLI runs through a Bash shell in this repository; POSIX redirection, heredocs, and common GNU utilities are available.
- Favor `bash -lc '…'` commands when chaining operations; avoid PowerShell-specific syntax.
- When writing files via scripts, ensure UTF-8 encoding and LF line endings.

## Testing Guidelines
- Tests should mirror module paths (e.g., `tests/models/test_manager.py`).
- Name tests with behavior (`test_generate_response_stream_handles_timeouts`).
- Validate both Ollama and OpenRouter flows when touching provider logic.
- Document new scenarios with fixtures or factories instead of inline duplication.

## Commit & Pull Request Guidelines
- Write imperative, scoped commit subjects (`Add OpenRouter rate limit retries`).
- Include concise body lines explaining rationale and side effects.
- PRs must describe behavior change, note impacted configs, and link issues when available.
- Attach validation evidence: pyright output, pytest summary, and debate smoke test notes if engine logic changed.

## Security & Configuration Tips
- Never commit secrets; rely on `OPENROUTER_API_KEY` and local `debate_config.json` copies.
- Validate user inputs via Pydantic models and strip transient data before logging.
- Keep configuration for local credentials out of the repo; `.env`/PowerShell profile entries are safer than hard-coded paths.
