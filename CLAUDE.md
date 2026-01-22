# Gemini LMS Pipeline

## Project overview and purpose
This repository contains a linear, four-agent prompt-chain that generates Canvas LMS-ready HTML fragments (optionally with hosted images). It is designed to produce textbook chapters, discussion prompts, and assignments with consistent structure and styling.

## Entry point
- `gemini_lms_pipeline.py`

## Pipeline architecture (4-agent workflow)
- Agent A: Content writer (drafts source content)
- Agent B: HTML structurer (converts content to semantic HTML)
- Agent C: Publisher/styler (accessibility, layout, knowledge checks)
- Agent D: Illustrator (plans images, generates them, uploads via ImageKit, injects into HTML)

## Content types supported
- textbook
- discussion
- assignment

## Environment variables required
- `GOOGLE_API_KEY` or `GEMINI_API_KEY` (required)
- `IMAGEKIT_PRIVATE_KEY`, `IMAGEKIT_PUBLIC_KEY`, `IMAGEKIT_URL_ENDPOINT` (required when images are enabled)
- `IMAGEKIT_FOLDER` (optional override for the ImageKit upload folder)

## Directory structure
- `gemini_lms_pipeline.py`: main pipeline entry
- `prompts/`: per-agent prompt templates
- `lms_output/`: pipeline runs stored as `YYYY-MM-DD_HH-MM-SS_<content-type>/` (older runs may omit `<content-type>`)
- `lms_output/legacy_chain_outputs/`: legacy outputs migrated from `chain_outputs/`
- `list_models.py`: optional utility
- `pyproject.toml`, `uv.lock`: dependency metadata

## Usage commands
- `python gemini_lms_pipeline.py`
- `python gemini_lms_pipeline.py --type textbook --topic "Test Topic"`
- `python gemini_lms_pipeline.py --no-images`
- `python gemini_lms_pipeline.py --type discussion --topic "Sample" --force-images`

## Key files vs. legacy/abandoned files
Key files:
- `gemini_lms_pipeline.py`
- `prompts/`
- `pyproject.toml`, `uv.lock`
- `list_models.py` (optional utility)

Legacy/abandoned files removed:
- `original_agent_chain.py`, `revised_agent_chain.py`, `revised_gemini_backup.py`
- `run_agent_c_only.py`, `run_agent_d_images.py`
- `openai/`, `anthropic/`, `generated_images/`
- `agent_*_output.*`, `output_agent_chain.html`, `anthropic.zip`, `nul`

## Cleanup recommendations
- Keep all pipeline outputs under `lms_output/` and avoid root-level `lms_output_*.html` artifacts (now disabled in code).
- Add/verify `.gitignore` entries for `lms_output/`, `generated_images/`, `.venv/`, `.mypy_cache/`, and `.idea/`.
