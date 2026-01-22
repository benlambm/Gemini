# Gemini LMS Pipeline

Generate Canvas LMS-ready educational HTML using a four-agent prompt chain (content → structure → style → images).

## Design setup (four-agent workflow)
- Agent A: Content writer (drafts source content).
- Agent B: HTML structurer (converts content to semantic HTML).
- Agent C: Publisher/styler (applies accessibility-minded styling and knowledge checks).
- Agent D: Illustrator (plans images, generates/hosts them, and injects into HTML).

## Workflow options
- Interactive mode (menu-driven): `python gemini_lms_pipeline.py`
- Non-interactive mode: pass `--type` and `--topic`
- Images on/off:
  - Default: images enabled only for `textbook`
  - Force images: `--force-images`
  - Disable images: `--no-images`
- Agent A prompt override (non-interactive): `--agent-a-instruction "<custom text>"`
- Agent A preset (non-interactive): `--agent-a-preset blog_intro_1000`

## Requirements
- Python 3.13+ (see `pyproject.toml`)
- API keys for Gemini (required) and ImageKit (optional for hosted images)

## Setup
1) Create and activate a virtual environment.
2) Install dependencies.

Example (pip):
```
python -m venv .venv
.\.venv\Scripts\activate
pip install google-genai imagekitio
```

Example (uv):
```
uv sync
```

## Environment variables
Required:
- `GOOGLE_API_KEY` or `GEMINI_API_KEY`

Required when images are enabled:
- `IMAGEKIT_PRIVATE_KEY`
- `IMAGEKIT_PUBLIC_KEY`
- `IMAGEKIT_URL_ENDPOINT`

Optional:
- `IMAGEKIT_FOLDER` (upload folder override)

## Run
Interactive:
```
python gemini_lms_pipeline.py
```

Non-interactive examples:
```
python gemini_lms_pipeline.py --type textbook --topic "Test Topic"
python gemini_lms_pipeline.py --type discussion --topic "AI scenario" --no-images
python gemini_lms_pipeline.py --type assignment --topic "Case study" --force-images
```

## Output structure
Runs are stored under `lms_output/`:
```
lms_output/
  YYYY-MM-DD_HH-MM-SS_<content-type>/
    content/
      Step1_Content.txt
      Step2_Structured.html
      Step3_Styled.html
      Step4_Final.html
    images/
      fig1-YYYY-MM-DD_HH-MM-SS.png
      ...
    pipeline.log
```

## Notes
- Image generation is only enabled by default for `textbook` content types.
- If Agent D JSON parsing fails, raw responses are saved as `debug_agent_d_brainstorm_attempt*.txt` in the run folder.
- The pipeline configures UTF-8 console output at startup to avoid Unicode errors on Windows terminals.
