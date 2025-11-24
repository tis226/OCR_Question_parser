# OCR Question Parser Solver

This repository contains a helper script to send the JSON exams in `exams/` to OpenAI's GPT o3 model and save the formatted answer key responses.

## Requirements
- Python 3.12+
- An `OPENAI_API_KEY` environment variable with access to GPT o3 (the default model is `o3-mini`).

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage
Run the solver to process every `.json` file in `exams/` and write the model outputs to `outputs/`:

```bash
python solve_exams.py --exams-dir exams --output-dir outputs --model o3-mini
```

Use `--include-pattern` to target a subset of files (for example, a single exam file):

```bash
python solve_exams.py --include-pattern "2025_exam_의학총론.json"
```

The script normalizes question lists regardless of whether each exam file uses a single object, an `exams` array, or a dictionary of question objects, and preserves the direct question order when sending data to the model.
