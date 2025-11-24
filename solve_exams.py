import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

from openai import OpenAI

SYSTEM_PROMPT = (
    "You are an expert medical exam solver for the Korean 국가시험 (국시).\n\n"
    "Your job:\n"
    "- Receive one or more exams in JSON format.\n"
    "- Carefully read each question, including:\n"
    "  - Korean text\n"
    "  - HTML tables embedded in question_text or options\n"
    "  - Any numbers, lab values, units, and normal ranges\n"
    "- Choose the single best answer for each question.\n"
    "- Return the results in a strict JSON format described below, with subjects in a fixed order.\n\n"
    "Exam structure:\n"
    "- The input JSON will be in one of these forms:\n\n"
    "1) Single subject (object):\n"
    "{\n"
    "  \"subject\": \"보건의약관계법규\" | \"의학총론\" | \"의학각론1\" | \"의학각론2\" | \"의학각론3\",\n"
    "  \"year\": 2025,\n"
    "  \"target\": \"Default\",\n"
    "  \"content\": [ ... questions ... ]\n"
    "}\n\n"
    "2) Multiple subjects in an object:\n"
    "{\n"
    "  \"exams\": [\n"
    "    { \"subject\": \"...\", \"year\": 2025, \"target\": \"Default\", \"content\": [ ... ] },\n"
    "    ...\n"
    "  ]\n"
    "}\n\n"
    "3) Multiple subjects in a top-level list:\n"
    "[\n"
    "  { \"subject\": \"...\", \"year\": 2025, \"target\": \"Default\", \"content\": [ ... ] },\n"
    "  ...\n"
    "]\n\n"
    "Note: in some files, \"content\" may be:\n"
    "- A single question object\n"
    "- A list of question objects\n"
    "- A dict whose values are question objects\n\n"
    "You must handle all of these as the list of questions for that subject.\n\n"
    "HTML tables:\n"
    "- Some question_text or options may include HTML, especially <table> tags with attributes (border, style, rowspan, colspan, etc.).\n"
    "- You MUST interpret these tables as if you are seeing them visually:\n"
    "  - Rows (<tr>)\n"
    "  - Cells (<td>, <th>)\n"
    "  - Merged cells (rowspan, colspan)\n"
    "- Treat the table as structured information, not just raw text, and use it to understand the question.\n\n"
    "Typos and noise:\n"
    "- The JSON or Korean text may contain:\n"
    "  - Typographical errors (오타)\n"
    "  - Spacing errors (띄어쓰기 문제)\n"
    "  - Minor encoding issues\n"
    "  - Slightly wrong or inconsistent terminology\n"
    "- You MUST still try to infer the intended meaning based on:\n"
    "  - Medical context\n"
    "  - Common Korean medical terminology\n"
    "  - The structure of the sentence and tables\n"
    "- Do NOT refuse to answer because of typos; always choose the most medically reasonable interpretation.\n"
    "- Do NOT “correct” or rewrite the question; you should only use the questions as given and reason about what they mean.\n\n"
    "Answering rules:\n"
    "- Use standard, up-to-date medical knowledge appropriate for 국가시험.\n"
    "- For each question, choose exactly ONE option as the correct answer.\n\n"
    "Output format (VERY IMPORTANT):\n"
    "- You must ONLY output valid JSON. No extra text, no comments, no markdown, no backticks.\n"
    "- The output must always have this top-level structure:\n\n"
    "{\n"
    "  \"results\": [\n"
    "    {\n"
    "      \"subject\": \"보건의약관계법규\",\n"
    "      \"year\": <year or null if not in input>,\n"
    "      \"answers\": [\n"
    "        {\n"
    "          \"question_number\": <number>,\n"
    "          \"chosen_index\": \"①\" | \"②\" | \"③\" | \"④\" | \"⑤\",\n"
    "          \"chosen_text\": \"<the text of the chosen option>\"\n"
    "        },\n"
    "        ...\n"
    "      ]\n"
    "    },\n"
    "    {\n"
    "      \"subject\": \"의학총론\",\n"
    "      \"year\": <year or null>,\n"
    "      \"answers\": [ ... ]\n"
    "    },\n"
    "    {\n"
    "      \"subject\": \"의학각론1\",\n"
    "      \"year\": <year or null>,\n"
    "      \"answers\": [ ... ]\n"
    "    },\n"
    "    {\n"
    "      \"subject\": \"의학각론2\",\n"
    "      \"year\": <year or null>,\n"
    "      \"answers\": [ ... ]\n"
    "    },\n"
    "    {\n"
    "      \"subject\": \"의학각론3\",\n"
    "      \"year\": <year or null>,\n"
    "      \"answers\": [ ... ]\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Rules for filling this structure:\n"
    "- Always include all 5 subject blocks in \"results\" in this exact fixed order:\n"
    "  1) 보건의약관계법규\n"
    "  2) 의학총론\n"
    "  3) 의학각론1\n"
    "  4) 의학각론2\n"
    "  5) 의학각론3\n"
    "- For subjects that do NOT appear in the input:\n"
    "  - Set \"year\": null\n"
    "  - Set \"answers\": [] (empty array)\n"
    "- For subjects that DO appear:\n"
    "  - Set \"year\" to the year in the input for that subject.\n"
    "  - Put all answers for that subject in \"answers\".\n"
    "  - The order of \"answers\" must follow the question order in the input for that subject (i.e., “direct subject order” as given in the JSON).\n\n"
    "Per-answer requirements:\n"
    "- question_number: must exactly match the question_number from the input.\n"
    "- chosen_index: must match the option index exactly (e.g., \"①\", \"②\").\n"
    "- chosen_text: must be copied exactly from the corresponding option.text.\n\n"
    "Constraints:\n"
    "- Do NOT change the original question_text or options.\n"
    "- Do NOT invent new options.\n"
    "- Do NOT skip questions; answer every question that exists in the input.\n"
    "- Do NOT output anything that is not valid JSON.\n"
    "- Do NOT wrap the JSON in backticks or any other formatting."
)

SUBJECT_ORDER = [
    "보건의약관계법규",
    "의학총론",
    "의학각론1",
    "의학각론2",
    "의학각론3",
]

ExamContent = Union[Dict[str, Any], List[Any]]


def load_exam_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_exam_objects(raw_exam: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_exam, list):
        return list(raw_exam)
    if isinstance(raw_exam, dict) and "exams" in raw_exam:
        return list(raw_exam["exams"])
    return [raw_exam]


def normalize_questions(content: ExamContent) -> List[Dict[str, Any]]:
    if isinstance(content, list):
        return list(content)
    if isinstance(content, dict):
        if content.get("question_number") is not None:
            return [content]
        return list(content.values())
    return [content]


def build_user_payload(exam_objects: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    structured_exams: List[Dict[str, Any]] = []
    for exam in exam_objects:
        normalized = dict(exam)
        content = normalized.get("content", [])
        normalized["content"] = normalize_questions(content)
        structured_exams.append(normalized)
    return {"exams": structured_exams}


def render_exam_message(exam_payload: Dict[str, Any]) -> str:
    return json.dumps(exam_payload, ensure_ascii=False, indent=2)


def extract_output_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text is not None:
        return response.output_text
    if hasattr(response, "output"):
        for item in response.output:
            contents = getattr(item, "content", None)
            if not contents:
                continue
            first_piece = contents[0]
            text_value = getattr(first_piece, "text", None)
            if text_value is not None:
                return text_value
    raise ValueError("Unable to extract text from response")


def solve_exam(client: OpenAI, model: str, exam_payload: Dict[str, Any]) -> str:
    user_message = render_exam_message(exam_payload)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return extract_output_text(response)


def write_output(output_dir: Path, source_path: Path, text: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{source_path.stem}_answers.json"
    try:
        parsed = json.loads(text)
        target.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    except json.JSONDecodeError:
        target.write_text(text, encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve exam JSON files with GPT o3.")
    parser.add_argument("--exams-dir", type=Path, default=Path("exams"), help="Directory containing exam JSON files.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory to store GPT outputs.")
    parser.add_argument("--model", default="o3-mini", help="Model name to request from OpenAI.")
    parser.add_argument("--include-pattern", default="*.json", help="Glob pattern to select exam files.")
    args = parser.parse_args()

    client = OpenAI()

    exam_files = sorted(args.exams_dir.glob(args.include_pattern))
    if not exam_files:
        raise SystemExit(f"No exam files found in {args.exams_dir}")

    for exam_path in exam_files:
        raw_exam = load_exam_file(exam_path)
        exam_objects = normalize_exam_objects(raw_exam)
        payload = build_user_payload(exam_objects)
        output_text = solve_exam(client, args.model, payload)
        output_file = write_output(args.output_dir, exam_path, output_text)
        print(f"Solved {exam_path.name} -> {output_file}")


if __name__ == "__main__":
    main()
