from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from cv_module.config import CLASS_LABELS_VI
from cv_module.config import PROJECT_ROOT
from cv_module.confidence import build_confidence_ir


DEFAULT_DS107_ROOT = PROJECT_ROOT.parent / "DS107"


def build_ds107_payload(cv_result, session_id=None):
    summary = cv_result.get("summary", {})
    detections = cv_result.get("detections", [])
    confidence_ir = cv_result.get("confidence_ir") or build_confidence_ir(detections)
    class_id = summary.get("predicted_label") or "normal"
    disease = CLASS_LABELS_VI.get(class_id, class_id)

    notes = [
        f"CV confidence level: {confidence_ir.get('level')}",
        f"LLM strategy: {confidence_ir.get('llm_strategy')}",
        f"Reason: {confidence_ir.get('reason')}",
    ]

    if confidence_ir.get("needs_llm_confirmation"):
        notes.append(
            "Ket qua YOLO chua chac chan; DS107 nen hoi them trieu chung, "
            "giai doan lua, mat do sau va dieu kien dong ruong."
        )

    return {
        "session_id": session_id,
        "source": "cv_yolo",
        "class_id": class_id,
        "disease": disease,
        "confidence": summary.get("predicted_confidence"),
        "image": cv_result.get("source"),
        "annotated_image": cv_result.get("annotated_image"),
        "needs_llm_confirmation": confidence_ir.get("needs_llm_confirmation", True),
        "llm_strategy": confidence_ir.get("llm_strategy"),
        "notes": " ".join(note for note in notes if note),
        "detections": detections,
    }


def build_ds107_session_command(payload, ds107_root, python_executable="python", overwrite=False):
    command = [
        python_executable,
        str(Path("advisor") / "gemini_rag.py"),
        "sessions",
        "start",
        payload["session_id"] or "cv-session",
        "--class-id",
        payload["class_id"],
        "--disease",
        payload["disease"],
        "--source",
        payload["source"],
    ]

    if payload.get("confidence") is not None:
        command.extend(["--confidence", str(payload["confidence"])])
    if payload.get("image"):
        command.extend(["--image", payload["image"]])
    if payload.get("notes"):
        command.extend(["--notes", payload["notes"]])
    if overwrite:
        command.append("--overwrite")

    return command


def start_ds107_session(payload, ds107_root, python_executable="python", overwrite=False):
    ds107_root = Path(ds107_root).resolve()
    command = build_ds107_session_command(
        payload,
        ds107_root=ds107_root,
        python_executable=python_executable,
        overwrite=overwrite,
    )
    return subprocess.run(command, cwd=ds107_root, check=True, text=True)


def load_first_result(path):
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        if not data:
            raise ValueError("Result file is empty.")
        return data[0]
    return data


def main():
    parser = argparse.ArgumentParser(description="Create DS107 payload from CV result.")
    parser.add_argument("--result", required=True, help="Path to CV JSON result.")
    parser.add_argument("--session-id", default="cv-session")
    parser.add_argument("--output", default="")
    parser.add_argument("--ds107-root", default=str(DEFAULT_DS107_ROOT))
    parser.add_argument("--start-session", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    payload = build_ds107_payload(load_first_result(args.result), session_id=args.session_id)
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.start_session:
        start_ds107_session(
            payload,
            ds107_root=args.ds107_root,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
