#!/usr/bin/env python3
"""
Document AI batch extractor.

Reads every supported document from the input folder (default: ./data),
runs it through a Google Cloud Document AI processor, and writes the
extracted information to the output folder (default: ./result).

Usage:
    python myrun.py
    python myrun.py --input data --output result
    python myrun.py --processor-type OCR_PROCESSOR

Configuration (CLI flags override environment variables override defaults):
    PROJECT_ID      GCP project        (default: gcloud's active project)
    LOCATION        Processor region   (default: us)   -> "us" or "eu"
    PROCESSOR_ID    Existing processor id to reuse (optional)
    PROCESSOR_TYPE  Processor to create if none exists (default: FORM_PARSER_PROCESSOR)

Auth uses Application Default Credentials (gcloud auth application-default login).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPICallError, PermissionDenied
from google.cloud import documentai

# --- File types Document AI can read (extension -> MIME type) -----------------
MIME_TYPES = {
    ".pdf": "application/pdf",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".gif": "image/gif",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}


# --- Config -------------------------------------------------------------------
def gcloud_project() -> str | None:
    try:
        out = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True, text=True, timeout=15,
        )
        val = out.stdout.strip()
        return val or None
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract information from documents with Google Cloud Document AI.")
    p.add_argument("--input", default=os.getenv("INPUT_DIR", "data"), help="Input folder (default: data)")
    p.add_argument("--output", default=os.getenv("OUTPUT_DIR", "result"), help="Output folder (default: result)")
    p.add_argument("--project", default=os.getenv("PROJECT_ID") or gcloud_project(), help="GCP project id")
    p.add_argument("--location", default=os.getenv("LOCATION", "us"), help="Processor region: us or eu (default: us)")
    p.add_argument("--processor-id", default=os.getenv("PROCESSOR_ID"), help="Existing processor id to reuse")
    p.add_argument("--processor-type", default=os.getenv("PROCESSOR_TYPE", "FORM_PARSER_PROCESSOR"),
                   help="Processor type to create if none exists (default: FORM_PARSER_PROCESSOR)")
    return p.parse_args()


# --- Document AI helpers ------------------------------------------------------
def make_client(location: str) -> documentai.DocumentProcessorServiceClient:
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    return documentai.DocumentProcessorServiceClient(client_options=opts)


def resolve_processor(client, project: str, location: str,
                      processor_id: str | None, processor_type: str) -> str:
    """Return a full processor resource name, reusing or creating as needed."""
    if processor_id:
        name = client.processor_path(project, location, processor_id)
        print(f"Using processor id: {processor_id}")
        return name

    parent = client.common_location_path(project, location)

    # Reuse an existing processor of the requested type if one already exists.
    for proc in client.list_processors(parent=parent):
        if proc.type_ == processor_type:
            print(f"Reusing existing processor: {proc.display_name} ({proc.name.split('/')[-1]})")
            return proc.name

    # Otherwise create one.
    print(f"No {processor_type} found. Creating one...")
    proc = client.create_processor(
        parent=parent,
        processor=documentai.Processor(
            type_=processor_type,
            display_name=f"cli-{processor_type.lower()}",
        ),
    )
    print(f"Created processor: {proc.display_name} ({proc.name.split('/')[-1]})")
    return proc.name


def layout_text(layout, full_text: str) -> str:
    """Reconstruct the text a layout element points to via its text anchor."""
    if not layout.text_anchor.text_segments:
        return ""
    parts = []
    for seg in layout.text_anchor.text_segments:
        start = int(seg.start_index) if seg.start_index else 0
        end = int(seg.end_index)
        parts.append(full_text[start:end])
    return "".join(parts).strip()


def extract(document) -> dict:
    """Pull the useful structured pieces out of a processed Document."""
    text = document.text
    result = {
        "page_count": len(document.pages),
        "text": text,
        "form_fields": [],
        "tables": [],
        "entities": [],
    }

    for page in document.pages:
        # Key-value pairs (Form Parser).
        for field in page.form_fields:
            result["form_fields"].append({
                "key": layout_text(field.field_name, text),
                "value": layout_text(field.field_value, text),
                "page": page.page_number,
            })
        # Tables.
        for table in page.tables:
            def rows(section):
                out = []
                for row in section:
                    out.append([layout_text(cell.layout, text) for cell in row.cells])
                return out
            result["tables"].append({
                "page": page.page_number,
                "header_rows": rows(table.header_rows),
                "body_rows": rows(table.body_rows),
            })

    # Entities (specialized parsers: invoice, receipt, etc.).
    for ent in document.entities:
        result["entities"].append({
            "type": ent.type_,
            "value": ent.mention_text or ent.normalized_value.text,
            "confidence": round(ent.confidence, 4),
        })

    return result


def process_file(client, processor_name: str, path: Path, mime: str):
    content = path.read_bytes()
    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=documentai.RawDocument(content=content, mime_type=mime),
    )
    result = client.process_document(request=request)
    return result.document


# --- Main ---------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    if not args.project:
        print("ERROR: No GCP project. Set PROJECT_ID or run: gcloud config set project <id>", file=sys.stderr)
        return 1

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.is_dir():
        print(f"ERROR: input folder not found: {in_dir}", file=sys.stderr)
        return 1

    files = sorted(f for f in in_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in MIME_TYPES)
    if not files:
        print(f"No supported documents in '{in_dir}'. Supported: {', '.join(sorted(MIME_TYPES))}")
        return 0

    print(f"Project : {args.project}")
    print(f"Location: {args.location}")
    print(f"Input   : {in_dir}  ({len(files)} file(s))")
    print(f"Output  : {out_dir}\n")

    client = make_client(args.location)
    try:
        processor_name = resolve_processor(
            client, args.project, args.location, args.processor_id, args.processor_type
        )
    except PermissionDenied as e:
        print(f"ERROR: permission denied resolving processor: {e.message}", file=sys.stderr)
        print("Make sure the Document AI API is enabled and your account has the "
              "'Document AI Editor' role.", file=sys.stderr)
        return 1

    print()
    ok, failed = 0, 0
    for f in files:
        mime = MIME_TYPES[f.suffix.lower()]
        print(f"-> {f.name} ...", end=" ", flush=True)
        try:
            document = process_file(client, processor_name, f, mime)
            data = extract(document)

            (out_dir / f"{f.stem}.txt").write_text(data["text"], encoding="utf-8")
            (out_dir / f"{f.stem}.json").write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"OK  ({data['page_count']} page(s), "
                  f"{len(data['form_fields'])} field(s), "
                  f"{len(data['tables'])} table(s))")
            ok += 1
        except GoogleAPICallError as e:
            print(f"FAILED: {e.message}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAILED: {e}")
            failed += 1

    print(f"\nDone. {ok} succeeded, {failed} failed. Results in '{out_dir}'.")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
