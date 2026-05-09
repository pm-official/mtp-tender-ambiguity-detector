"""Tender + corrigendum acquisition utilities."""

from src.acquire.manifest import (
    Manifest,
    TenderDocument,
    empty_manifest,
    infer_corrigendum_sequence,
    infer_role_from_filename,
    manifest_path,
    read_manifest,
    sha256_of_file,
    write_manifest,
)
from src.acquire.pdf_validator import PDFInfo, inspect_pdf, quick_magic_check

__all__ = [
    "Manifest",
    "TenderDocument",
    "empty_manifest",
    "infer_corrigendum_sequence",
    "infer_role_from_filename",
    "manifest_path",
    "read_manifest",
    "sha256_of_file",
    "write_manifest",
    "PDFInfo",
    "inspect_pdf",
    "quick_magic_check",
]
