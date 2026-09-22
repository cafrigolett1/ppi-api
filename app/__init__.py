"""PII anonymisation service.

Detects and redacts sensitive entities in text - see README.md for the full
architecture. This file exists as a real package marker (not left empty)
specifically so individual COPY instructions in Dockerfile.dashboard, which
reference it by exact filename, have something unambiguous to copy.
"""
