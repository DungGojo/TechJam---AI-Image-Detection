#!/usr/bin/env python3
"""User-facing entry point for the full pipeline; implementation lives in Workflows.classify."""

from Workflows.classify import classify_images, collect_images, main

__all__ = ["classify_images", "collect_images", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
