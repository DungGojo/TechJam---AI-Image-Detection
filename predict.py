#!/usr/bin/env python3
"""Stable competition entry point; implementation lives in Workflows.predict."""

from Workflows.predict import FALLBACK_PRED, main, predict_dir

__all__ = ["FALLBACK_PRED", "main", "predict_dir"]


if __name__ == "__main__":
    raise SystemExit(main())
