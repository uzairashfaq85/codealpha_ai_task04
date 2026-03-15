"""Compatibility entry point for video recognition.

Created: Aug 2024
Purpose: Keeps root-level command compatibility while delegating to src package.
"""

from src.scripts.recognition import main, parse_args


if __name__ == "__main__":
    main(parse_args())
