"""Compatibility entry point for labeled image visualization.

Created: Aug 2024
Purpose: Keeps root-level command compatibility while delegating to src package.
"""

from src.scripts.recognition_videos import main


if __name__ == "__main__":
    main()
