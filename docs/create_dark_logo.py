#!/usr/bin/env python3
"""Create a dark mode version of the logo by selectively replacing colors."""

import re
from pathlib import Path


# Manual color mappings for dark mode
COLOR_MAP = {
    "#2D303B": "#FDFDFD",
    "#FDFDFD": "#2D303B",
    "#2E313D": "#FDFDFD",
}


def convert_color_for_dark_mode(hex_color: str) -> str:
    """Convert a color for dark mode using manual mappings.

    Colors not in the mapping are kept unchanged.
    """
    # Normalize to uppercase for comparison
    normalized = hex_color.upper()

    # Return mapped color or original if not in map
    return COLOR_MAP.get(normalized, hex_color)


def process_svg(input_path: Path, output_path: Path) -> None:
    """Convert light mode SVG to dark mode by selectively replacing colors."""
    print(f"Reading {input_path}...")
    content = input_path.read_text()

    # Find all fill and stroke color attributes
    # Pattern matches: fill="#RRGGBB" or stroke="#RRGGBB"
    pattern = r'(fill|stroke)="#([0-9A-Fa-f]{6})"'

    # Track color changes for reporting
    color_changes = {}

    def replace_color(match: re.Match) -> str:
        attr_name = match.group(1)  # 'fill' or 'stroke'
        original_color = f"#{match.group(2)}"  # hex color with #
        new_color = convert_color_for_dark_mode(original_color)

        # Track changes
        if original_color != new_color:
            if original_color not in color_changes:
                color_changes[original_color] = new_color

        return f'{attr_name}="{new_color}"'

    # Replace all colors
    modified_content = re.sub(pattern, replace_color, content)

    # Report changes
    print("\nColor replacements:")
    for original, new in sorted(color_changes.items()):
        print(f"  {original} -> {new}")

    print(f"\nTotal unique colors changed: {len(color_changes)}")

    # Write output
    print(f"\nWriting {output_path}...")
    output_path.write_text(modified_content)
    print("Done!")


if __name__ == "__main__":
    # Paths relative to script location
    script_dir = Path(__file__).parent
    static_dir = script_dir / "_static"

    light_logo = static_dir / "logo_light.svg"
    dark_logo = static_dir / "logo_dark.svg"

    if not light_logo.exists():
        print(f"Error: {light_logo} not found!")
        exit(1)

    process_svg(light_logo, dark_logo)
    print(f"\nDark mode logo created at: {dark_logo}")
    print("Preview by opening it in a browser or image viewer.")
    print("\nThe colorful elements (planets, stars, orbits) should remain vibrant,")
    print("while the background became dark and text became light.")
