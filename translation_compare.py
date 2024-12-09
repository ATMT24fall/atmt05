def compare_translations(original_file, new_file, output_file):
    """Compare two translation files and save differences to output file."""
    with open(original_file, "r", encoding="utf-8") as f1, open(
        new_file, "r", encoding="utf-8"
    ) as f2:
        original = f1.readlines()
        new = f2.readlines()

    differences = []
    for i, (orig_line, new_line) in enumerate(zip(original, new), 1):
        if orig_line.strip() != new_line.strip():
            differences.append(
                {
                    "line_number": i,
                    "original": orig_line.strip(),
                    "new": new_line.strip(),
                }
            )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Found {len(differences)} differences:\n\n")
        for diff in differences:
            f.write(f"Line {diff['line_number']}:\n")
            f.write(f"Original: {diff['original']}\n")
            f.write(f"New:      {diff['new']}\n\n")


# Usage example
compare_translations(
    "assignments/05/beamsize3/translations.txt",
    "assignments/05/beamsize3_constant_prune/translations.txt",
    "differences_constant_prune.txt",
)
