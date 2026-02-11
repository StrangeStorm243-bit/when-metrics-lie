from __future__ import annotations


def short(s: str | None, n: int = 12) -> str:
    """Truncate a string to n characters, or return '-' if None."""
    if s is None:
        return "-"
    if len(s) <= n:
        return s
    return s[:n] + "..."


def format_table(rows: list[list[str]], headers: list[str]) -> str:
    """
    Format a table with headers and rows.
    Computes column widths and left-aligns all cells.
    Returns a single string with newlines.
    """
    if not headers:
        return ""

    if not rows:
        # Return just headers with separator
        widths = [len(h) for h in headers]
        header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
        return header_line

    # Compute column widths (max of header and all row values)
    num_cols = len(headers)
    widths = [len(h) for h in headers]

    for row in rows:
        if len(row) != num_cols:
            # Pad row if needed
            row.extend([""] * (num_cols - len(row)))
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))

    # Build output lines
    lines = []

    # Header
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    lines.append(header_line)

    # Separator
    separator = "  ".join("-" * w for w in widths)
    lines.append(separator)

    # Rows
    for row in rows:
        # Ensure row has correct length
        padded_row = row + [""] * (num_cols - len(row))
        row_line = "  ".join(str(cell).ljust(w) for cell, w in zip(padded_row, widths))
        lines.append(row_line)

    return "\n".join(lines)
