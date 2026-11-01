from typing import Union, List, Dict, Optional, Callable
import pandas as pd
import unicodedata


def get_display_width(text: str) -> int:
    """计算字符串在等宽字体中的显示宽度，考虑中文字符占两格宽"""
    width = 0
    for char in text:
        # 使用 unicodedata.east_asian_width 来判断字符宽度
        eaw = unicodedata.east_asian_width(char)
        if eaw in ('F', 'W'):  # Fullwidth 或 Wide 字符（如中文）
            width += 2
        elif eaw in ('H', 'Na', 'N'):  # Halfwidth、Narrow 或 Neutral 字符
            width += 1
        else:  # Ambiguous 字符，通常按1个宽度计算
            width += 1
    return width


def escape_markdown(text: str) -> str:
    """Escapes the subset of Markdown special characters that can break tables."""
    special = {"|": "\\|", "*": "\\*", "_": "\\_", "`": "\\`"}
    for ch, repl in special.items():
        text = text.replace(ch, repl)
    return text


def _pad(cell: str, width: int, align: str) -> str:
    """Pad *cell* to *width* according to *align* spec (' :--', '--:', ':')."""
    cell_width = get_display_width(cell)
    padding = width - cell_width

    if padding <= 0:
        return cell

    if align == ":--":  # left
        return cell + " " * padding
    elif align == "--:":  # right
        return " " * padding + cell
    else:  # centered
        left_pad = padding // 2
        right_pad = padding - left_pad
        return " " * left_pad + cell + " " * right_pad


def _make_align_row(widths: List[int], aligns: List[str]) -> str:
    """Construct the Markdown alignment row, honouring *widths* so every dash-column matches the content width."""
    bits = []
    for w, a in zip(widths, aligns):
        if a == ":--":  # left
            bits.append(":" + "-" * (w - 1))
        elif a == "--:":  # right
            bits.append("-" * (w - 1) + ":")
        else:  # centered
            if w <= 1:
                bits.append(":")
            else:
                bits.append(":" + "-" * (w - 2) + ":")
    return "| " + " | ".join(bits) + " |"


def dataframe_to_markdown(
    df: Union[pd.DataFrame, List[Dict[str, any]]],
    index: bool = False,
    headers: bool = True,
    columns: Optional[Dict[str, str]] = None,
    alignments: Optional[List[str]] = None,
    float_format: Optional[Union[str, Dict[str, str]]] = None,
    formatters: Optional[Dict[str, Callable]] = None,
    max_colwidth: Optional[Union[int, Dict[str, int]]] = None,
    caption: Optional[str] = None,
    escape_special_chars: bool = False,
    nan_rep: str = "",
    stubs: Optional[List[str]] = None,
    multiindex_style: str = "default",
    ignore_columns: Optional[List[str]] = None,
    pretty: bool = True,
) -> str:
    """Convert a :class:`pandas.DataFrame` to a GitHub‑flavoured Markdown table.

    Args:
        df (Union[pd.DataFrame, List[Dict[str, Any]]]):
            The source data. Lists or list‑like dictionaries will be wrapped into a
            ``DataFrame`` automatically.
        index (bool, optional):
            Whether to include the DataFrame index as the first column. Defaults to
            ``False``.
        headers (bool, optional):
            Output the column header row. Defaults to ``True``.
        columns (Optional[Dict[str, str]], optional):
            Mapping of *old column name* → *new column name* used to rename
            columns prior to rendering.
        alignments (Optional[List[str]], optional):
            A list specifying alignment for each column (and the index column, if
            ``index`` is ``True``). Use ``":--"`` for left, ``"--:"`` for right,
            or ``":"``/``":--:"`` for centre alignment. If *None*, numeric
            columns default to right‑alignment and others to left‑alignment.
        float_format (Optional[Union[str, Dict[str, str]]], optional):
            ``printf``‑style format string applied to *all* floating‑point
            columns, or a per‑column mapping.
        formatters (Optional[Dict[str, Callable]], optional):
            Mapping of *column name* → *callable*. Each callable receives the raw
            cell value and must return a **string**.
        max_colwidth (Optional[Union[int, Dict[str, int]]], optional):
            Maximum width in characters for each column. Longer content is
            truncated with "…". Provide a single ``int`` or a mapping.
        caption (Optional[str], optional):
            If given, rendered as a bold title *above* the table.
        escape_special_chars (bool, optional):
            Escape Markdown special characters (``| * _ ``` etc.). Defaults to
            ``False``.
        nan_rep (str, optional):
            Replacement text for ``NaN``/``None``. Defaults to an empty string.
        stubs (Optional[List[str]], optional):
            List of index labels to emphasise (bold) when ``index`` is ``True``.
        multiindex_style (str, optional):
            Rendering style for a MultiIndex. Must be one of ``"default"``,
            ``"compact"`` (reset index to columns) or ``"merged"`` (merge levels
            with ``"/"``). Defaults to ``"default"``.
        ignore_columns (Optional[List[str]], optional):
            Column names to drop before rendering.
        pretty (bool, optional):
            When *True*, pads each cell with spaces so the *raw* Markdown
            text looks perfectly aligned in monospaced views (terminals, e‑mails
            etc.). Defaults to ``True``.

    Returns:
        str: A Markdown‑formatted table suitable for GitHub, Notion, etc.
    """

    # --- 1. Normalise input -------------------------------------------------
    if isinstance(df, list):
        df = pd.DataFrame(df)
    df_copy = df.copy()

    if ignore_columns:
        df_copy = df_copy.drop(columns=ignore_columns)

    df_copy = df_copy.fillna(nan_rep)

    # Rename columns if mapping provided
    if columns:
        df_copy = df_copy.rename(columns=columns)
    col_names = df_copy.columns.tolist()

    # --- 2. Handle index ----------------------------------------------------
    index_names: List[str] = []
    if index:
        if isinstance(df_copy.index, pd.MultiIndex):
            if multiindex_style == "compact":
                df_copy = df_copy.reset_index()
                index = False
            elif multiindex_style == "merged":
                df_copy.index = [
                    "/".join(map(str, tup)).rstrip("/")
                    for tup in df_copy.index.tolist()
                ]
            else:  # default
                index_names = [
                    name if name is not None else f"Level {i}"
                    for i, name in enumerate(df_copy.index.names)
                ]
        else:
            index_names = [df_copy.index.name if df_copy.index.name else ""]

    # --- 3. Default alignments ---------------------------------------------
    if alignments is None:
        alignments = []
        if index:
            alignments.append(":--")  # index column left‑aligned
        for col in col_names:
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                alignments.append("--:")
            else:
                alignments.append(":--")

    # --- 4. Apply formatting/conversion -------------------------------------
    # Float formatting map
    if isinstance(float_format, str):
        float_format = {
            c: float_format
            for c in col_names
            if pd.api.types.is_float_dtype(df_copy[c])
        }

    for col in col_names:
        # format floats
        if (
            float_format
            and col in float_format
            and pd.api.types.is_float_dtype(df_copy[col])
        ):
            df_copy[col] = df_copy[col].map(
                lambda x: float_format[col] % x if not pd.isna(x) else nan_rep
            )
        # to string early so padding works correctly
        df_copy[col] = df_copy[col].astype(str)

        # column width limit (pre‑escape so we slice plain text)
        # 修改：使用显示宽度来截断文本
        if max_colwidth:
            width = (
                max_colwidth[col] if isinstance(max_colwidth, dict) else max_colwidth
            )
            if width:
                def truncate_by_display_width(s, max_width):
                    if get_display_width(s) <= max_width:
                        return s

                    result = ""
                    current_width = 0
                    for char in s:
                        char_width = get_display_width(char)
                        if current_width + char_width + 3 > max_width:  # 为"..."保留3个宽度
                            break
                        result += char
                        current_width += char_width
                    return result + "..."

                df_copy[col] = df_copy[col].map(
                    lambda s: truncate_by_display_width(s, width)
                )

    # escape markdown reserved chars
    if escape_special_chars:
        df_copy = df_copy.applymap(escape_markdown)
        if index:
            df_copy.index = df_copy.index.map(escape_markdown)

    # custom formatters (run *after* escaping so they can contain MD markup)
    if formatters:
        for col, func in formatters.items():
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].map(func)

    # --- 5. Build the table rows -------------------------------------------
    rows: List[List[str]] = []

    # headers
    if headers:
        header_cells: List[str] = []
        if index:
            header_cells.extend(index_names or [""])
        header_cells.extend([str(c) for c in col_names])
        rows.append(header_cells)

    # data rows
    if index:
        for ix, record in df_copy.iterrows():
            idx_cell = str(ix)
            if stubs and idx_cell in stubs:
                idx_cell = f"**{idx_cell}**"
            row = [idx_cell] + [str(record[c]) for c in col_names]
            rows.append(row)
    else:
        for _, record in df_copy.iterrows():
            rows.append([str(record[c]) for c in col_names])

    # --- 6A. Pretty rendering ----------------------------------------------
    if pretty:
        # 修改：使用显示宽度计算列宽
        widths = [max(get_display_width(r[i]) for r in rows) for i in range(len(rows[0]))]

        lines: List[str] = []
        if caption:
            cap = escape_markdown(caption) if escape_special_chars else caption
            lines.append(f"**{cap}**\n")

        # header row (already first in rows list if headers True)
        if headers:
            header_line = (
                "| "
                + " | ".join(
                    _pad(cell, widths[i], ":--")  # header text always left‑aligned
                    for i, cell in enumerate(rows[0])
                )
                + " |"
            )
            lines.append(header_line)

            # alignment row – uses user‑supplied/default aligns
            lines.append(_make_align_row(widths, alignments))
            data_iter = rows[1:]
        else:
            data_iter = rows

        # data rows
        for r in data_iter:
            line = (
                "| "
                + " | ".join(
                    _pad(cell, widths[i], alignments[i]) for i, cell in enumerate(r)
                )
                + " |"
            )
            lines.append(line)

        return "\n".join(lines)

    # --- 6B. Legacy/compact rendering (no padding) -------------------------
    markdown_lines: List[str] = []
    if caption:
        markdown_lines.append(
            f"**{escape_markdown(caption) if escape_special_chars else caption}**\n"
        )

    # build header & alignment rows (simpler than pretty)
    if headers:
        if index:
            header_txt = (
                "| "
                + ((" | ".join(index_names) + " | ") if index_names else " | ")
                + " | ".join(col_names)
                + " |"
            )
        else:
            header_txt = "| " + " | ".join(col_names) + " |"
        markdown_lines.append(header_txt)

        align_map = {":--": ":---", "--:": "---:", ":": ":---:"}
        markdown_lines.append(
            "| " + " | ".join(align_map.get(a, ":---:") for a in alignments) + " |"
        )

    # data rows
    if index:
        for ix, record in df_copy.iterrows():
            idx_cell = str(ix)
            if stubs and idx_cell in stubs:
                idx_cell = f"**{idx_cell}**"
            markdown_lines.append(
                "| "
                + idx_cell
                + " | "
                + " | ".join(str(record[c]) for c in col_names)
                + " |"
            )
    else:
        for _, record in df_copy.iterrows():
            markdown_lines.append(
                "| " + " | ".join(str(record[c]) for c in col_names) + " |"
            )

    return "\n".join(markdown_lines)
