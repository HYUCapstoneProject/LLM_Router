from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence


COLORS = {
    "easy": "#4CAF50",
    "medium": "#FFB300",
    "hard": "#E53935",
}


def _color(label: str, fallback_index: int) -> str:
    fallback = ("#4C78A8", "#F58518", "#E45756")
    return COLORS.get(label, fallback[fallback_index % len(fallback)])


def _svg_document(body: str, *, width: int, height: int, title: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        f'<text x="{width / 2}" y="28" text-anchor="middle" '
        f'font-family="sans-serif" font-size="18" font-weight="bold">'
        f'{escape(title)}</text>\n{body}\n</svg>\n'
    )


def _stacked_bar_svg(
    distribution: Mapping[str, Any], labels: Sequence[str]
) -> str:
    width, height = 820, 250
    chart_x, chart_width = 150, 600
    rows: list[str] = []
    for row_index, group in enumerate(labels):
        y = 65 + row_index * 48
        rows.append(
            f'<text x="140" y="{y + 18}" text-anchor="end" '
            f'font-family="sans-serif" font-size="14">predicted {escape(group)}</text>'
        )
        ratios = distribution["groups"][group]["ratios"]
        cursor = float(chart_x)
        for label_index, label in enumerate(labels):
            ratio = float(ratios[label])
            segment_width = ratio * chart_width
            rows.append(
                f'<rect x="{cursor:.2f}" y="{y}" width="{segment_width:.2f}" '
                f'height="28" fill="{_color(label, label_index)}"/>'
            )
            if segment_width >= 42:
                rows.append(
                    f'<text x="{cursor + segment_width / 2:.2f}" y="{y + 19}" '
                    'text-anchor="middle" font-family="sans-serif" font-size="12" '
                    f'fill="white">{ratio * 100:.1f}%</text>'
                )
            cursor += segment_width
    for label_index, label in enumerate(labels):
        x = 170 + label_index * 170
        rows.append(
            f'<rect x="{x}" y="218" width="14" height="14" '
            f'fill="{_color(label, label_index)}"/>'
        )
        rows.append(
            f'<text x="{x + 20}" y="230" font-family="sans-serif" '
            f'font-size="12">RouterArena {escape(label)}</text>'
        )
    return _svg_document(
        "\n".join(rows),
        width=width,
        height=height,
        title="RouterArena labels within predicted groups",
    )


def _heatmap_color(value: float) -> str:
    bounded = max(0.0, min(1.0, value))
    red = int(245 - 155 * bounded)
    green = int(249 - 115 * bounded)
    blue = int(255 - 55 * bounded)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _heatmap_svg(distribution: Mapping[str, Any], labels: Sequence[str]) -> str:
    width, height = 520, 420
    origin_x, origin_y, cell = 155, 95, 90
    rows: list[str] = []
    for column_index, label in enumerate(labels):
        x = origin_x + column_index * cell + cell / 2
        rows.append(
            f'<text x="{x}" y="78" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13">{escape(label)}</text>'
        )
    for row_index, group in enumerate(labels):
        y = origin_y + row_index * cell
        rows.append(
            f'<text x="145" y="{y + cell / 2 + 5}" text-anchor="end" '
            f'font-family="sans-serif" font-size="13">predicted {escape(group)}</text>'
        )
        ratios = distribution["groups"][group]["ratios"]
        for column_index, label in enumerate(labels):
            value = float(ratios[label])
            x = origin_x + column_index * cell
            rows.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{_heatmap_color(value)}" stroke="white"/>'
            )
            rows.append(
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 5}" '
                'text-anchor="middle" font-family="sans-serif" font-size="14" '
                f'fill="#111">{value * 100:.1f}%</text>'
            )
    rows.append(
        '<text x="290" y="390" text-anchor="middle" font-family="sans-serif" '
        'font-size="12">columns: RouterArena official label</text>'
    )
    return _svg_document(
        "\n".join(rows),
        width=width,
        height=height,
        title="Row-normalized predicted-group heatmap",
    )


def write_routerarena_distribution_charts(
    metrics: Mapping[str, Any], output_dir: Path
) -> tuple[Path, Path]:
    labels = tuple(str(label) for label in metrics["label_order"])
    distribution = metrics["routerarena_labels_within_predicted_groups"]
    output_dir.mkdir(parents=True, exist_ok=True)
    stacked_path = output_dir / "predicted_group_distribution.svg"
    heatmap_path = output_dir / "predicted_group_heatmap.svg"
    stacked_path.write_text(
        _stacked_bar_svg(distribution, labels), encoding="utf-8"
    )
    heatmap_path.write_text(_heatmap_svg(distribution, labels), encoding="utf-8")
    return stacked_path, heatmap_path
