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
        row_total = int(distribution["groups"][group]["row_count"])
        rows.append(
            f'<text x="145" y="{y + cell / 2 + 5}" text-anchor="end" '
            f'font-family="sans-serif" font-size="13">'
            f'predicted {escape(group)} (n={row_total:,})</text>'
        )
        ratios = distribution["groups"][group]["ratios"]
        counts = distribution["groups"][group]["counts"]
        for column_index, label in enumerate(labels):
            value = float(ratios[label])
            count = int(counts[label])
            x = origin_x + column_index * cell
            rows.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{_heatmap_color(value)}" stroke="white"/>'
            )
            rows.append(
                f'<text x="{x + cell / 2}" y="{y + cell / 2 - 4}" '
                'text-anchor="middle" font-family="sans-serif" font-size="14" '
                f'fill="#111">{value * 100:.1f}%</text>'
            )
            rows.append(
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 15}" '
                'text-anchor="middle" font-family="sans-serif" font-size="12" '
                f'fill="#333">n={count:,}</text>'
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


def write_routerarena_epoch_progress_chart(
    epoch_records: Sequence[Mapping[str, Any]], output_path: Path
) -> Path:
    if not epoch_records:
        raise ValueError("At least one epoch record is required")
    records = sorted(epoch_records, key=lambda row: int(row["epoch_number"]))
    width, height = 860, 460
    left, top, chart_width, chart_height = 90, 70, 700, 300
    metric_styles = {
        "accuracy": ("Accuracy", "#1565C0"),
        "macro_f1": ("Macro F1", "#2E7D32"),
        "under_routing_rate": ("Under-routing", "#C62828"),
    }
    rows: list[str] = []
    for tick in range(0, 6):
        ratio = tick / 5
        y = top + chart_height - ratio * chart_height
        rows.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" '
            'y2="{y:.1f}" stroke="#e0e0e0"/>'
        )
        rows.append(
            f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12">{ratio * 100:.0f}%</text>'
        )
    x_positions: list[float] = []
    denominator = max(1, len(records) - 1)
    for index, record in enumerate(records):
        x = left + chart_width * index / denominator
        x_positions.append(x)
        rows.append(
            f'<text x="{x:.1f}" y="{top + chart_height + 28}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="13">'
            f'Epoch {int(record["epoch_number"])}</text>'
        )
    for metric_index, (key, (label, color)) in enumerate(metric_styles.items()):
        points: list[str] = []
        for x, record in zip(x_positions, records, strict=True):
            value = max(0.0, min(1.0, float(record[key])))
            y = top + chart_height - value * chart_height
            points.append(f"{x:.1f},{y:.1f}")
            rows.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>'
            )
            rows.append(
                f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="11" fill="{color}">'
                f'{value * 100:.1f}%</text>'
            )
        rows.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
            'stroke-width="3"/>'
        )
        legend_x = 180 + metric_index * 190
        rows.append(
            f'<line x1="{legend_x}" y1="425" x2="{legend_x + 28}" y2="425" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        rows.append(
            f'<text x="{legend_x + 36}" y="430" font-family="sans-serif" '
            f'font-size="13">{escape(label)}</text>'
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _svg_document(
            "\n".join(rows),
            width=width,
            height=height,
            title="RouterArena inference-only metrics by training epoch",
        ),
        encoding="utf-8",
    )
    return output_path
