from theme import PALETTE


def _relative_luminance(color):
    channels = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(foreground, background):
    luminances = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)),
        reverse=True,
    )
    return (luminances[0] + 0.05) / (luminances[1] + 0.05)


def test_palette_text_colors_meet_wcag_aa():
    text_colors = (
        "text",
        "secondary_text",
        "primary",
        "forecast",
        "error",
        "ok",
    )
    backgrounds = ("background", "page_background")

    for foreground in text_colors:
        for background in backgrounds:
            ratio = _contrast_ratio(PALETTE[foreground], PALETTE[background])
            assert ratio >= 4.5, (
                f"{foreground} on {background} has contrast ratio "
                f"{ratio:.2f}:1 (minimum 4.5:1)"
            )
