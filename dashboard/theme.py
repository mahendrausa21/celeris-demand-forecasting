from plotly import graph_objects as go


PALETTE = {
    "text": "#0F172A",
    "secondary_text": "#475569",
    "background": "#FFFFFF",
    "page_background": "#F8FAFC",
    "primary": "#0F766E",
    "forecast": "#B45309",
    "error": "#B91C1C",
    "ok": "#15803D",
    "grid": "#E2E8F0",
}


def css():
    text_rgb = ", ".join(
        str(int(PALETTE["text"][index:index + 2], 16))
        for index in (1, 3, 5)
    )
    return f"""
    <style>
    :root {{ color-scheme: light; }}
    #MainMenu, footer {{ visibility: hidden; }}
    [data-testid="stAppViewContainer"] {{
        background: {PALETTE["page_background"]};
        color: {PALETTE["text"]};
    }}
    [data-testid="stSidebar"] {{
        background: {PALETTE["page_background"]};
        border-right: 1px solid {PALETTE["grid"]};
    }}
    [data-testid="stCaptionContainer"] {{
        color: {PALETTE["secondary_text"]};
        font-size: 14px;
    }}
    .main .block-container {{
        max-width: 1440px;
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
    }}
    [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3 {{
        border-left: 4px solid {PALETTE["primary"]};
        padding-left: .65rem;
        margin-top: 1.6rem;
    }}
    .product-header {{
        background: linear-gradient(
            120deg,
            {PALETTE["primary"]},
            color-mix(in srgb, {PALETTE["primary"]} 82%, {PALETTE["text"]})
        );
        border-radius: 18px;
        color: {PALETTE["background"]};
        margin-bottom: 1.25rem;
        padding: 1.5rem 1.75rem;
    }}
    .product-header h1 {{
        color: {PALETTE["background"]};
        font-size: 2rem;
        margin: 0 0 .45rem;
    }}
    .product-header p {{
        color: {PALETTE["background"]};
        font-size: 1rem;
        margin: 0;
    }}
    .product-header .data-range {{
        font-size: .875rem;
        font-weight: 600;
        margin-top: .9rem;
    }}
    .kpi-card {{
        background: {PALETTE["background"]};
        border: 1px solid {PALETTE["grid"]};
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba({text_rgb}, .08);
        min-height: 118px;
        padding: 1.1rem 1.25rem;
    }}
    .kpi-label {{
        align-items: center;
        color: {PALETTE["secondary_text"]};
        display: flex;
        font-size: .875rem;
        font-weight: 600;
        gap: .45rem;
        margin-bottom: .45rem;
    }}
    .kpi-icon {{ font-size: 1rem; }}
    .kpi-value {{
        color: {PALETTE["text"]};
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.15;
    }}
    .kpi-delta {{
        color: {PALETTE["primary"]};
        font-size: .875rem;
        font-weight: 600;
        margin-top: .55rem;
    }}
    </style>
    """


def plotly_template():
    secondary = PALETTE["secondary_text"]
    return go.layout.Template(
        layout=go.Layout(
            colorway=[
                PALETTE["primary"],
                PALETTE["forecast"],
                secondary,
                PALETTE["error"],
                PALETTE["ok"],
            ],
            font=dict(family="Arial, sans-serif", size=13, color=PALETTE["text"]),
            title=dict(font=dict(size=16, color=PALETTE["text"])),
            xaxis=dict(
                gridcolor=PALETTE["grid"],
                title=dict(font=dict(size=13, color=secondary)),
                tickfont=dict(size=12, color=secondary),
                zerolinecolor=PALETTE["grid"],
            ),
            yaxis=dict(
                gridcolor=PALETTE["grid"],
                title=dict(font=dict(size=13, color=secondary)),
                tickfont=dict(size=12, color=secondary),
                zerolinecolor=PALETTE["grid"],
            ),
            legend=dict(font=dict(size=12, color=PALETTE["text"])),
            paper_bgcolor=PALETTE["background"],
            plot_bgcolor=PALETTE["background"],
            hoverlabel=dict(
                bgcolor=PALETTE["background"],
                font=dict(color=PALETTE["text"], size=12),
            ),
        )
    )
