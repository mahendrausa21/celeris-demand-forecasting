import html
import json
import math
from pathlib import Path
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"

# Known future price policy, applied cumulatively to each SKU's last listed price.
JAN_2027_PRICE_RISE = 1.045
JAN_2028_PRICE_RISE = 1.04

PRIMARY = "#0F766E"
FORECAST = "#D97706"
SLATE = "#475569"
ERROR = "#DC2626"
OK = "#16A34A"
CHART_COLORS = [PRIMARY, FORECAST, SLATE, ERROR, OK]

st.set_page_config(page_title="Celeris Demand Forecasting", layout="wide")
st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    [data-testid="stAppViewContainer"] {background: #F8FAFC;}
    .main .block-container {max-width: 1440px; padding-top: 1.6rem; padding-bottom: 2.5rem;}
    .product-header {
        background: linear-gradient(120deg, #FFFFFF 0%, #F0FDFA 100%);
        border: 1px solid #CBD5E1; border-radius: 18px;
        padding: 1.25rem 1.5rem; margin-bottom: 1.25rem;
    }
    .product-header h1 {color: #0F172A; font-size: 1.8rem; margin: 0 0 .25rem 0;}
    .product-header p {color: #475569; font-size: 1rem; margin: 0;}
    .kpi-card {
        background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, .07);
        padding: 1.1rem 1.25rem; min-height: 112px;
    }
    .kpi-label {color: #64748B; font-size: .82rem; font-weight: 600; margin-bottom: .35rem;}
    .kpi-value {color: #0F172A; font-size: 1.8rem; font-weight: 700; line-height: 1.15;}
    .kpi-delta {font-size: .78rem; font-weight: 600; margin-top: .55rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="product-header">
      <h1>Celeris Demand Forecasting &amp; Planning Control Tower</h1>
      <p>Monitor demand signals, forecast performance, and data quality in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("History: Jan 2019 - Oct 2026  |  Forecast: Nov 2026 - May 2028")


def render_kpi_card(column, label, value, delta=None):
    delta_html = ""
    if delta:
        delta_html = f'<div class="kpi-delta" style="color: {PRIMARY};">{html.escape(delta)}</div>'
    column.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{html.escape(label)}</div>
          <div class="kpi-value">{html.escape(value)}</div>
          {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_template(
    fig,
    unit,
    value_format=",.0f",
    hovermode="closest",
    secondary_y_format=None,
):
    horizontal_bars = any(
        getattr(trace, "type", None) == "bar"
        and getattr(trace, "orientation", None) == "h"
        for trace in fig.data
    )
    fig.update_layout(
        template="plotly_white",
        colorway=CHART_COLORS,
        font=dict(family="Arial, sans-serif", size=12, color="#0F172A"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=24, r=20, t=68, b=36),
        hovermode=hovermode,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text="",
        ),
    )
    fig.update_xaxes(
        showgrid=horizontal_bars,
        gridcolor="#E2E8F0",
        gridwidth=1,
        showline=False,
        zeroline=False,
        title_font=dict(size=12, color=SLATE),
        tickfont=dict(size=11, color=SLATE),
    )
    fig.update_yaxes(
        showgrid=not horizontal_bars,
        gridcolor="#E2E8F0",
        gridwidth=1,
        showline=False,
        zeroline=False,
        tickformat=None if horizontal_bars else (",.1%" if value_format == ".1%" else value_format),
        title_font=dict(size=12, color=SLATE),
        tickfont=dict(size=11, color=SLATE),
    )
    if horizontal_bars:
        fig.update_xaxes(tickformat=",.1%" if value_format == ".1%" else value_format)
    if secondary_y_format is not None:
        fig.update_yaxes(tickformat=secondary_y_format, secondary_y=True)
    for trace in fig.data:
        if trace.hovertemplate is None:
            label = trace.name or "Value"
            unit_suffix = "" if value_format.endswith("%") else f" {html.escape(unit)}"
            if horizontal_bars and getattr(trace, "orientation", None) == "h":
                trace.hovertemplate = (
                    f"<b>%{{y}}</b><br>{html.escape(label)}: "
                    f"%{{x:{value_format}}}{unit_suffix}<extra></extra>"
                )
            else:
                trace.hovertemplate = (
                    f"<b>%{{x}}</b><br>{html.escape(label)}: "
                    f"%{{y:{value_format}}}{unit_suffix}<extra></extra>"
                )
    return fig


def render_plotly_chart(
    fig,
    unit,
    value_format=",.0f",
    hovermode="closest",
    secondary_y_format=None,
):
    chart_template(fig, unit, value_format, hovermode, secondary_y_format)
    title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else "Dashboard chart"
    st.plotly_chart(
        fig,
        width="stretch",
        alt=f"{title}; values shown in {unit}",
    )


def table_config(df, percentage_columns=(), wide_columns=()):
    config = {}
    for column in df.columns:
        if column in percentage_columns:
            continue
        if column in wide_columns:
            config[column] = st.column_config.TextColumn(width="large")
        elif pd.api.types.is_numeric_dtype(df[column]):
            config[column] = st.column_config.NumberColumn(format="localized")
    return config


def format_table(df, percentage_columns=()):
    return df.style.format(
        {column: "{:.1%}" for column in percentage_columns if column in df.columns}
    )


@st.cache_data
def load_history():
    return pd.read_parquet(OUTPUTS_DIR / "clean_history.parquet")


@st.cache_data
def load_forecast():
    forecast = pd.read_csv(OUTPUTS_DIR / "forecast.csv")
    test = pd.read_parquet(OUTPUTS_DIR / "clean_test.parquet")
    forecast["Date"] = pd.to_datetime(forecast["Date"])
    forecast = forecast.merge(
        test[
            [
                "Date",
                "SKU_ID",
                "Region",
                "Product_Family",
                "Customer_Name",
                "Sales_Channel",
                "After_EOL",
                "New_Series",
            ]
        ],
        on=["Date", "SKU_ID", "Region"],
        how="left",
        validate="many_to_one",
    )
    return forecast


@st.cache_data
def load_prices():
    return pd.read_parquet(OUTPUTS_DIR / "clean_prices.parquet")


@st.cache_data
def load_sku_master():
    return pd.read_parquet(OUTPUTS_DIR / "clean_sku_master.parquet")


@st.cache_data
def load_backtest():
    return pd.read_csv(OUTPUTS_DIR / "backtest_metrics.csv")


@st.cache_data
def load_dq_report():
    with open(OUTPUTS_DIR / "dq_report.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_anomaly_register():
    path = OUTPUTS_DIR / "anomaly_register.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_unconstrained_history():
    path = OUTPUTS_DIR / "unconstrained_history.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data
def load_macro_leadlag_grid():
    return pd.read_csv(OUTPUTS_DIR / "macro_leadlag_grid.csv")


@st.cache_data
def load_macro_leadlag():
    return pd.read_csv(OUTPUTS_DIR / "macro_leadlag.csv")


@st.cache_data
def load_macro_sensitivity():
    return pd.read_csv(OUTPUTS_DIR / "macro_sensitivity.csv")


@st.cache_data
def load_ev_shift():
    ev_shift = pd.read_csv(OUTPUTS_DIR / "ev_shift.csv")
    ev_shift["Date"] = pd.to_datetime(ev_shift["Date"])
    return ev_shift


@st.cache_data
def load_series_profile():
    return pd.read_csv(OUTPUTS_DIR / "series_profile.csv")


@st.cache_data
def load_forecast_ranges_aggregate():
    ranges = pd.read_csv(OUTPUTS_DIR / "forecast_ranges_aggregate.csv")
    ranges["Date"] = pd.to_datetime(ranges["Date"])
    return ranges


@st.cache_data
def load_forecast_ranges():
    ranges = pd.read_csv(OUTPUTS_DIR / "forecast_ranges.csv")
    ranges["Date"] = pd.to_datetime(ranges["Date"])
    return ranges


@st.cache_data
def load_range_evaluation():
    return pd.read_csv(OUTPUTS_DIR / "range_evaluation.csv")


@st.cache_data
def load_launch_ramp():
    return pd.read_csv(OUTPUTS_DIR / "launch_ramp.csv")


@st.cache_data
def load_stockouts():
    stockouts = pd.read_parquet(OUTPUTS_DIR / "clean_stockouts.parquet")
    stockouts["Date"] = pd.to_datetime(stockouts["Date"])
    return stockouts


@st.cache_data
def load_forecast_source():
    return pd.read_csv(OUTPUTS_DIR / "forecast.csv")


@st.cache_data
def load_submission():
    return pd.read_csv(OUTPUTS_DIR / "submission.csv")


@st.cache_data
def load_planner_exceptions():
    return pd.read_csv(OUTPUTS_DIR / "planner_exceptions.csv")


def normalize_dates(df, column="Date"):
    df = df.copy()
    df[column] = pd.to_datetime(df[column])
    df["Month"] = df[column].dt.to_period("M").dt.to_timestamp()
    return df


def add_forecast_revenue(forecast, prices):
    forecast = normalize_dates(forecast)
    latest_prices = (
        prices.assign(Date=pd.to_datetime(prices["Date"]))
        .sort_values("Date")
        .groupby("SKU_ID", as_index=False)
        .tail(1)[["SKU_ID", "List_Price_USD"]]
        .rename(columns={"List_Price_USD": "Last_List_Price_USD"})
    )
    forecast = forecast.merge(latest_prices, on="SKU_ID", how="left")
    forecast["Future_Price_USD"] = forecast["Last_List_Price_USD"]
    forecast.loc[forecast["Month"] >= pd.Timestamp("2027-01-01"), "Future_Price_USD"] *= JAN_2027_PRICE_RISE
    forecast.loc[forecast["Month"] >= pd.Timestamp("2028-01-01"), "Future_Price_USD"] *= JAN_2028_PRICE_RISE
    forecast["Forecast_Revenue_USD"] = forecast["Units_Forecast"] * forecast["Future_Price_USD"]
    return forecast


def filter_multiselect(df, selections, column):
    if selections:
        return df[df[column].isin(selections)].copy()
    return df


def apply_available_filters(df, selections):
    for column, values in selections.items():
        if values and column in df.columns:
            df = filter_multiselect(df, values, column)
    return df


def apply_shared_filters(history, forecast, selections):
    history_filtered = history.copy()
    forecast_filtered = forecast.copy()
    for column, values in selections.items():
        history_filtered = filter_multiselect(history_filtered, values, column)
        forecast_filtered = filter_multiselect(forecast_filtered, values, column)
    return history_filtered, forecast_filtered


def money(value):
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"
    if abs_value >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.0f}"


def compact_units(value):
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:,.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:,.1f}K"
    return f"{value:,.0f}"


def monthly_history(df):
    df = normalize_dates(df)
    units = df.groupby("Month", as_index=False)["Units_Sold"].sum(min_count=1)
    revenue = df.groupby("Month", as_index=False)["Revenue_USD"].sum(min_count=1)
    return units.merge(revenue, on="Month", how="outer").sort_values("Month")


def monthly_forecast(df):
    df = normalize_dates(df)
    units = df.groupby("Month", as_index=False)["Units_Forecast"].sum(min_count=1)
    revenue = df.groupby("Month", as_index=False)["Forecast_Revenue_USD"].sum(min_count=1)
    return units.merge(revenue, on="Month", how="outer").sort_values("Month")


def monthly_unconstrained(df):
    df = normalize_dates(df)
    observed = df.groupby("Month", as_index=False)["Units_Observed"].sum(min_count=1)
    unconstrained = df.groupby("Month", as_index=False)["Units_Unconstrained"].sum(min_count=1)
    return observed.merge(unconstrained, on="Month", how="outer").sort_values("Month")


def selected_forecast_ranges(selections):
    family = selections["Product_Family"]
    sku = selections["SKU_ID"]
    region = selections["Region"]
    other_filters = selections["Sales_Channel"] or selections["Customer_Name"]

    if other_filters:
        return None
    if not family and not sku and not region:
        ranges = load_forecast_ranges_aggregate()
        selected = ranges[(ranges["Level"] == "total") & (ranges["Group"] == "All")]
    elif len(family) == 1 and not sku and not region:
        ranges = load_forecast_ranges_aggregate()
        selected = ranges[(ranges["Level"] == "family") & (ranges["Group"] == family[0])]
    elif not family and not sku and len(region) == 1:
        ranges = load_forecast_ranges_aggregate()
        selected = ranges[(ranges["Level"] == "region") & (ranges["Group"] == region[0])]
    elif len(family) == 1 and not sku and len(region) == 1:
        ranges = load_forecast_ranges_aggregate()
        group = f"{family[0]} | {region[0]}"
        selected = ranges[
            (ranges["Level"] == "family_region") & (ranges["Group"] == group)
        ]
    elif not family and len(sku) == 1 and len(region) == 1:
        ranges = load_forecast_ranges()
        selected = ranges[
            (ranges["SKU_ID"] == sku[0]) & (ranges["Region"] == region[0])
        ]
    else:
        return None
    return selected[["Date", "P10", "P90"]].sort_values("Date")


def add_forecast_region(fig, forecast_start):
    fig.add_vrect(
        x0=forecast_start,
        x1=pd.Timestamp("2028-05-31"),
        fillcolor=FORECAST,
        opacity=0.08,
        line_width=0,
        annotation_text="Forecast",
        annotation_position="top left",
    )


def plot_timeline(
    history_monthly,
    forecast_monthly,
    metric,
    title,
    unconstrained_monthly=None,
    forecast_band=None,
):
    history_label = "Units" if metric == "Units_Sold" else "Revenue (USD)"
    forecast_label = "Forecast Units" if metric == "Units_Sold" else "Forecast Revenue"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_monthly["Month"],
            y=history_monthly[metric],
            mode="lines",
            name=history_label,
            line=dict(color=PRIMARY, width=3),
            connectgaps=False,
        )
    )
    if metric == "Units_Sold" and forecast_band is not None and not forecast_band.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast_band["Date"],
                y=forecast_band["P10"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                connectgaps=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_band["Date"],
                y=forecast_band["P90"],
                mode="lines",
                name="P10-P90 range",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(217, 119, 6, 0.16)",
                hovertemplate=(
                    "Month: %{x|%b %Y}<br>P90: %{y:,.0f} units<extra></extra>"
                ),
                connectgaps=False,
            )
        )
    forecast_metric = "Units_Forecast" if metric == "Units_Sold" else "Forecast_Revenue_USD"
    fig.add_trace(
        go.Scatter(
            x=forecast_monthly["Month"],
            y=forecast_monthly[forecast_metric],
            mode="lines",
            name=forecast_label,
            line=dict(color=FORECAST, width=3, dash="dash"),
            connectgaps=False,
        )
    )
    if unconstrained_monthly is not None and not unconstrained_monthly.empty:
        if metric == "Units_Sold":
            fig.add_trace(
                go.Scatter(
                    x=unconstrained_monthly["Month"],
                    y=unconstrained_monthly["Units_Unconstrained"],
                    mode="lines",
                    name="Unconstrained demand",
                    line=dict(color=SLATE, width=3),
                    connectgaps=False,
                )
            )
    if not forecast_monthly.empty:
        add_forecast_region(fig, forecast_monthly["Month"].min())
    fig.add_vline(
        x=pd.Timestamp("2026-11-01"),
        line_color=FORECAST,
        line_dash="dot",
        line_width=2,
        annotation_text="Nov 2026",
        annotation_position="top",
    )
    fig.update_layout(
        title=title,
        height=380,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=70, b=20),
        yaxis_title=history_label,
    )
    fig.update_xaxes(title="Month")
    return fig


with st.spinner("Loading history, forecast, and planning data..."):
    history = normalize_dates(load_history())
    forecast = add_forecast_revenue(load_forecast(), load_prices())
    sku_master = load_sku_master()
    unconstrained = load_unconstrained_history()
if unconstrained is not None:
    unconstrained = normalize_dates(unconstrained)

st.sidebar.title("Celeris Control Tower")
page = st.sidebar.radio(
    "Navigate",
    [
        "About this project",
        "Executive overview",
        "Data health",
        "Demand explorer",
        "Anomaly monitor",
        "Forecast & accuracy",
        "Procurement plan",
        "Planner exceptions",
        "Stockouts & lost sales",
        "Lifecycle",
        "Data download",
        "Macro drivers",
        "What-if",
        "EV shift",
    ],
)

st.sidebar.subheader("Filters")


def ordered_options(df, column):
    values = sorted([value for value in df[column].dropna().unique().tolist()], key=str)
    return values


family_options = sorted(set(ordered_options(history, "Product_Family")) | set(ordered_options(forecast, "Product_Family")))
sku_options = sorted(set(ordered_options(history, "SKU_ID")) | set(ordered_options(forecast, "SKU_ID")))
region_options = sorted(set(ordered_options(history, "Region")) | set(ordered_options(forecast, "Region")))
channel_options = sorted(set(ordered_options(history, "Sales_Channel")) | set(ordered_options(forecast, "Sales_Channel")))
customer_options = sorted(set(ordered_options(history, "Customer_Name")) | set(ordered_options(forecast, "Customer_Name")))

def reset_shared_filters():
    for key in ["shared_family", "shared_sku", "shared_region", "shared_channel", "shared_customer"]:
        st.session_state[key] = []


with st.sidebar:
    st.button("Reset filters", on_click=reset_shared_filters, key="reset_shared_filters", width="stretch")
    selected_family = st.multiselect("Product_Family", family_options, key="shared_family")
    selected_sku = st.multiselect("SKU_ID", sku_options, key="shared_sku")
    selected_region = st.multiselect("Region", region_options, key="shared_region")
    selected_channel = st.multiselect("Sales_Channel", channel_options, key="shared_channel")
    selected_customer = st.multiselect("Customer_Name", customer_options, key="shared_customer")

selections = {
    "Product_Family": selected_family,
    "SKU_ID": selected_sku,
    "Region": selected_region,
    "Sales_Channel": selected_channel,
    "Customer_Name": selected_customer,
}
history_filtered, forecast_filtered = apply_shared_filters(history, forecast, selections)
unconstrained_filtered = unconstrained.copy() if unconstrained is not None else None
if unconstrained_filtered is not None:
    unconstrained_filtered = filter_multiselect(unconstrained_filtered, selected_family, "Product_Family")
    unconstrained_filtered = filter_multiselect(unconstrained_filtered, selected_sku, "SKU_ID")
    unconstrained_filtered = filter_multiselect(unconstrained_filtered, selected_region, "Region")
    unconstrained_filtered = filter_multiselect(unconstrained_filtered, selected_channel, "Sales_Channel")
    unconstrained_filtered = filter_multiselect(unconstrained_filtered, selected_customer, "Customer_Name")

matching_series = forecast_filtered[["SKU_ID", "Region"]].drop_duplicates().shape[0]
st.sidebar.caption(f"{matching_series:,} series match the current selection")
if history_filtered.empty and forecast_filtered.empty:
    st.info("No series match the current filters. Adjust a filter or use Reset filters.")


if page == "About this project":
    st.header("About this project")
    st.subheader("The problem")
    st.write(
        "Reported shipments can understate demand when stockouts, data gaps, customer changes, "
        "and product lifecycle events distort the observed history. This project prepares that "
        "history for planning and produces SKU-region demand forecasts with accuracy and range checks."
    )

    st.subheader("The five steps")
    steps = [
        ("Cleaning", "Standardize and repair sales, price, stockout, and product records."),
        ("Anomalies", "Identify data errors, demand gaps, and real supply or customer events."),
        ("Drivers", "Review family-level relationships between demand and external indicators."),
        ("Forecast", "Generate SKU-region forecasts and evaluate accuracy and uncertainty."),
        ("Dashboard", "Explore demand, exceptions, lifecycle signals, and procurement plans."),
    ]
    for column, (title, description) in zip(st.columns(len(steps)), steps):
        with column:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(description)

    about_dq = load_dq_report()
    about_anomalies = load_anomaly_register()
    about_backtest = load_backtest()
    about_ranges = load_range_evaluation()
    about_launch_ramp = load_launch_ramp()
    final_model = forecast["Model"].dropna().astype(str).iloc[0]
    overall_metrics = (
        about_backtest[
            (about_backtest["variant"] == "rounded")
            & (about_backtest["group_type"] == "overall")
        ]
        .groupby("model", as_index=False)
        .agg(WAPE=("WAPE", "mean"), Bias=("Bias", "mean"), Rows=("Rows", "sum"))
        .sort_values("WAPE")
    )
    final_metrics = overall_metrics[overall_metrics["model"] == final_model].iloc[0]
    series_coverage = about_ranges[
        (about_ranges["Level"] == "series")
        & (about_ranges["Group"] == "overall")
    ]["Coverage_P10_P90"].iloc[0]
    clean_history_rows = about_dq["files"]["train"][1]
    anomaly_count = len(about_anomalies) if about_anomalies is not None else 0
    forecast_series_count = forecast[["SKU_ID", "Region"]].drop_duplicates().shape[0]
    forecast_units = float(forecast["Units_Forecast"].sum())

    st.subheader("At a glance")
    kpis = st.columns(5)
    render_kpi_card(kpis[0], "Clean history rows", f"{clean_history_rows:,}")
    render_kpi_card(kpis[1], "Anomaly records", f"{anomaly_count:,}")
    render_kpi_card(
        kpis[2],
        "Forecast SKU-regions",
        f"{forecast_series_count:,}",
        f"{compact_units(forecast_units)} forecast units",
    )
    render_kpi_card(
        kpis[3],
        f"{final_model} WAPE",
        f"{final_metrics['WAPE']:.1%}",
        f"Bias: {final_metrics['Bias']:+.1%}",
    )
    render_kpi_card(kpis[4], "Held-out P10-P90 coverage", f"{series_coverage:.1%}")

    st.subheader("Model comparison")
    st.caption("WAPE and Bias are averaged across the rounded overall backtest results.")
    comparison = overall_metrics.rename(
        columns={"model": "Model", "Rows": "Backtest rows"}
    )
    comparison["Model"] = comparison["Model"].apply(
        lambda model: f"Ablation step {model[len('ablation '):]}"
        if model.startswith("ablation ")
        else model
    )
    comparison = comparison[["Model", "WAPE", "Bias", "Backtest rows"]]
    comparison_styled = format_table(comparison, ("WAPE", "Bias")).apply(
        lambda row: [
            "background-color: #CCFBF1; font-weight: bold"
            if row["Model"] == final_model
            else ""
            for _ in row
        ],
        axis=1,
    )
    comparison_config = table_config(
        comparison,
        percentage_columns=("WAPE", "Bias"),
        wide_columns=("Model",),
    )
    st.dataframe(
        comparison_styled,
        column_config=comparison_config,
        width="stretch",
        hide_index=True,
        alt=f"WAPE and Bias comparison by model; {final_model} is the final model",
    )

    st.subheader("Launch ramp")
    ramp_table = about_launch_ramp.copy()
    ramp_table["Median ratio"] = ramp_table["Ratio_Median"].map("{:.1%}".format)
    ramp_table["P25-P75 range"] = ramp_table.apply(
        lambda row: f"{row['Ratio_P25']:.1%} - {row['Ratio_P75']:.1%}",
        axis=1,
    )
    ramp_table = ramp_table[
        ["Phase", "Median ratio", "P25-P75 range", "Pairs", "SKU_Pairs"]
    ]
    st.dataframe(
        ramp_table,
        column_config=table_config(ramp_table, wide_columns=("Phase", "P25-P75 range")),
        width="stretch",
        hide_index=True,
        alt="Median launch ramp ratios with P25-P75 ranges and pair counts",
    )

    region_accuracy = (
        about_backtest[
            (about_backtest["variant"] == "rounded")
            & (about_backtest["group_type"] == "region")
            & (about_backtest["model"] == final_model)
        ]
        .groupby("group", as_index=False)
        .agg(WAPE=("WAPE", "mean"))
        .sort_values("WAPE", ascending=False)
    )
    weakest_region = region_accuracy.iloc[0]
    launch_forecasts = forecast[forecast["Adjustment"] == "new_series"]
    new_launch_series = launch_forecasts[["SKU_ID", "Region"]].drop_duplicates().shape[0]
    launch_pair_counts = about_launch_ramp.iloc[0]
    coverage_column = "Coverage_P10_P90"
    coverage_quantiles = [
        int(value) for value in re.findall(r"P(\d+)", coverage_column)
    ]
    coverage_target = (coverage_quantiles[1] - coverage_quantiles[0]) / 100
    aggregate_ranges = about_ranges[about_ranges["Level"] == "aggregate"]
    aggregate_coverage_min = aggregate_ranges["Coverage_P10_P90"].min()
    aggregate_coverage_max = aggregate_ranges["Coverage_P10_P90"].max()
    max_widening = aggregate_ranges["Widening_Factor"].max()
    bias_direction = "low" if final_metrics["Bias"] < 0 else "high"
    st.subheader("Known limitations")
    st.markdown(
        "\n".join(
            [
                f"- **Regional accuracy:** {weakest_region['group']} is the weakest region "
                f"by final-model WAPE ({weakest_region['WAPE']:.1%}).",
                f"- **New launches:** {new_launch_series:,} new launches "
                f"({len(launch_forecasts):,} forecast rows) have no history; their ramp is measured "
                f"on {int(launch_pair_counts['Pairs']):,} past predecessor -> successor series pairs "
                f"({int(launch_pair_counts['SKU_Pairs']):,} SKU pairs).",
                f"- **Forecast bias:** the final model leans approximately "
                f"{abs(final_metrics['Bias']):.1%} {bias_direction} in held-out backtests.",
                f"- **Aggregate ranges:** bands are widened for a {coverage_target:.0%} coverage target; "
                f"observed aggregate coverage is {aggregate_coverage_min:.1%} to "
                f"{aggregate_coverage_max:.1%}, with widening up to {max_widening:.2f}x.",
            ]
        )
    )


elif page == "Executive overview":
    st.header("Executive overview")
    show_unconstrained = st.toggle(
        "Show unconstrained demand",
        value=False,
        help="Compare observed units with estimated unconstrained demand.",
    )
    history_monthly = monthly_history(history_filtered)
    forecast_monthly = monthly_forecast(forecast_filtered)
    forecast_band = selected_forecast_ranges(selections)
    unconstrained_monthly = monthly_unconstrained(unconstrained_filtered) if (
        show_unconstrained and unconstrained_filtered is not None
    ) else None

    history_units = float(history_filtered["Units_Sold"].sum(min_count=1))
    history_revenue = float(history_filtered["Revenue_USD"].sum(min_count=1))
    forecast_units = float(forecast_filtered["Units_Forecast"].sum())
    forecast_revenue = float(forecast_filtered["Forecast_Revenue_USD"].sum())

    last_19_start = history_monthly["Month"].max() - pd.DateOffset(months=18)
    last_19 = history_monthly[history_monthly["Month"] >= last_19_start]
    baseline_units = float(last_19["Units_Sold"].sum(min_count=1))
    forecast_vs_history = (
        (forecast_units / baseline_units - 1) * 100 if baseline_units and pd.notna(baseline_units) else None
    )

    cols = st.columns(4)
    forecast_delta = (
        f"Forecast units versus the last 19 months of history: {forecast_vs_history:+.1f}%"
        if forecast_vs_history is not None
        else None
    )
    render_kpi_card(cols[0], "History units", compact_units(history_units))
    render_kpi_card(cols[1], "History revenue", money(history_revenue))
    render_kpi_card(cols[2], "19-month forecast units", compact_units(forecast_units), forecast_delta)
    render_kpi_card(cols[3], "19-month forecast revenue", money(forecast_revenue))

    trends_tab, mix_tab = st.tabs(["Performance trends", "Demand mix"])
    with trends_tab:
        st.caption(
            "How to read this: teal shows history, amber shows the forecast, and the vertical marker denotes Nov 2026."
        )
        if forecast_band is None:
            st.caption(
                "Ranges are available for the total, one family, one region, one family + region, or a single SKU-region"
            )
        else:
            st.caption(
                "P10-P90 contained 80% of actual months in held-out backtests "
                "(78.9% at series level; aggregate bands widened to reach 80%)."
            )
        render_plotly_chart(
            plot_timeline(
                history_monthly,
                forecast_monthly,
                "Units_Sold",
                "Monthly units",
                unconstrained_monthly,
                forecast_band,
            ),
            "units",
        )
        render_plotly_chart(
            plot_timeline(history_monthly, forecast_monthly, "Revenue_USD", "Monthly revenue"),
            "USD",
            hovermode="x unified",
        )

    with mix_tab:
        st.caption("How to read this: compare teal history bars with amber forecast bars for each segment.")
        c1, c2, c3 = st.columns(3)
        mix_specs = [
            (c1, "Product_Family", "Mix by family"),
            (c2, "Region", "Mix by region"),
            (c3, "Sales_Channel", "Mix by channel"),
        ]
        for container, column, title in mix_specs:
            combined_history = history_filtered.groupby(column, as_index=False)["Units_Sold"].sum(min_count=1)
            combined_forecast = forecast_filtered.groupby(column, as_index=False)["Units_Forecast"].sum(min_count=1)
            combined_forecast = combined_forecast.rename(columns={"Units_Forecast": "Units_Sold"})
            mix = pd.concat([combined_history.assign(Period="History"), combined_forecast.assign(Period="Forecast")])
            if mix.empty:
                container.info("No data for the current selection")
                continue
            fig = go.Figure()
            for period, color in [("History", PRIMARY), ("Forecast", FORECAST)]:
                period_data = mix[mix["Period"] == period]
                fig.add_trace(go.Bar(name=period, x=period_data[column], y=period_data["Units_Sold"], marker_color=color))
            fig.update_layout(
                title=title,
                barmode="group",
                height=350,
                margin=dict(l=20, r=20, t=60, b=20),
                xaxis_tickangle=-30,
            )
            fig.update_yaxes(title="Units")
            with container:
                render_plotly_chart(fig, "units")


elif page == "Data health":
    st.header("Data health")
    with st.spinner("Loading the data quality report..."):
        dq = load_dq_report()
    issues = pd.DataFrame(dq["issues"])

    st.subheader("Data quality issues")
    if not issues.empty:
        grouped = (
            issues.groupby(["file", "issue"], as_index=False)
            .agg(rows_affected=("rows_affected", "sum"), action=("action", "first"))
            .sort_values(["file", "rows_affected"], ascending=[True, False])
        )
        st.dataframe(
            grouped,
            column_config=table_config(grouped, wide_columns=("file", "issue", "action")),
            width="stretch",
            hide_index=True,
        )

        top = grouped.sort_values("rows_affected", ascending=False).head(15)
        fig = go.Figure(
            go.Bar(
                x=top["rows_affected"],
                y=top["issue"],
                orientation="h",
                marker_color=PRIMARY,
                text=top["rows_affected"],
                textposition="auto",
            )
        )
        fig.update_layout(
            title="Top issues by rows affected",
            height=max(420, len(top) * 24),
            margin=dict(l=260, r=20, t=60, b=20),
            yaxis={"categoryorder": "total ascending"},
        )
        fig.update_xaxes(title="Rows affected")
        st.caption("How to read this: longer bars indicate more affected records; the table lists the associated action.")
        render_plotly_chart(fig, "rows")
    else:
        st.info("No data quality issues are available for this report.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Raw vs clean row counts")
        st.caption("How to read this: compare source and cleaned record counts for each input file.")
        files = pd.DataFrame(
            [{"file": name, "raw_rows": counts[0], "clean_rows": counts[1]} for name, counts in dq["files"].items()]
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Raw", x=files["file"], y=files["raw_rows"], marker_color=SLATE))
        fig.add_trace(go.Bar(name="Clean", x=files["file"], y=files["clean_rows"], marker_color=OK))
        fig.update_layout(barmode="group", height=380, xaxis_tickangle=-30)
        fig.update_yaxes(title="Rows")
        render_plotly_chart(fig, "rows")

    with c2:
        st.subheader("History row-status breakdown")
        st.caption("How to read this: each bar shows the count of history records in a row-status category.")
        status = pd.DataFrame(
            [{"row_status": key, "rows": value} for key, value in dq["history_row_status"].items()]
        )
        fig = go.Figure(
            go.Bar(
                x=status["row_status"],
                y=status["rows"],
                marker_color=CHART_COLORS,
                text=status["rows"],
                textposition="auto",
            )
        )
        fig.update_layout(height=380, xaxis_tickangle=-20)
        fig.update_yaxes(title="Rows")
        render_plotly_chart(fig, "rows")


elif page == "Demand explorer":
    st.header("Demand explorer")
    show_unconstrained = st.toggle(
        "Show unconstrained demand",
        value=False,
        help="Compare observed units with estimated unconstrained demand.",
    )

    family_values = ordered_options(history_filtered if not history_filtered.empty else history, "Product_Family")
    family = st.selectbox("Family", family_values) if family_values else None

    family_history = history_filtered if family is None else history_filtered[history_filtered["Product_Family"] == family]
    family_forecast = forecast_filtered if family is None else forecast_filtered[forecast_filtered["Product_Family"] == family]
    sku_values = ordered_options(family_history if not family_history.empty else family_forecast, "SKU_ID")
    sku = st.selectbox("SKU", sku_values) if sku_values else None

    sku_history = family_history if sku is None else family_history[family_history["SKU_ID"] == sku]
    sku_forecast = family_forecast if sku is None else family_forecast[family_forecast["SKU_ID"] == sku]
    region_values = ordered_options(sku_history if not sku_history.empty else sku_forecast, "Region")
    region = st.selectbox("Region", region_values) if region_values else None

    region_history = sku_history if region is None else sku_history[sku_history["Region"] == region]
    region_forecast = sku_forecast if region is None else sku_forecast[sku_forecast["Region"] == region]
    customer_values = ordered_options(region_history if not region_history.empty else region_forecast, "Customer_Name")
    customer = st.selectbox("Customer", customer_values) if customer_values else None

    if customer is not None:
        drill_history = region_history[region_history["Customer_Name"] == customer]
        drill_forecast = region_forecast[region_forecast["Customer_Name"] == customer]
    else:
        drill_history = region_history
        drill_forecast = region_forecast

    drill_unconstrained = None
    if unconstrained_filtered is not None:
        drill_unconstrained = unconstrained_filtered.copy()
        if family is not None:
            drill_unconstrained = drill_unconstrained[
                drill_unconstrained["Product_Family"] == family
            ]
        if sku is not None:
            drill_unconstrained = drill_unconstrained[
                drill_unconstrained["SKU_ID"] == sku
            ]
        if region is not None:
            drill_unconstrained = drill_unconstrained[
                drill_unconstrained["Region"] == region
            ]
        if customer is not None:
            drill_unconstrained = drill_unconstrained[
                drill_unconstrained["Customer_Name"] == customer
            ]

    history_monthly = monthly_history(drill_history)
    forecast_monthly = monthly_forecast(drill_forecast)
    unconstrained_monthly = (
        monthly_unconstrained(drill_unconstrained)
        if show_unconstrained and drill_unconstrained is not None
        else None
    )
    st.caption(
        "How to read this: teal shows observed units, amber shows the forecast, and the marker denotes Nov 2026."
    )
    render_plotly_chart(
        plot_timeline(
            history_monthly,
            forecast_monthly,
            "Units_Sold",
            "Monthly units for selected demand drill-down",
            unconstrained_monthly,
        ),
        "units",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.caption("How to read this: each bar is average demand for that calendar month across the selected series.")
        season = normalize_dates(drill_history)
        season["Calendar_Month"] = season["Month"].dt.month
        season = season.groupby("Calendar_Month", as_index=False)["Units_Sold"].mean()
        season["Calendar_Month_Name"] = pd.to_datetime(
            season["Calendar_Month"].astype(str),
            format="%m",
        ).dt.strftime("%b")
        fig = go.Figure(
            go.Bar(
                x=season["Calendar_Month_Name"],
                y=season["Units_Sold"],
                marker_color=PRIMARY,
            )
        )
        fig.update_layout(title="Seasonality: average units by calendar month", height=360)
        fig.update_yaxes(title="Average units")
        render_plotly_chart(fig, "units")

    with c2:
        zero_data = normalize_dates(drill_history)
        valid = zero_data[~zero_data["row_status"].isin(["missing_row", "units_unknown"])].copy()
        monthly_valid = valid.groupby("Month", as_index=False)["Units_Sold"].sum(min_count=1)
        monthly_valid = monthly_valid.dropna(subset=["Units_Sold"])
        zero_share = (
            float((monthly_valid["Units_Sold"] == 0).mean()) if not monthly_valid.empty else 0.0
        )
        render_kpi_card(c2, "Share of zero months", f"{zero_share:.1%}")
        st.caption("Excludes missing_row and units_unknown gaps")

    if sku is not None:
        master_row = sku_master[sku_master["SKU_ID"] == sku]
        if not master_row.empty:
            launch = master_row["Launch_Date"].dropna()
            eol = master_row["EOL_Date"].dropna()
            marker_text = []
            if not launch.empty:
                marker_text.append(f"Launch: {pd.to_datetime(launch.iloc[0]).strftime('%Y-%m-%d')}")
            if not eol.empty:
                marker_text.append(f"EOL: {pd.to_datetime(eol.iloc[0]).strftime('%Y-%m-%d')}")
            if marker_text:
                st.info(" | ".join(marker_text))

    st.subheader("Demand profile")
    st.caption("How to read this: the selected series is classified by demand intermittency and variability.")
    with st.spinner("Loading series demand profiles..."):
        series_profile = load_series_profile()
    profile_pairs = pd.concat(
        [
            history_filtered[["SKU_ID", "Region"]],
            forecast_filtered[["SKU_ID", "Region"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    profile_filtered = series_profile.merge(
        profile_pairs,
        on=["SKU_ID", "Region"],
        how="inner",
    )
    selected_profile = profile_filtered[
        (profile_filtered["SKU_ID"] == sku)
        & (profile_filtered["Region"] == region)
    ] if sku is not None and region is not None else profile_filtered.iloc[0:0]
    profile_population = profile_filtered.merge(
        family_history[["SKU_ID", "Region"]].drop_duplicates(),
        on=["SKU_ID", "Region"],
        how="inner",
    )
    if region is not None:
        profile_population = profile_population[profile_population["Region"] == region]

    profile_class, profile_chart = st.columns([1, 2])
    with profile_class:
        if selected_profile.empty:
            st.info("No demand profile is available for the selected series.")
        else:
            selected_class = str(selected_profile.iloc[0]["Class"]).replace("_", " ").title()
            render_kpi_card(profile_class, "Selected series class", selected_class)
    with profile_chart:
        st.caption("Series counts by class for the selected family and region under the shared filters.")
        if profile_population.empty:
            st.info("No series profiles match the current filters.")
        else:
            class_counts = (
                profile_population["Class"]
                .value_counts()
                .sort_index()
                .rename_axis("Class")
                .reset_index(name="Series")
            )
            fig = go.Figure(
                go.Bar(
                    x=class_counts["Class"],
                    y=class_counts["Series"],
                    marker_color=CHART_COLORS,
                )
            )
            fig.update_layout(title="Series counts by demand class", height=340)
            fig.update_yaxes(title="Series")
            render_plotly_chart(fig, "series")


elif page == "Anomaly monitor":
    st.header("Anomaly monitor")
    with st.spinner("Loading the anomaly register..."):
        anomalies = load_anomaly_register()
    if anomalies is None:
        st.info("The anomaly register is not available yet. Generate it in the pipeline to explore anomalies here.")
    else:
        anomaly_filtered = anomalies
        for column in ["Product_Family", "SKU_ID", "Region", "Sales_Channel", "Customer_Name"]:
            if column in anomaly_filtered.columns:
                values = selections[column]
                if values:
                    anomaly_filtered = anomaly_filtered[anomaly_filtered[column].isin(values)]

        anomaly_filtered = anomaly_filtered.copy()
        anomaly_filtered["Date_Start"] = pd.to_datetime(anomaly_filtered["Date_Start"], errors="coerce")
        anomaly_filtered["Date_End"] = pd.to_datetime(anomaly_filtered["Date_End"], errors="coerce")

        if anomaly_filtered.empty:
            st.info("No anomalies match the current filters.")

        c1, c2 = st.columns(2)
        with c1:
            st.caption("How to read this: bar height shows how many flagged records fall in each category.")
            category_counts = (
                anomaly_filtered["Category"]
                .value_counts(dropna=False)
                .rename_axis("Category")
                .reset_index(name="Count")
            )
            fig = go.Figure(
                go.Bar(
                    x=category_counts["Category"],
                    y=category_counts["Count"],
                    marker_color=PRIMARY,
                    text=category_counts["Count"],
                    textposition="auto",
                )
            )
            fig.update_layout(title="Anomalies by Category", height=340)
            fig.update_yaxes(title="Anomalies")
            render_plotly_chart(fig, "anomalies")
        with c2:
            st.caption("How to read this: compare anomaly counts across the recorded anomaly types.")
            type_counts = (
                anomaly_filtered["Type"]
                .value_counts(dropna=False)
                .rename_axis("Type")
                .reset_index(name="Count")
            )
            fig = go.Figure(
                go.Bar(
                    x=type_counts["Type"],
                    y=type_counts["Count"],
                    marker_color=FORECAST,
                    text=type_counts["Count"],
                    textposition="auto",
                )
            )
            fig.update_layout(
                title="Anomalies by Type",
                height=340,
                xaxis_tickangle=-30,
            )
            fig.update_yaxes(title="Anomalies")
            render_plotly_chart(fig, "anomalies")

        st.subheader("Units time series")
        st.caption("How to read this: teal shows observed units; colored markers show anomaly points and their categories.")
        fig = go.Figure()
        history_units_monthly = monthly_history(history_filtered)
        fig.add_trace(
            go.Scatter(
                x=history_units_monthly["Month"],
                y=history_units_monthly["Units_Sold"],
                mode="lines",
                name="Units observed",
                line=dict(color=PRIMARY, width=3),
            )
        )
        if not anomaly_filtered.empty:
            category_colors = {
                "error_fixed": OK,
                "real_event": PRIMARY,
                "data_gap": ERROR,
                "master_data": SLATE,
            }
            for category, group in anomaly_filtered.groupby("Category", dropna=False):
                points = group.copy()
                points["Units_Point"] = points["Units_Observed"].fillna(points["Units_Expected"]).fillna(0)
                fig.add_trace(
                    go.Scatter(
                        x=points["Date_Start"],
                        y=points["Units_Point"],
                        mode="markers",
                        name=str(category),
                        marker=dict(
                            color=category_colors.get(category, FORECAST),
                            size=9,
                            line=dict(width=1, color="white"),
                        ),
                        customdata=points[["Type", "Reason", "Action"]].fillna("").to_numpy(),
                        hovertemplate=(
                            "<b>%{x|%Y-%m-%d}</b><br>"
                            "Units: %{y:,.0f}<br>"
                            "Type: %{customdata[0]}<br>"
                            "Reason: %{customdata[1]}<br>"
                            "Action: %{customdata[2]}"
                            "<extra></extra>"
                        ),
                    )
                )
        fig.update_layout(
            title="Observed units with anomalies",
            height=420,
            hovermode="closest",
        )
        fig.update_xaxes(title="Date")
        fig.update_yaxes(title="Units")
        render_plotly_chart(fig, "units")

        st.subheader("Filter anomaly register")
        st.caption("Refine the register by anomaly type, business segment, and date range.")
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            category_filter = st.multiselect(
                "Category",
                ordered_options(anomaly_filtered, "Category"),
                key="anomaly_category",
            )
        with f2:
            type_filter = st.multiselect(
                "Type",
                ordered_options(anomaly_filtered, "Type"),
                key="anomaly_type",
            )
        with f3:
            region_filter = st.multiselect(
                "Region",
                ordered_options(anomaly_filtered, "Region"),
                key="anomaly_region",
            )
        with f4:
            family_filter = st.multiselect(
                "Product Family",
                ordered_options(anomaly_filtered, "Product_Family"),
                key="anomaly_family",
            )

        valid_dates = anomaly_filtered[
            anomaly_filtered["Date_Start"].notna() | anomaly_filtered["Date_End"].notna()
        ]
        if not valid_dates.empty:
            start_values = valid_dates["Date_Start"].dropna()
            end_values = valid_dates["Date_End"].dropna()
            min_date = start_values.min().date() if not start_values.empty else pd.Timestamp.now().date()
            max_date = end_values.max().date() if not end_values.empty else min_date
        else:
            min_date = pd.Timestamp.now().date()
            max_date = min_date
        date_filter = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        table_filtered = anomaly_filtered.copy()
        table_filtered = filter_multiselect(table_filtered, category_filter, "Category")
        table_filtered = filter_multiselect(table_filtered, type_filter, "Type")
        table_filtered = filter_multiselect(table_filtered, region_filter, "Region")
        table_filtered = filter_multiselect(table_filtered, family_filter, "Product_Family")
        if isinstance(date_filter, tuple) and len(date_filter) == 2:
            range_start = pd.Timestamp(date_filter[0])
            range_end = pd.Timestamp(date_filter[1])
            table_filtered = table_filtered[
                (table_filtered["Date_Start"].isna() | (table_filtered["Date_Start"] <= range_end))
                & (table_filtered["Date_End"].isna() | (table_filtered["Date_End"] >= range_start))
            ]

        st.dataframe(
            table_filtered,
            column_config=table_config(table_filtered, wide_columns=("Reason", "Action")),
            hide_index=True,
            alt="Filtered anomaly register",
        )
        st.download_button(
            "Download anomaly CSV",
            data=table_filtered.to_csv(index=False).encode("utf-8"),
            file_name="anomaly_register_filtered.csv",
            mime="text/csv",
            width="stretch",
        )


elif page == "Forecast & accuracy":
    st.header("Forecast & accuracy")
    history_monthly = monthly_history(history_filtered)
    forecast_monthly = monthly_forecast(forecast_filtered)
    forecast_band = selected_forecast_ranges(selections)
    st.caption(
        "How to read this: teal shows observed demand, amber shows the forecast, and the vertical marker denotes Nov 2026."
    )
    if forecast_band is None:
        st.caption(
            "Ranges are available for the total, one family, one region, one family + region, or a single SKU-region"
        )
    else:
        st.caption(
            "P10-P90 contained 80% of actual months in held-out backtests "
            "(78.9% at series level; aggregate bands widened to reach 80%)."
        )
    render_plotly_chart(
        plot_timeline(
            history_monthly,
            forecast_monthly,
            "Units_Sold",
            "Forecast vs history",
            forecast_band=forecast_band,
        ),
        "units",
    )

    st.subheader("Forecast units by adjustment")
    st.caption("Review forecast volume by adjustment type; use the selector to focus the table.")
    adjustment_forecast = forecast_filtered.copy()
    adjustment_forecast["Adjustment"] = (
        adjustment_forecast["Adjustment"].fillna("Unadjusted").astype(str)
    )
    adjustment_options = ordered_options(adjustment_forecast, "Adjustment")
    adjustment_filter = st.multiselect(
        "Adjustment",
        adjustment_options,
        key="forecast_adjustment",
    )
    adjustment_forecast = filter_multiselect(
        adjustment_forecast,
        adjustment_filter,
        "Adjustment",
    )
    adjustment_units = (
        adjustment_forecast.groupby("Adjustment", as_index=False)
        .agg(Units_Forecast=("Units_Forecast", "sum"))
        .sort_values("Units_Forecast", ascending=False)
    )
    st.dataframe(
        adjustment_units,
        column_config=table_config(adjustment_units),
        hide_index=True,
        alt="Forecast units grouped by adjustment",
    )

    with st.spinner("Loading backtest accuracy metrics..."):
        backtest = load_backtest()
    rounded = backtest[backtest["variant"] == "rounded"].copy()
    final_model = forecast["Model"].dropna().astype(str).iloc[0]
    target = rounded[rounded["model"] == final_model].copy()

    st.subheader(f"{final_model} accuracy")
    st.caption("How to read this: WAPE and Bias summarize the rounded backtest results by segment and forecast horizon.")
    accuracy_tabs = st.tabs(["Family", "Region", "Horizon"])
    for tab, group_type in zip(accuracy_tabs, ["family", "region", "horizon"]):
        with tab:
            grouped = (
                target[target["group_type"] == group_type]
                .groupby("group", as_index=False)
                .agg(WAPE=("WAPE", "mean"), Bias=("Bias", "mean"), Rows=("Rows", "sum"))
                .sort_values("WAPE", ascending=False)
            )
            if grouped.empty:
                st.info("No backtest rows available")
            else:
                st.dataframe(
                    format_table(grouped, ("WAPE", "Bias")),
                    column_config=table_config(
                        grouped,
                        percentage_columns=("WAPE", "Bias"),
                        wide_columns=("group",),
                    ),
                    width="stretch",
                    hide_index=True,
                    alt=f"{group_type.title()} WAPE and bias for {final_model}",
                )
                fig = go.Figure()
                fig.add_trace(go.Bar(name="WAPE", x=grouped["group"], y=grouped["WAPE"], marker_color=PRIMARY))
                fig.update_layout(
                    title=f"WAPE by {group_type}",
                    height=380,
                    xaxis_tickangle=-30,
                )
                fig.update_yaxes(title="WAPE", tickformat=".1%")
                render_plotly_chart(fig, "WAPE", value_format=".1%")

    st.subheader("Overall model comparison")
    st.caption('Models prefixed with "ablation" are labeled as ablation steps.')
    overall = (
        rounded[rounded["group_type"] == "overall"]
        .groupby("model", as_index=False)
        .agg(WAPE=("WAPE", "mean"), Bias=("Bias", "mean"), Rows=("Rows", "sum"))
        .sort_values("WAPE")
    )
    overall["Model"] = overall["model"].apply(
        lambda model: f"Ablation step {model[len('ablation '):]}"
        if model.startswith("ablation ")
        else model
    )
    overall = overall[["Model", "WAPE", "Bias", "Rows"]]
    overall_styled = overall.style.format({"WAPE": "{:.1%}", "Bias": "{:.1%}"}).apply(
        lambda row: [
            "background-color: #CCFBF1; font-weight: bold"
            if row["Model"] == final_model
            else ""
            for _ in row
        ],
        axis=1,
    )
    overall_config = table_config(
        overall,
        percentage_columns=("WAPE", "Bias"),
        wide_columns=("Model",),
    )
    overall_config["Model"] = st.column_config.TextColumn(pinned=True, width="large")
    st.dataframe(
        overall_styled,
        column_config=overall_config,
        width="stretch",
        hide_index=True,
        alt="Overall model WAPE and bias comparison",
    )


elif page == "Procurement plan":
    st.header("Procurement plan")
    controls = st.columns(3)
    service_level = controls[0].selectbox(
        "Service level",
        ["Point forecast", "P50", "P90"],
        key="procurement_service_level",
    )
    planning_window = controls[1].slider(
        "Planning window (months)",
        min_value=1,
        max_value=12,
        value=6,
        key="procurement_window",
    )
    group_by = controls[2].selectbox(
        "Group by",
        ["Product_Family", "Region", "SKU_ID"],
        key="procurement_group_by",
    )

    procurement = load_forecast_ranges().merge(
        sku_master[["SKU_ID", "Product_Family"]].drop_duplicates("SKU_ID"),
        on="SKU_ID",
        how="left",
        validate="many_to_one",
    )
    eligible_series = forecast_filtered[["SKU_ID", "Region"]].drop_duplicates()
    procurement = procurement.merge(
        eligible_series,
        on=["SKU_ID", "Region"],
        how="inner",
        validate="many_to_one",
    )
    planning_start = pd.Timestamp("2026-11-01")
    planning_end = planning_start + pd.DateOffset(months=planning_window)
    procurement = procurement[
        (procurement["Date"] >= planning_start)
        & (procurement["Date"] < planning_end)
    ].copy()

    latest_prices = (
        load_prices()
        .assign(Date=lambda prices: pd.to_datetime(prices["Date"]))
        .sort_values("Date")
        .groupby("SKU_ID", as_index=False)
        .tail(1)[["SKU_ID", "List_Price_USD"]]
    )
    procurement = procurement.merge(
        latest_prices,
        on="SKU_ID",
        how="left",
        validate="many_to_one",
    )
    quantile_column = {
        "Point forecast": "Units_Forecast",
        "P50": "P50",
        "P90": "P90",
    }[service_level]
    procurement["Units_To_Order"] = procurement[quantile_column]
    procurement["Safety_Buffer"] = (
        procurement["Units_To_Order"] - procurement["Units_Forecast"]
    )
    procurement["Value_USD"] = (
        procurement["Units_To_Order"] * procurement["List_Price_USD"]
    )
    procurement_summary = (
        procurement.groupby(group_by, as_index=False)
        .agg(
            Units_To_Order=("Units_To_Order", "sum"),
            Safety_Buffer=("Safety_Buffer", "sum"),
            Value_USD=("Value_USD", "sum"),
        )
        .sort_values("Safety_Buffer", ascending=False)
        .reset_index(drop=True)
    )
    procurement_summary = procurement_summary.rename(
        columns={
            group_by: "Group",
            "Units_To_Order": f"Units to order ({service_level})",
            "Safety_Buffer": "Safety buffer",
            "Value_USD": "Value (USD)",
        }
    )
    st.caption(
        f"Planning window: {planning_start:%b %Y} through "
        f"{(planning_end - pd.DateOffset(months=1)):%b %Y}."
    )
    st.caption(
        "Summing SKU-level P90s over-covers the true aggregate; use as the buffer if every SKU must be covered at P90."
    )
    st.dataframe(
        procurement_summary,
        column_config=table_config(procurement_summary, wide_columns=("Group",)),
        width="stretch",
        hide_index=True,
        alt=f"Procurement plan grouped by {group_by}",
    )
    st.download_button(
        "Download procurement plan CSV",
        data=procurement_summary.to_csv(index=False).encode("utf-8"),
        file_name="procurement_plan.csv",
        mime="text/csv",
        width="stretch",
    )

    top_buffers = procurement_summary.head(20).sort_values("Safety buffer")
    if top_buffers.empty:
        st.info("No procurement series match the current filters and planning window.")
    else:
        buffer_figure = go.Figure(
            go.Bar(
                x=top_buffers["Safety buffer"],
                y=top_buffers["Group"],
                orientation="h",
                marker_color=FORECAST,
                text=top_buffers["Safety buffer"],
                textposition="auto",
            )
        )
        buffer_figure.update_layout(
            title=f"20 largest buffers by {group_by}",
            height=max(420, len(top_buffers) * 24),
            margin=dict(l=220, r=20, t=60, b=20),
        )
        buffer_figure.update_xaxes(title="Safety buffer (units)")
        buffer_figure.update_yaxes(categoryorder="array", categoryarray=top_buffers["Group"])
        render_plotly_chart(buffer_figure, "units")


elif page == "Planner exceptions":
    st.header("Planner exceptions")
    exceptions = load_planner_exceptions()
    for column in ["Product_Family", "SKU_ID", "Region", "Customer_Name"]:
        exceptions = filter_multiselect(exceptions, selections[column], column)
    if selected_channel:
        eligible_exception_series = forecast_filtered[
            ["SKU_ID", "Region", "Product_Family", "Customer_Name"]
        ].drop_duplicates()
        exceptions = exceptions.merge(
            eligible_exception_series,
            on=["SKU_ID", "Region", "Product_Family", "Customer_Name"],
            how="inner",
            validate="many_to_one",
        )

    severity_columns = st.columns(3)
    for column, severity in zip(severity_columns, ["high", "medium", "low"]):
        count = int((exceptions["Severity"] == severity).sum())
        render_kpi_card(column, f"{severity.title()} severity", f"{count:,}")

    st.subheader("Exceptions by type")
    exception_types = (
        exceptions.groupby("Exception_Type", as_index=False)
        .size()
        .rename(columns={"size": "Exceptions"})
        .sort_values("Exceptions", ascending=False)
    )
    if exception_types.empty:
        st.info("No planner exceptions match the current sidebar filters.")
    else:
        exception_figure = go.Figure(
            go.Bar(
                x=exception_types["Exceptions"],
                y=exception_types["Exception_Type"],
                orientation="h",
                marker_color=PRIMARY,
                text=exception_types["Exceptions"],
                textposition="auto",
            )
        )
        exception_figure.update_layout(
            title="Planner exception counts by type",
            height=max(360, len(exception_types) * 30),
            margin=dict(l=220, r=20, t=60, b=20),
            yaxis={"categoryorder": "total ascending"},
        )
        exception_figure.update_xaxes(title="Exceptions")
        render_plotly_chart(exception_figure, "exceptions")

    st.subheader("Filter planner exceptions")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        severity_filter = st.multiselect(
            "Severity",
            ordered_options(exceptions, "Severity"),
            key="planner_exception_severity",
        )
    with f2:
        type_filter = st.multiselect(
            "Type",
            ordered_options(exceptions, "Exception_Type"),
            key="planner_exception_type",
        )
    with f3:
        family_filter = st.multiselect(
            "Family",
            ordered_options(exceptions, "Product_Family"),
            key="planner_exception_family",
        )
    with f4:
        region_filter = st.multiselect(
            "Region",
            ordered_options(exceptions, "Region"),
            key="planner_exception_region",
        )

    table_filtered = exceptions.copy()
    table_filtered = filter_multiselect(table_filtered, severity_filter, "Severity")
    table_filtered = filter_multiselect(table_filtered, type_filter, "Exception_Type")
    table_filtered = filter_multiselect(table_filtered, family_filter, "Product_Family")
    table_filtered = filter_multiselect(table_filtered, region_filter, "Region")
    severity_styles = {
        "high": "background-color: #FEE2E2; color: #991B1B; font-weight: bold",
        "medium": "background-color: #FEF3C7; color: #92400E; font-weight: bold",
        "low": "background-color: #DCFCE7; color: #166534; font-weight: bold",
    }
    styled_exceptions = format_table(table_filtered, ("Change_Pct",)).apply(
        lambda column: [severity_styles.get(value, "") for value in column],
        subset=["Severity"],
    )
    st.dataframe(
        styled_exceptions,
        column_config=table_config(
            table_filtered,
            percentage_columns=("Change_Pct",),
            wide_columns=("Detail", "Suggested_Action", "Customer_Name"),
        ),
        width="stretch",
        hide_index=True,
        alt="Filterable planner exceptions by severity, type, family, and region",
    )
    st.download_button(
        "Download planner exceptions CSV",
        data=table_filtered.to_csv(index=False).encode("utf-8"),
        file_name="planner_exceptions_filtered.csv",
        mime="text/csv",
        width="stretch",
    )


elif page == "Stockouts & lost sales":
    st.header("Stockouts & lost sales")
    stockouts = load_stockouts().merge(
        sku_master[["SKU_ID", "Product_Family"]].drop_duplicates("SKU_ID"),
        on="SKU_ID",
        how="left",
        validate="many_to_one",
    )
    stockouts = apply_available_filters(stockouts, selections)
    if selected_channel or selected_customer:
        if unconstrained_filtered is not None:
            eligible_stockout_series = unconstrained_filtered[
                ["SKU_ID", "Region"]
            ].drop_duplicates()
            stockouts = stockouts.merge(
                eligible_stockout_series,
                on=["SKU_ID", "Region"],
                how="inner",
                validate="many_to_one",
            )

    latest_prices = (
        load_prices()
        .assign(Date=lambda prices: pd.to_datetime(prices["Date"]))
        .sort_values("Date")
        .groupby("SKU_ID", as_index=False)
        .tail(1)[["SKU_ID", "List_Price_USD"]]
    )
    stockouts = stockouts.merge(
        latest_prices,
        on="SKU_ID",
        how="left",
        validate="many_to_one",
    )
    stockouts["Lost_Revenue_USD"] = (
        stockouts["Lost_Units"] * stockouts["List_Price_USD"]
    )

    requested_units = float(stockouts["Requested_Units"].sum())
    shipped_units = float(stockouts["Shipped_Units"].sum())
    lost_units = float(stockouts["Lost_Units"].sum())
    backordered_units = float(stockouts["Backordered_Units"].sum())
    lost_revenue = float(stockouts["Lost_Revenue_USD"].sum())
    fill_rate = shipped_units / requested_units if requested_units else 0
    kpis = st.columns(4)
    render_kpi_card(kpis[0], "Lost units", compact_units(lost_units))
    render_kpi_card(kpis[1], "Backordered units", compact_units(backordered_units))
    render_kpi_card(
        kpis[2],
        "Shipped vs requested",
        f"{compact_units(shipped_units)} / {compact_units(requested_units)}",
        f"Fill rate: {fill_rate:.1%}",
    )
    render_kpi_card(kpis[3], "Lost revenue at list price", money(lost_revenue))

    hidden_stockouts = pd.DataFrame()
    if unconstrained_filtered is not None:
        hidden_stockouts = unconstrained_filtered.copy()
        hidden_stockouts["Date"] = pd.to_datetime(hidden_stockouts["Date"])
        hidden_stockouts = hidden_stockouts[
            hidden_stockouts["Product_Family"].eq("Electrical Assemblies")
            & hidden_stockouts["Uncon_Source"].eq("estimated_chip_shortage")
            & hidden_stockouts["Date"].dt.year.isin([2021, 2022])
        ]
    estimated_hidden_units = (
        hidden_stockouts["Units_Unconstrained"].fillna(0)
        - hidden_stockouts["Units_Observed"].fillna(0)
    ).sum()
    st.info(
        "The stockout file is incomplete. Hidden stockouts in 2021-22 for Electrical Assemblies "
        "are estimated in unconstrained_history using Uncon_Source='estimated_chip_shortage'. "
        f"The estimate under the current sidebar filters is {compact_units(float(estimated_hidden_units))} units."
    )

    if stockouts.empty:
        st.info("No stockout records match the current sidebar filters.")
    else:
        stockouts["Month"] = stockouts["Date"].dt.to_period("M").dt.to_timestamp()
        monthly_stockouts = (
            stockouts.groupby("Month", as_index=False)
            .agg(
                Lost_Units=("Lost_Units", "sum"),
                Backordered_Units=("Backordered_Units", "sum"),
            )
            .sort_values("Month")
        )
        monthly_figure = go.Figure()
        monthly_figure.add_trace(
            go.Bar(
                name="Lost units",
                x=monthly_stockouts["Month"],
                y=monthly_stockouts["Lost_Units"],
                marker_color=FORECAST,
            )
        )
        monthly_figure.add_trace(
            go.Bar(
                name="Backordered units",
                x=monthly_stockouts["Month"],
                y=monthly_stockouts["Backordered_Units"],
                marker_color=PRIMARY,
            )
        )
        monthly_figure.update_layout(
            title="Monthly lost and backordered units",
            barmode="group",
            height=390,
            hovermode="x unified",
        )
        monthly_figure.update_xaxes(title="Month")
        monthly_figure.update_yaxes(title="Units")
        render_plotly_chart(monthly_figure, "units", hovermode="x unified")

        reason_units = (
            stockouts.groupby("Reason", as_index=False)
            .agg(
                Lost_Units=("Lost_Units", "sum"),
                Backordered_Units=("Backordered_Units", "sum"),
            )
            .sort_values("Lost_Units", ascending=True)
        )
        reason_figure = go.Figure()
        reason_figure.add_trace(
            go.Bar(
                name="Lost units",
                x=reason_units["Lost_Units"],
                y=reason_units["Reason"],
                orientation="h",
                marker_color=FORECAST,
            )
        )
        reason_figure.add_trace(
            go.Bar(
                name="Backordered units",
                x=reason_units["Backordered_Units"],
                y=reason_units["Reason"],
                orientation="h",
                marker_color=PRIMARY,
            )
        )
        reason_figure.update_layout(
            title="Lost and backordered units by reason",
            barmode="group",
            height=max(360, len(reason_units) * 34),
            margin=dict(l=220, r=20, t=60, b=20),
        )
        reason_figure.update_xaxes(title="Units")
        render_plotly_chart(reason_figure, "units")

        family_losses = (
            stockouts.groupby("Product_Family", as_index=False)
            .agg(Lost_Units=("Lost_Units", "sum"))
            .sort_values("Lost_Units", ascending=True)
        )
        family_figure = go.Figure(
            go.Bar(
                x=family_losses["Lost_Units"],
                y=family_losses["Product_Family"],
                orientation="h",
                marker_color=PRIMARY,
            )
        )
        family_figure.update_layout(
            title="Lost units by product family",
            height=max(380, len(family_losses) * 36),
            margin=dict(l=250, r=20, t=60, b=20),
        )
        family_figure.update_xaxes(title="Lost units")
        render_plotly_chart(family_figure, "units")

        top_sku_regions = (
            stockouts.groupby(["SKU_ID", "Region"], as_index=False)
            .agg(
                Lost_Units=("Lost_Units", "sum"),
                Backordered_Units=("Backordered_Units", "sum"),
                Lost_Revenue_USD=("Lost_Revenue_USD", "sum"),
            )
            .sort_values("Lost_Units", ascending=False)
            .head(20)
        )
        top_figure = go.Figure(
            go.Bar(
                x=top_sku_regions["Lost_Units"],
                y=top_sku_regions["SKU_ID"] + " | " + top_sku_regions["Region"],
                orientation="h",
                marker_color=FORECAST,
            )
        )
        top_figure.update_layout(
            title="Top 20 SKU-regions by lost units",
            height=max(440, len(top_sku_regions) * 25),
            margin=dict(l=260, r=20, t=60, b=20),
        )
        top_figure.update_xaxes(title="Lost units")
        top_figure.update_yaxes(categoryorder="total ascending")
        render_plotly_chart(top_figure, "units")
        st.dataframe(
            top_sku_regions,
            column_config=table_config(
                top_sku_regions,
                wide_columns=("SKU_ID", "Region"),
            ),
            width="stretch",
            hide_index=True,
            alt="Top 20 SKU-regions with lost units and estimated lost revenue",
        )


elif page == "Lifecycle":
    st.header("Lifecycle")
    lifecycle_master = sku_master.copy()
    date_columns = ["Launch_Date", "Master_Launch_Date", "EOL_Date"]
    for column in date_columns:
        lifecycle_master[column] = pd.to_datetime(
            lifecycle_master[column],
            errors="coerce",
        )
    lifecycle_master = apply_available_filters(
        lifecycle_master,
        {
            "Product_Family": selected_family,
            "SKU_ID": selected_sku,
        },
    )
    if selected_region or selected_channel or selected_customer:
        lifecycle_master = lifecycle_master[
            lifecycle_master["SKU_ID"].isin(forecast_filtered["SKU_ID"].unique())
        ]
    both_launch_dates_missing = (
        lifecycle_master["Launch_Date"].isna()
        & lifecycle_master["Master_Launch_Date"].isna()
    )
    lifecycle_master["Master_Data_Error"] = (
        lifecycle_master["Launch_Date"].ne(lifecycle_master["Master_Launch_Date"])
        & ~both_launch_dates_missing
    )

    st.subheader("Launches and EOLs by lifecycle status")
    lifecycle_summary = (
        lifecycle_master.groupby("Lifecycle_Status", as_index=False)
        .agg(
            SKUs=("SKU_ID", "nunique"),
            Launches=("Launch_Date", "count"),
            EOLs=("EOL_Date", "count"),
            Master_Data_Errors=("Master_Data_Error", "sum"),
        )
        .sort_values("Lifecycle_Status")
    )
    st.dataframe(
        lifecycle_summary,
        column_config=table_config(lifecycle_summary),
        width="stretch",
        hide_index=True,
        alt="Launches, EOLs, and master-data errors by lifecycle status",
    )
    lifecycle_detail = lifecycle_master[
        [
            "SKU_ID",
            "Product_Family",
            "Lifecycle_Status",
            "Launch_Date",
            "Master_Launch_Date",
            "EOL_Date",
            "Replaces_SKU",
            "Master_Data_Error",
        ]
    ].sort_values(["Lifecycle_Status", "Launch_Date", "SKU_ID"])
    st.dataframe(
        lifecycle_detail,
        column_config=table_config(
            lifecycle_detail,
            wide_columns=("Product_Family", "Replaces_SKU"),
        ),
        width="stretch",
        hide_index=True,
        alt="SKU launch and EOL dates with master-data error flags",
    )
    mismatch_count = int(lifecycle_master["Master_Data_Error"].sum())
    if mismatch_count:
        st.warning(
            f"{mismatch_count} SKU(s) have a Master_Launch_Date that differs from Launch_Date; "
            "these are flagged as master-data errors."
        )

    timeline_skus = lifecycle_master[
        lifecycle_master["Lifecycle_Status"].isin(
            ["EOL Announced", "Planned Launch", "Discontinued"]
        )
    ]
    st.subheader("Lifecycle event timeline")
    if timeline_skus.empty:
        st.info("No lifecycle events match the current sidebar filters.")
    else:
        status_colors = {
            "EOL Announced": FORECAST,
            "Planned Launch": PRIMARY,
            "Discontinued": SLATE,
        }
        timeline_figure = go.Figure()
        for status, color in status_colors.items():
            status_rows = timeline_skus[
                timeline_skus["Lifecycle_Status"] == status
            ]
            for event_name, column, symbol in [
                ("Launch", "Launch_Date", "circle"),
                ("EOL", "EOL_Date", "diamond"),
            ]:
                events = status_rows.dropna(subset=[column])
                if events.empty:
                    continue
                timeline_figure.add_trace(
                    go.Scatter(
                        x=events[column],
                        y=events["SKU_ID"],
                        mode="markers",
                        name=f"{status} - {event_name}",
                        marker=dict(color=color, size=11, symbol=symbol),
                        hovertemplate=(
                            f"<b>%{{y}}</b><br>{status} {event_name}: "
                            "%{x|%b %Y}<extra></extra>"
                        ),
                    )
                )
        timeline_figure.update_layout(
            title="Launch and EOL dates for selected lifecycle statuses",
            height=max(420, len(timeline_skus) * 36),
            hovermode="closest",
            margin=dict(l=100, r=20, t=60, b=20),
        )
        timeline_figure.update_xaxes(title="Date")
        timeline_figure.update_yaxes(title="SKU", categoryorder="category descending")
        render_plotly_chart(timeline_figure, "date")

    replacement_pairs = sku_master.loc[
        sku_master["Replaces_SKU"].notna(),
        ["Replaces_SKU", "SKU_ID"],
    ].drop_duplicates()
    filtered_forecast_skus = set(forecast_filtered["SKU_ID"].dropna())
    replacement_pairs = replacement_pairs[
        replacement_pairs["Replaces_SKU"].isin(filtered_forecast_skus)
        & replacement_pairs["SKU_ID"].isin(filtered_forecast_skus)
    ].sort_values(["Replaces_SKU", "SKU_ID"])
    st.subheader("Predecessor and successor forecast")
    if replacement_pairs.empty:
        st.info("No predecessor/successor pairs have both forecasts available under the current sidebar filters.")
    else:
        pair_labels = {
            f"{row.Replaces_SKU} -> {row.SKU_ID}": (
                row.Replaces_SKU,
                row.SKU_ID,
            )
            for row in replacement_pairs.itertuples(index=False)
        }
        selected_pair = st.selectbox(
            "Predecessor -> successor",
            list(pair_labels),
            key="lifecycle_pair",
        )
        predecessor, successor = pair_labels[selected_pair]
        pair_forecast = forecast_filtered[
            forecast_filtered["SKU_ID"].isin([predecessor, successor])
        ]
        pair_monthly = (
            pair_forecast.groupby(["Month", "SKU_ID"], as_index=False)
            .agg(Units_Forecast=("Units_Forecast", "sum"))
            .sort_values("Month")
        )
        pair_figure = go.Figure()
        for sku_id, label, color in [
            (predecessor, "Predecessor", SLATE),
            (successor, "Successor", FORECAST),
        ]:
            sku_forecast = pair_monthly[pair_monthly["SKU_ID"] == sku_id]
            pair_figure.add_trace(
                go.Scatter(
                    x=sku_forecast["Month"],
                    y=sku_forecast["Units_Forecast"],
                    mode="lines+markers",
                    name=f"{label}: {sku_id}",
                    line=dict(color=color, width=3),
                )
            )
        pair_figure.update_layout(
            title=f"Forecast for {predecessor} and {successor}",
            height=390,
            hovermode="x unified",
        )
        pair_figure.update_xaxes(title="Month")
        pair_figure.update_yaxes(title="Forecast units")
        render_plotly_chart(pair_figure, "units", hovermode="x unified")


elif page == "Data download":
    st.header("Data download")
    st.caption(
        "Sidebar filters are applied to matching columns in each output; filters without a corresponding column are ignored."
    )
    download_items = [
        ("Clean history", load_history(), "clean_history.csv"),
        (
            "Unconstrained history",
            load_unconstrained_history(),
            "unconstrained_history.csv",
        ),
        ("Forecast", load_forecast_source(), "forecast.csv"),
        ("Forecast ranges (series)", load_forecast_ranges(), "forecast_ranges.csv"),
        (
            "Forecast ranges (aggregate)",
            load_forecast_ranges_aggregate(),
            "forecast_ranges_aggregate.csv",
        ),
        ("Anomaly register", load_anomaly_register(), "anomaly_register.csv"),
        ("Planner exceptions", load_planner_exceptions(), "planner_exceptions.csv"),
        ("Submission", load_submission(), "submission.csv"),
    ]
    columns = st.columns(2)
    for index, (label, data, file_name) in enumerate(download_items):
        if data is None:
            st.info(f"{label} is not available in outputs.")
            continue
        filtered_data = apply_available_filters(data, selections)
        columns[index % 2].download_button(
            f"Download {label} CSV",
            data=filtered_data.to_csv(index=False).encode("utf-8"),
            file_name=file_name,
            mime="text/csv",
            key=f"download_{file_name}",
            width="stretch",
        )


elif page == "Macro drivers":
    st.header("Macro drivers")
    with st.spinner("Loading macro lead-lag relationships..."):
        leadlag_grid = load_macro_leadlag_grid()
        leadlag = load_macro_leadlag()

    if selected_family:
        leadlag_grid = leadlag_grid[leadlag_grid["Family"].isin(selected_family)]
        leadlag = leadlag[leadlag["Family"].isin(selected_family)]

    best_lags = leadlag.merge(
        leadlag_grid.rename(columns={"Lag": "Best_Lag", "Corr": "Corr_at_Best_Lag"}),
        on=["Family", "Indicator", "Best_Lag"],
        how="left",
    )
    if best_lags.empty:
        st.info("No macro relationships match the current family filters.")
    else:
        st.caption(
            "Correlation of year-over-year demand growth with the indicator, indicator leading by N months. "
            "'Stable' = significant (permutation test) and same sign and similar lag in both halves of the data."
        )
        heatmap_data = best_lags.copy()
        heatmap_data["Lag_Label"] = heatmap_data.apply(
            lambda row: f"{int(row['Best_Lag'])} mo"
            + (" *" if row["Signal"] == "stable" else ""),
            axis=1,
        )
        correlation_matrix = heatmap_data.pivot(
            index="Indicator",
            columns="Family",
            values="Corr_at_Best_Lag",
        )
        lag_matrix = heatmap_data.pivot(
            index="Indicator",
            columns="Family",
            values="Lag_Label",
        ).reindex(index=correlation_matrix.index, columns=correlation_matrix.columns)
        signal_matrix = heatmap_data.pivot(
            index="Indicator",
            columns="Family",
            values="Signal",
        ).reindex(index=correlation_matrix.index, columns=correlation_matrix.columns)
        heatmap = go.Figure(
            go.Heatmap(
                z=correlation_matrix.to_numpy(),
                x=correlation_matrix.columns.tolist(),
                y=correlation_matrix.index.tolist(),
                text=lag_matrix.to_numpy(),
                customdata=signal_matrix.to_numpy(),
                texttemplate="%{text}",
                textfont=dict(size=10, color="#0F172A"),
                colorscale=[
                    [0, ERROR],
                    [0.5, "#F8FAFC"],
                    [1, PRIMARY],
                ],
                zmin=-1,
                zmax=1,
                zmid=0,
                xgap=2,
                ygap=2,
                colorbar=dict(title="Corr"),
                hovertemplate=(
                    "<b>%{y}</b><br>Family: %{x}<br>Correlation: %{z:.3f}"
                    "<br>Best lag: %{text}<br>Signal: %{customdata}<extra></extra>"
                ),
            )
        )
        heatmap.update_layout(
            title="Best-lag correlation by indicator and family",
            height=max(480, 44 * len(correlation_matrix.index)),
        )
        heatmap.update_xaxes(title="Family", tickangle=-35)
        heatmap.update_yaxes(title="Indicator")
        st.caption("An asterisk marks stable relationships; each cell is colored by correlation and labeled with its lag.")
        render_plotly_chart(heatmap, "correlation")

        st.subheader("Correlation by lag")
        selection_columns = st.columns(2)
        family_options = sorted(best_lags["Family"].unique().tolist())
        selected_macro_family = selection_columns[0].selectbox(
            "Family",
            family_options,
            key="macro_family",
        )
        family_relationships = best_lags[best_lags["Family"] == selected_macro_family]
        indicator_options = sorted(family_relationships["Indicator"].unique().tolist())
        selected_indicator = selection_columns[1].selectbox(
            "Indicator",
            indicator_options,
            key="macro_indicator",
        )
        lag_series = leadlag_grid[
            (leadlag_grid["Family"] == selected_macro_family)
            & (leadlag_grid["Indicator"] == selected_indicator)
        ].sort_values("Lag")
        best_row = family_relationships[
            family_relationships["Indicator"] == selected_indicator
        ].iloc[0]
        lag_figure = go.Figure()
        lag_figure.add_trace(
            go.Scatter(
                x=lag_series["Lag"],
                y=lag_series["Corr"],
                mode="lines+markers",
                name="Correlation",
                line=dict(color=PRIMARY, width=3),
                hovertemplate="Lag: %{x} months<br>Correlation: %{y:.3f}<extra></extra>",
            )
        )
        lag_figure.add_trace(
            go.Scatter(
                x=[best_row["Best_Lag"]],
                y=[best_row["Corr_at_Best_Lag"]],
                mode="markers",
                name=f"Best lag ({int(best_row['Best_Lag'])} months)",
                marker=dict(color=FORECAST, size=13, symbol="diamond"),
                hovertemplate="Best lag: %{x} months<br>Correlation: %{y:.3f}<extra></extra>",
            )
        )
        lag_figure.update_layout(
            title=f"{selected_indicator} - {selected_macro_family}",
            height=390,
        )
        lag_figure.update_xaxes(title="Indicator lead (months)", tickformat="d")
        lag_figure.update_yaxes(title="Correlation", tickformat=".3f")
        render_plotly_chart(lag_figure, "correlation", value_format=".3f")

        st.subheader("Stable relationships")
        stable_relationships = (
            leadlag[leadlag["Signal"] == "stable"][
                ["Family", "Indicator", "Best_Lag", "Corr", "Corr_H1", "Corr_H2"]
            ]
            .sort_values(["Family", "Indicator"])
            .reset_index(drop=True)
        )
        if stable_relationships.empty:
            st.info("No stable relationships match the current family filters.")
        else:
            stable_styled = stable_relationships.style.format(
                {"Corr": "{:.3f}", "Corr_H1": "{:.3f}", "Corr_H2": "{:.3f}"}
            )
            st.dataframe(
                stable_styled,
                column_config={
                    "Family": st.column_config.TextColumn(width="large"),
                    "Indicator": st.column_config.TextColumn(width="large"),
                    "Best_Lag": st.column_config.NumberColumn(format="localized"),
                },
                hide_index=True,
                alt="Stable macro relationships by family and indicator",
            )


elif page == "What-if":
    st.header("What-if")
    st.caption("Estimate how a selected macro indicator change could shift the current forecast.")
    with st.spinner("Loading macro sensitivity estimates..."):
        sensitivity = load_macro_sensitivity()

    indicator_options = sorted(sensitivity["Indicator"].dropna().unique().tolist())
    if not indicator_options:
        st.info("No macro sensitivity estimates are available.")
    else:
        default_indicator = (
            "ISM_Manufacturing_PMI"
            if "ISM_Manufacturing_PMI" in indicator_options
            else indicator_options[0]
        )
        controls = st.columns(3)
        indicator = controls[0].selectbox(
            "Indicator",
            indicator_options,
            index=indicator_options.index(default_indicator),
            key="whatif_indicator",
        )
        unit = "index points" if indicator == "ISM_Manufacturing_PMI" else "percentage points"
        indicator_change = controls[1].slider(
            f"Change ({unit})",
            min_value=-10.0,
            max_value=10.0,
            value=-3.0,
            step=0.5,
            key="whatif_change",
        )
        forecast_months = sorted(pd.Timestamp(month) for month in forecast["Month"].dropna().unique())
        default_start = pd.Timestamp("2026-11-01")
        start_index = (
            forecast_months.index(default_start)
            if default_start in forecast_months
            else 0
        )
        if forecast_months:
            scenario_start = controls[2].selectbox(
                "Scenario start month",
                forecast_months,
                index=start_index,
                format_func=lambda month: month.strftime("%Y-%m"),
                key="whatif_start_month",
            )
        else:
            scenario_start = None

        if scenario_start is None:
            st.info("No forecast months are available for a scenario.")
        elif forecast_filtered.empty:
            st.info("No forecast series match the current filters.")
        else:
            scenario = forecast_filtered.copy()
            scenario["Scenario_Units"] = scenario["Units_Forecast"]
            scenario["Low_Units"] = scenario["Units_Forecast"]
            scenario["High_Units"] = scenario["Units_Forecast"]
            sensitivity_rows = sensitivity[sensitivity["Indicator"] == indicator]

            for row in sensitivity_rows.itertuples(index=False):
                effective_start = scenario_start + pd.DateOffset(months=int(row.Best_Lag))
                affected = (
                    (scenario["Product_Family"] == row.Family)
                    & (scenario["Month"] >= effective_start)
                )
                if not affected.any():
                    continue
                base_units = scenario.loc[affected, "Units_Forecast"]
                scenario.loc[affected, "Scenario_Units"] = base_units * math.exp(
                    float(row.Slope_LogGrowth_per_Unit) * indicator_change
                )
                h1_units = base_units * math.exp(float(row.Slope_H1) * indicator_change)
                h2_units = base_units * math.exp(float(row.Slope_H2) * indicator_change)
                scenario.loc[affected, "Low_Units"] = pd.concat(
                    [h1_units, h2_units],
                    axis=1,
                ).min(axis=1)
                scenario.loc[affected, "High_Units"] = pd.concat(
                    [h1_units, h2_units],
                    axis=1,
                ).max(axis=1)

            monthly_scenario = (
                scenario.groupby("Month", as_index=False)
                .agg(
                    Base_Units=("Units_Forecast", "sum"),
                    Scenario_Units=("Scenario_Units", "sum"),
                    Low_Units=("Low_Units", "sum"),
                    High_Units=("High_Units", "sum"),
                )
                .sort_values("Month")
            )
            scenario_figure = go.Figure()
            scenario_figure.add_trace(
                go.Scatter(
                    x=monthly_scenario["Month"],
                    y=monthly_scenario["Low_Units"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            scenario_figure.add_trace(
                go.Scatter(
                    x=monthly_scenario["Month"],
                    y=monthly_scenario["High_Units"],
                    mode="lines",
                    name="Slope H1/H2 range",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(217, 119, 6, 0.14)",
                    hovertemplate="Month: %{x|%b %Y}<br>High range: %{y:,.0f} units<extra></extra>",
                )
            )
            scenario_figure.add_trace(
                go.Scatter(
                    x=monthly_scenario["Month"],
                    y=monthly_scenario["Base_Units"],
                    mode="lines",
                    name="Base forecast",
                    line=dict(color=PRIMARY, width=3),
                    hovertemplate="Month: %{x|%b %Y}<br>Base: %{y:,.0f} units<extra></extra>",
                )
            )
            scenario_figure.add_trace(
                go.Scatter(
                    x=monthly_scenario["Month"],
                    y=monthly_scenario["Scenario_Units"],
                    mode="lines",
                    name="Scenario forecast",
                    line=dict(color=FORECAST, width=3, dash="dash"),
                    hovertemplate="Month: %{x|%b %Y}<br>Scenario: %{y:,.0f} units<extra></extra>",
                )
            )
            scenario_figure.update_layout(
                title="Base and scenario forecast",
                height=430,
                hovermode="x unified",
            )
            scenario_figure.update_xaxes(title="Month")
            scenario_figure.update_yaxes(title="Forecast units")
            st.caption(
                "How to read this: the solid teal line is the base forecast, the dashed amber line is the scenario, "
                "and the shaded band uses the two half-period slopes."
            )
            render_plotly_chart(scenario_figure, "units", hovermode="x unified")

            family_scenario = (
                scenario.groupby("Product_Family", as_index=False)
                .agg(
                    Base_Units=("Units_Forecast", "sum"),
                    Scenario_Units=("Scenario_Units", "sum"),
                    Low_Units=("Low_Units", "sum"),
                    High_Units=("High_Units", "sum"),
                )
                .rename(columns={"Product_Family": "Family"})
            )
            family_scenario["Change_Pct"] = family_scenario["Scenario_Units"].div(
                family_scenario["Base_Units"].where(family_scenario["Base_Units"] != 0)
            ).sub(1)
            family_scenario = family_scenario.rename(
                columns={
                    "Base_Units": "Base units",
                    "Scenario_Units": "Scenario units",
                    "Low_Units": "Low units",
                    "High_Units": "High units",
                    "Change_Pct": "Change",
                }
            )
            st.subheader("Scenario units by family")
            st.caption(
                "Families without a stable relationship for the selected indicator remain at the base forecast."
            )
            st.dataframe(
                format_table(family_scenario, ("Change",)),
                column_config=table_config(
                    family_scenario,
                    percentage_columns=("Change",),
                    wide_columns=("Family",),
                ),
                hide_index=True,
                alt="Base, scenario, and sensitivity-range units by family",
            )
            st.caption(
                "Estimated from 2020-2026 history; indicates direction and size, not a guarantee. "
                "Families without a stable relationship are unchanged."
            )


elif page == "EV shift":
    st.header("EV shift")
    with st.spinner("Loading powertrain application data..."):
        ev_shift = load_ev_shift()

    st.caption("EV shift data is an aggregate output and is not segmented by the shared filters.")
    powertrain_figure = make_subplots(specs=[[{"secondary_y": True}]])
    for application, color in [
        ("EV", PRIMARY),
        ("ICE", SLATE),
        ("Universal", FORECAST),
    ]:
        powertrain_figure.add_trace(
            go.Scatter(
                x=ev_shift["Date"],
                y=ev_shift[application],
                mode="lines",
                name=application,
                stackgroup="applications",
                line=dict(color=color, width=1),
                hovertemplate=(
                    f"Month: %{{x|%b %Y}}<br>{application}: "
                    "%{y:,.0f} units<extra></extra>"
                ),
            ),
            secondary_y=False,
        )
    powertrain_figure.add_trace(
        go.Scatter(
            x=ev_shift["Date"],
            y=ev_shift["EV_Share"],
            mode="lines",
            name="EV share",
            line=dict(color=OK, width=3, dash="dot"),
            hovertemplate="Month: %{x|%b %Y}<br>EV share: %{y:.1%}<extra></extra>",
        ),
        secondary_y=True,
    )
    powertrain_figure.update_layout(
        title="Powertrain application units and EV share",
        height=470,
        hovermode="x unified",
    )
    powertrain_figure.update_xaxes(title="Month")
    powertrain_figure.update_yaxes(title_text="Units", secondary_y=False)
    powertrain_figure.update_yaxes(title_text="EV share", tickformat=".1%", secondary_y=True)
    st.caption("How to read this: stacked areas show units by application; the dotted line tracks EV share.")
    render_plotly_chart(
        powertrain_figure,
        "units",
        hovermode="x unified",
        secondary_y_format=".1%",
    )

    adoption_figure = go.Figure(
        go.Scatter(
            x=ev_shift["Date"],
            y=ev_shift["EV_Adoption_Index"],
            mode="lines",
            name="EV adoption index",
            line=dict(color=PRIMARY, width=3),
            hovertemplate="Month: %{x|%b %Y}<br>EV adoption index: %{y:,.1f}<extra></extra>",
        )
    )
    adoption_figure.update_layout(
        title="EV adoption index",
        height=370,
        hovermode="x unified",
    )
    adoption_figure.update_xaxes(title="Month")
    adoption_figure.update_yaxes(title="EV adoption index")
    st.caption("How to read this: the index tracks the EV adoption measure over time.")
    render_plotly_chart(adoption_figure, "index points", value_format=",.1f", hovermode="x unified")
    st.info("EV share rose to about 10% by 2025 and has been flat since.")
