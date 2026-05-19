import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback

PAGE_BG = "#f7f8fa"
CARD_BG = "#ffffff"
BORDER  = "#dde1e7"
TEXT    = "#1c2230"
MUTED   = "#5a6474"
ACCENT  = "#0969da"   # blue  - receivers / selected
SENDER  = "#e07b39"   # orange - senders
UNSEL   = "#b8bfc9"   # grey  - dimmed items
GRID    = "#edf0f3"
CHORO   = "Blues"     # sequential scale for choropleth

CB_PALETTE = [
    "#0072B2", "#E69F00", "#56B4E9", "#D55E00",
    "#CC79A7", "#F0E442", "#7B2D8B", "#009999",
    "#3D5A80", "#CC8833", "#5B8DB8", "#8C69B2",
    "#4A7FA5", "#B07D3A", "#A05C7A", "#E0A040",
]

df_all = pd.read_parquet("data_all_sexes.parquet", engine="fastparquet")
df_all = df_all[
    (df_all["destination_code"] < 900) &
    (df_all["origin_code"]      < 900)
].copy()
df_all["destination"] = df_all["destination"].str.replace("*", "", regex=False)
df_all["origin"]      = df_all["origin"].str.replace("*", "", regex=False)
df_all["migrant_stock"] = pd.to_numeric(df_all["migrant_stock"], errors="coerce")
df_all["year"]          = df_all["year"].astype(int)

df = df_all[df_all["sex"] == "both_sexes"].copy()

YEARS     = sorted(df["year"].unique())
COUNTRIES = sorted(df["destination"].dropna().unique())

#precompute sums
AGG_DEST = (
    df.groupby(["destination", "year"], as_index=False)["migrant_stock"]
    .sum().dropna()
)
AGG_ORIG = (
    df.groupby(["origin", "year"], as_index=False)["migrant_stock"]
    .sum().dropna()
)
FLOWS_INTO = (
    df.groupby(["destination", "year", "origin"], as_index=False)["migrant_stock"]
    .sum().dropna()
)
FLOWS_FROM = (
    df.groupby(["origin", "year", "destination"], as_index=False)["migrant_stock"]
    .sum().dropna()
)
GENDER_RECV = (
    df_all[df_all["sex"].isin(["male", "female"])]
    .dropna(subset=["migrant_stock"])
    .groupby(["destination", "sex", "year"], as_index=False)["migrant_stock"].sum()
)
GENDER_SENT = (
    df_all[df_all["sex"].isin(["male", "female"])]
    .dropna(subset=["migrant_stock"])
    .groupby(["origin", "sex", "year"], as_index=False)["migrant_stock"].sum()
)

GENDER_GLOBAL = (
    GENDER_RECV.groupby(["sex", "year"], as_index=False)["migrant_stock"].sum()
)
GLOBAL_TREND = (
    AGG_DEST.groupby("year", as_index=False)["migrant_stock"].sum()
)

MALE_COLOR   = "#0072B2"
FEMALE_COLOR = "#CC79A7"

COUNTRY_COLOR = {
    c: CB_PALETTE[i % len(CB_PALETTE)]
    for i, c in enumerate(COUNTRIES)
}

#latitude,longitude
CENTROIDS = {
    "Afghanistan": (33.9, 67.7), "Albania": (41.2, 20.2),
    "Algeria": (28.0, 1.7), "Angola": (11.2, 17.9),
    "Argentina": (-38.4, -63.6), "Australia": (-25.3, 133.8),
    "Austria": (47.5, 14.6), "Azerbaijan": (40.1, 47.6),
    "Bangladesh": (23.7, 90.4), "Belarus": (53.7, 28.0),
    "Belgium": (50.5, 4.5), "Brazil": (-14.2, -51.9),
    "Bulgaria": (42.7, 25.5), "Cambodia": (12.6, 104.9),
    "Cameroon": (3.8, 11.5), "Canada": (56.1, -106.3),
    "Chile": (-35.7, -71.5), "China": (35.9, 104.2),
    "China, Hong Kong SAR": (22.3, 114.2),
    "Colombia": (4.6, -74.3), "Croatia": (45.1, 15.2),
    "Cuba": (21.5, -79.0), "Czech Republic": (49.8, 15.5),
    "Denmark": (56.3, 9.5), "Ecuador": (-1.8, -78.2),
    "Egypt": (26.8, 30.8), "Ethiopia": (9.1, 40.5),
    "Finland": (61.9, 25.7), "France": (46.2, 2.2),
    "Germany": (51.2, 10.5), "Ghana": (8.0, -1.0),
    "Greece": (39.1, 21.8), "Hungary": (47.2, 19.5),
    "India": (20.6, 78.9), "Indonesia": (-0.8, 113.9),
    "Iran (Islamic Republic of)": (32.4, 53.7),
    "Iraq": (33.2, 43.7), "Ireland": (53.4, -8.2),
    "Israel": (31.0, 34.9), "Italy": (41.9, 12.6),
    "Japan": (36.2, 138.3), "Jordan": (30.6, 36.2),
    "Kazakhstan": (48.0, 66.9), "Kenya": (-0.0, 37.9),
    "Kuwait": (29.3, 47.5), "Lebanon": (33.9, 35.9),
    "Libya": (26.3, 17.2), "Malaysia": (4.2, 101.9),
    "Mexico": (23.6, -102.6), "Morocco": (31.8, -7.1),
    "Mozambique": (-18.7, 35.5), "Myanmar": (16.9, 96.1),
    "Nepal": (28.4, 84.1), "Netherlands": (52.1, 5.3),
    "New Zealand": (-40.9, 174.9), "Nigeria": (9.1, 8.7),
    "Norway": (60.5, 8.5), "Pakistan": (30.4, 69.3),
    "Peru": (-9.2, -75.0), "Philippines": (12.9, 121.8),
    "Poland": (51.9, 19.1), "Portugal": (39.4, -8.2),
    "Qatar": (25.4, 51.2), "Romania": (45.9, 24.9),
    "Russian Federation": (61.5, 105.3),
    "Saudi Arabia": (23.9, 45.1), "Senegal": (14.5, -14.5),
    "Serbia": (44.0, 21.0), "Somalia": (5.2, 46.2),
    "South Africa": (-30.6, 22.9), "South Sudan": (7.9, 29.7),
    "Spain": (40.5, -3.7), "Sri Lanka": (7.9, 80.8),
    "Sudan": (12.9, 30.2), "Sweden": (60.1, 18.6),
    "Switzerland": (46.8, 8.2), "Syrian Arab Republic": (34.8, 38.9),
    "Thailand": (15.9, 101.0), "Tunisia": (33.9, 9.5),
    "Turkiye": (38.9, 35.2), "Uganda": (1.4, 32.3),
    "Ukraine": (48.4, 31.2), "United Arab Emirates": (23.4, 53.8),
    "United Kingdom": (55.4, -3.4),
    "United States of America": (37.1, -95.7),
    "Uruguay": (-32.5, -55.8),
    "Venezuela (Bolivarian Republic of)": (6.4, -66.6),
    "Viet Nam": (14.1, 108.3), "Yemen": (15.6, 48.5),
    "Zambia": (-13.1, 27.8), "Zimbabwe": (-19.0, 29.2),
}

#name label to fit inside the borders
LARGE_COUNTRIES = {
    "United States of America", "Canada", "Brazil", "Argentina",
    "Russian Federation", "Australia", "China", "India",
    "Saudi Arabia", "Mexico", "Iran (Islamic Republic of)",
    "Kazakhstan", "Peru", "South Africa", "Ethiopia",
    "Egypt", "Nigeria", "Algeria", "Sudan",
    "Venezuela (Bolivarian Republic of)", "Colombia",
    "Angola", "Mozambique", "Zambia",
}

def shorten(name: str) -> str:
    return (name
        .replace("United States of America", "USA")
        .replace("United Kingdom", "UK")
        .replace("Russian Federation", "Russia")
        .replace("Syrian Arab Republic", "Syria")
        .replace("Venezuela (Bolivarian Republic of)", "Venezuela")
        .replace("Iran (Islamic Republic of)", "Iran")
        .replace("China, Hong Kong SAR", "Hong Kong")
        .replace("Viet Nam", "Vietnam")
        .replace("Turkiye", "Turkey")
        .replace("Türkiye", "Turkey"))

#big number format
def fmt_m(v: float) -> str:
    if v >= 1e6:
        return f"{v / 1e6:.1f} M"
    if v >= 1e3:
        return f"{v / 1e3:.0f} K"
    return str(int(v))

CARD = dict(
    background=CARD_BG,
    border=f"1px solid {BORDER}",
    borderRadius="6px",
    padding="10px 12px",
    display="flex",
    flexDirection="column",
    overflow="hidden",
)

def section_label(text: str) -> html.Div:
    return html.Div(text, style=dict(
        color=MUTED,
        fontSize="10px",
        letterSpacing="0.5px",
        textTransform="uppercase",
        marginBottom="6px",
        fontFamily="inherit",
    ))

def kpi_pill(text: str, color: str) -> html.Div:
    return html.Div(text, style=dict(
        background=PAGE_BG,
        border=f"1px solid {color}",
        borderRadius="4px",
        padding="3px 10px",
        fontSize="11px",
        color=color,
        whiteSpace="nowrap",
        fontFamily="inherit",
    ))



app = Dash(__name__, suppress_callback_exceptions=True)

app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>Global Migration Flows</title>
{%favicon%}
{%css%}
<style>
  body { margin: 0; font-family: system-ui, -apple-system, sans-serif; }
  .Select-control {
    background-color: #fff !important; border-color: #dde1e7 !important;
    color: #1c2230 !important;
  }
  .Select-menu-outer {
    background-color: #fff !important; border-color: #dde1e7 !important;
    z-index: 9999 !important;
  }
  .Select-option {
    background-color: #fff !important; color: #1c2230 !important;
    font-size: 12px !important;
  }
  .Select-option:hover, .Select-option.is-focused {
    background-color: #e8f0fe !important; color: #0969da !important;
  }
  .Select-option.is-selected { background-color: #d2e3fc !important; }
  .Select-value-label { color: #1c2230 !important; }
  .Select-placeholder { color: #5a6474 !important; }
  .Select-input input { color: #1c2230 !important; background: #fff !important; }
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: #f7f8fa; }
  ::-webkit-scrollbar-thumb { background: #dde1e7; border-radius: 2px; }
  .rc-slider-handle {
    border-color: #0969da !important;
    background-color: #0969da !important;
  }
  .rc-slider-handle:hover, .rc-slider-handle-dragging {
    border-color: #0969da !important;
    box-shadow: 0 0 0 5px rgba(9,105,218,0.2) !important;
  }
</style>
{%scripts%}
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""

app.layout = html.Div(style=dict(
    background=PAGE_BG,
    height="100vh",
    width="100vw",
    overflow="hidden",
    display="flex",
    flexDirection="column",
    fontFamily="system-ui, -apple-system, sans-serif",
    color=TEXT,
    boxSizing="border-box",
), children=[

    dcc.Store(id="sel-country", data=None),
    dcc.Store(id="map-zoom",    data=1.0),
    dcc.Store(id="map-mode",    data="global"),

    #header
    html.Div(style=dict(
        background=CARD_BG,
        borderBottom=f"1px solid {BORDER}",
        padding="6px 20px",
        display="flex",
        justifyContent="space-between",
        alignItems="center",
        flexShrink="0",
    ), children=[
        html.Div(style=dict(display="flex", flexDirection="column", gap="1px"), children=[
            html.Div("Global Migration Flows",
                     style=dict(fontSize="15px", fontWeight="700", color=TEXT)),
            html.Div("Noman Shahzad · Stepan Pshenichnyi · Vasiliki Korai",
                     style=dict(fontSize="10px", color=MUTED)),
        ]),
        html.Div(id="kpi-pills",
                 style=dict(display="flex", gap="8px", alignItems="center")),
        html.Div(style=dict(textAlign="right"), children=[
            html.Div("UN International Migrant Stock · 1990–2024",
                     style=dict(fontSize="10px", color=MUTED)),
            html.Div("Source: UN DESA · data.un.org",
                     style=dict(fontSize="10px", color=MUTED)),
        ]),
    ]),

    # year slider
    html.Div(style=dict(
        background=CARD_BG,
        borderBottom=f"1px solid {BORDER}",
        padding="4px 20px 8px",
        flexShrink="0",
    ), children=[
        html.Div(style=dict(display="flex", alignItems="center", gap="16px"), children=[
            html.Div("Year", style=dict(
                color=TEXT, fontSize="12px", fontWeight="600", whiteSpace="nowrap",
            )),
            dcc.Slider(
                id="year-slider",
                min=YEARS[0], max=YEARS[-1], step=None,
                marks={int(y): dict(label=str(y), style=dict(
                    fontSize="11px", color=MUTED,
                )) for y in YEARS},
                value=2024,
                included=False,
            ),
        ]),
    ]),

    html.Div(style=dict(
        flex="1",
        overflow="hidden",
        display="flex",
        flexDirection="column",
        gap="6px",
        padding="6px",
    ), children=[
        html.Div(style=dict(
            flex="3",
            display="flex",
            gap="6px",
            overflow="hidden",
        ), children=[
            html.Div(style=dict(flex="3", **CARD), children=[
                html.Div(style=dict(
                    display="flex", justifyContent="space-between",
                    alignItems="center", marginBottom="6px",
                ), children=[
                    html.Div(id="map-label", style=dict(
                        color=MUTED, fontSize="10px", letterSpacing="0.5px",
                        textTransform="uppercase", fontFamily="inherit",
                    ), children="Global Migrant Stock  ·  Click a country to explore"),
                    html.Div(id="map-toggles", style=dict(display="none"), children=[
                        html.Button("Origins",      id="btn-origins", n_clicks=0,
                                    style=dict(background=ACCENT, color="#fff",
                                               border="none", borderRadius="4px",
                                               padding="2px 10px", cursor="pointer",
                                               fontSize="10px", fontFamily="inherit")),
                        html.Button("Destinations", id="btn-dest",    n_clicks=0,
                                    style=dict(background=PAGE_BG, color=SENDER,
                                               border=f"1px solid {SENDER}",
                                               borderRadius="4px", padding="2px 10px",
                                               cursor="pointer", fontSize="10px",
                                               fontFamily="inherit")),
                    ]),
                ]),
                dcc.Graph(
                    id="choropleth",
                    style=dict(flex="1", minHeight="0"),
                    config=dict(displayModeBar=False, responsive=True),
                ),
                html.Div(id="map-footer", style=dict(
                    display="none", justifyContent="flex-end", marginTop="4px",
                ), children=[
                    html.Button(
                        "Clear selection",
                        id="clear-btn", n_clicks=0,
                        style=dict(
                            background=PAGE_BG, color=MUTED,
                            border=f"1px solid {BORDER}",
                            borderRadius="4px", padding="2px 8px",
                            cursor="pointer", fontSize="10px",
                            fontFamily="inherit",
                        ),
                    ),
                ]),
            ]),

            html.Div(style=dict(
                flex="2",
                display="flex",
                flexDirection="column",
                gap="6px",
                overflow="hidden",
            ), children=[
                html.Div(style=dict(flex="1", **CARD), children=[
                    html.Div(id="label-recv", style=dict(
                        color=MUTED, fontSize="10px", letterSpacing="0.5px",
                        textTransform="uppercase", marginBottom="6px",
                        fontFamily="inherit",
                    ), children="Top 5 Receivers Worldwide"),
                    dcc.Graph(
                        id="bar-recv",
                        style=dict(flex="1", minHeight="0"),
                        config=dict(displayModeBar=False),
                    ),
                ]),
                html.Div(style=dict(flex="1", **CARD), children=[
                    html.Div(id="label-send", style=dict(
                        color=MUTED, fontSize="10px", letterSpacing="0.5px",
                        textTransform="uppercase", marginBottom="6px",
                        fontFamily="inherit",
                    ), children="Top 5 Senders Worldwide"),
                    dcc.Graph(
                        id="bar-send",
                        style=dict(flex="1", minHeight="0"),
                        config=dict(displayModeBar=False),
                    ),
                ]),
            ]),
        ]),
        html.Div(id="detail-panel", style=dict(display="none"), children=[
            html.Div(style=dict(flex="1", **CARD), children=[
                html.Div(id="gender-title", style=dict(
                    color=MUTED, fontSize="10px",
                    letterSpacing="0.5px", textTransform="uppercase",
                    marginBottom="6px",
                )),
                dcc.Graph(
                    id="gender-chart",
                    style=dict(flex="1", minHeight="0"),
                    config=dict(displayModeBar=False),
                ),
            ]),

            html.Div(style=dict(flex="1", **CARD), children=[
                html.Div(id="label-timeseries", style=dict(
                    color=MUTED, fontSize="10px", letterSpacing="0.5px",
                    textTransform="uppercase", marginBottom="6px",
                    fontFamily="inherit",
                ), children="Migration Trends  ·  1990–2024"),
                dcc.Graph(
                    id="timeseries",
                    style=dict(flex="1", minHeight="0"),
                    config=dict(displayModeBar=False),
                ),
            ]),
        ]),
    ]),
])

@callback(
    Output("sel-country", "data"),
    Input("choropleth",   "clickData"),
    Input("clear-btn",    "n_clicks"),
    State("sel-country",  "data"),
)
def update_selection(map_click, clear_n, current):
    from dash import ctx
    if ctx.triggered_id == "clear-btn":
        return None
    if ctx.triggered_id == "choropleth" and map_click:
        clicked = map_click["points"][0].get("location")
        return None if clicked == current else clicked
    return current


@callback(
    Output("detail-panel", "style"),
    Input("sel-country",   "data"),
)
def toggle_detail_panel(country):
    return dict(flex="2", display="flex", gap="6px", overflow="hidden")


@callback(
    Output("kpi-pills",  "children"),
    Input("year-slider", "value"),
    Input("sel-country", "data"),
)
def update_kpi(year, country):
    yr_dest = AGG_DEST[AGG_DEST["year"] == year]
    yr_orig = AGG_ORIG[AGG_ORIG["year"] == year]

    if country:
        recv = float(yr_dest.loc[yr_dest["destination"] == country,
                                 "migrant_stock"].sum())
        sent = float(yr_orig.loc[yr_orig["origin"] == country,
                                 "migrant_stock"].sum())
        return [
            kpi_pill(str(year), MUTED),
            kpi_pill(shorten(country), ACCENT),
            kpi_pill(f"Received  {fmt_m(recv)}", ACCENT),
            kpi_pill(f"Sent  {fmt_m(sent)}", SENDER),
        ]

    total = float(yr_dest["migrant_stock"].sum())
    top_r = shorten(yr_dest.nlargest(1, "migrant_stock")["destination"].values[0])
    top_s = shorten(yr_orig.nlargest(1, "migrant_stock")["origin"].values[0])
    return [
        kpi_pill(str(year), MUTED),
        kpi_pill(f"Global total  {fmt_m(total)}", MUTED),
        kpi_pill(f"Top receiver  {top_r}", ACCENT),
        kpi_pill(f"Top sender  {top_s}", SENDER),
    ]

@callback(
    Output("map-zoom",   "data"),
    Input("choropleth",  "relayoutData"),
    State("map-zoom",    "data"),
    prevent_initial_call=True,
)
def track_zoom(relayout, current_zoom):
    if relayout and "geo.projection.scale" in relayout:
        return float(relayout["geo.projection.scale"])
    return current_zoom

@callback(
    Output("map-mode",     "data"),
    Output("map-toggles",  "style"),
    Output("btn-origins",  "style"),
    Output("btn-dest",     "style"),
    Output("map-label",    "children"),
    Output("map-footer",   "style"),
    Input("btn-origins",   "n_clicks"),
    Input("btn-dest",      "n_clicks"),
    Input("sel-country",   "data"),
    State("map-mode",      "data"),
)
def update_map_controls(orig_n, dest_n, selected, current_mode):
    from dash import ctx
    tog_hidden  = dict(display="none")
    tog_visible = dict(display="flex", gap="4px")

    def orig_style(active):
        if active:
            return dict(background=ACCENT, color="#fff", border="none",
                        borderRadius="4px", padding="2px 10px",
                        cursor="pointer", fontSize="10px", fontFamily="inherit")
        return dict(background=PAGE_BG, color=ACCENT,
                    border=f"1px solid {ACCENT}", borderRadius="4px",
                    padding="2px 10px", cursor="pointer",
                    fontSize="10px", fontFamily="inherit")

    def dest_style(active):
        if active:
            return dict(background=SENDER, color="#fff", border="none",
                        borderRadius="4px", padding="2px 10px",
                        cursor="pointer", fontSize="10px", fontFamily="inherit")
        return dict(background=PAGE_BG, color=SENDER,
                    border=f"1px solid {SENDER}", borderRadius="4px",
                    padding="2px 10px", cursor="pointer",
                    fontSize="10px", fontFamily="inherit")

    footer_hidden  = dict(display="none")
    footer_visible = dict(display="flex", justifyContent="flex-end", marginTop="4px")

    if not selected:
        return ("global", tog_hidden,
                orig_style(True), dest_style(False),
                "Global Migrant Stock  ·  Click a country to explore",
                footer_hidden)

    if ctx.triggered_id == "btn-origins":
        mode = "origins"
    elif ctx.triggered_id == "btn-dest":
        mode = "destinations"
    elif ctx.triggered_id == "sel-country":
        mode = "origins" 
    else:
        mode = current_mode

    label = f"Migration map  ·  {shorten(selected)}"
    return (mode, tog_visible,
            orig_style(mode == "origins"),
            dest_style(mode == "destinations"),
            label, footer_visible)

@callback(
    Output("choropleth", "figure"),
    Input("year-slider", "value"),
    Input("sel-country", "data"),
    Input("map-zoom",    "data"),
    Input("map-mode",    "data"),
)
def update_map(year, selected, zoom_scale, map_mode):
    agg = AGG_DEST[AGG_DEST["year"] == year].copy()
    fig = go.Figure()

    def make_colorbar(title, tickvals, ticktext, x=1.0):
        return dict(
            title=dict(text=title, font=dict(size=9, color=MUTED)),
            tickvals=tickvals, ticktext=ticktext,
            len=0.45, thickness=9, x=x,
            tickfont=dict(size=8, color=MUTED),
        )

    log_ticks = [3, 4, 5, 6, 7]
    log_text  = ["1 K", "10 K", "100 K", "1 M", "10 M"]

    if map_mode == "global" or not selected:
        agg["log_stock"] = np.log10(agg["migrant_stock"].clip(lower=1))
        fig.add_trace(go.Choropleth(
            locations=agg["destination"],
            locationmode="country names",
            z=agg["log_stock"],
            customdata=agg["migrant_stock"],
            colorscale=CHORO,
            zmin=3, zmax=np.log10(5e7),
            showscale=True,
            colorbar=make_colorbar("Migrants", log_ticks, log_text),
            marker=dict(line=dict(color="#ffffff", width=0.4)),
            hovertemplate="<b>%{location}</b><br>%{customdata:,.0f}<extra></extra>",
        ))

        if selected and selected in agg["destination"].values:
            sel_log = float(agg.loc[agg["destination"]==selected,"log_stock"].values[0])
            fig.add_trace(go.Choropleth(
                locations=[selected], locationmode="country names",
                z=[sel_log], colorscale=[[0, ACCENT],[1, ACCENT]],
                zmin=3, zmax=np.log10(5e7),
                showscale=False, showlegend=False,
                marker=dict(line=dict(color="#000000", width=2.5)),
                hoverinfo="skip",
            ))

    elif map_mode == "origins":
        flows = FLOWS_INTO[
            (FLOWS_INTO["destination"] == selected) & (FLOWS_INTO["year"] == year)
        ][["origin", "migrant_stock"]].copy()

        if not flows.empty:
            flows["log_stock"] = np.log10(flows["migrant_stock"].clip(lower=1))
            zmax_val = max(float(flows["log_stock"].max()), 3.1)
            fig.add_trace(go.Choropleth(
                locations=flows["origin"],
                locationmode="country names",
                z=flows["log_stock"],
                customdata=flows["migrant_stock"],
                colorscale="Blues",
                zmin=3, zmax=zmax_val,
                showscale=True,
                colorbar=make_colorbar(
                    f"Into {shorten(selected)}", log_ticks, log_text),
                marker=dict(line=dict(color="#ffffff", width=0.4)),
                hovertemplate="<b>%{location}</b><br>%{customdata:,.0f}<extra></extra>",
            ))

        fig.add_trace(go.Choropleth(
            locations=[selected], locationmode="country names",
            z=[1], colorscale=[[0, ACCENT],[1, ACCENT]],
            zmin=0, zmax=1,
            showscale=False, showlegend=False,
            marker=dict(line=dict(color="#000000", width=2.5)),
            hovertemplate=f"<b>{shorten(selected)}</b><extra></extra>",
        ))

    else:
        flows = FLOWS_FROM[
            (FLOWS_FROM["origin"] == selected) & (FLOWS_FROM["year"] == year)
        ][["destination", "migrant_stock"]].copy()

        if not flows.empty:
            flows["log_stock"] = np.log10(flows["migrant_stock"].clip(lower=1))
            zmax_val = max(float(flows["log_stock"].max()), 3.1)
            fig.add_trace(go.Choropleth(
                locations=flows["destination"],
                locationmode="country names",
                z=flows["log_stock"],
                customdata=flows["migrant_stock"],
                colorscale="Oranges",
                zmin=3, zmax=zmax_val,
                showscale=True,
                colorbar=make_colorbar(
                    f"From {shorten(selected)}", log_ticks, log_text),
                marker=dict(line=dict(color="#ffffff", width=0.4)),
                hovertemplate="<b>%{location}</b><br>%{customdata:,.0f}<extra></extra>",
            ))

        fig.add_trace(go.Choropleth(
            locations=[selected], locationmode="country names",
            z=[1], colorscale=[[0, ACCENT],[1, ACCENT]],
            zmin=0, zmax=1,
            showscale=False, showlegend=False,
            marker=dict(line=dict(color="#000000", width=2.5)),
            hovertemplate=f"<b>{shorten(selected)}</b><extra></extra>",
        ))

    visible = set(CENTROIDS.keys()) if (zoom_scale or 1.0) > 2.5 else LARGE_COUNTRIES
    lats, lons, lbl = [], [], []
    src_countries = agg["destination"].tolist()
    for c in src_countries:
        if c in CENTROIDS and c in visible:
            lat, lon = CENTROIDS[c]
            lats.append(lat); lons.append(lon)
            lbl.append(shorten(c))
    fig.add_trace(go.Scattergeo(
        lat=lats, lon=lons, text=lbl, mode="text",
        textfont=dict(size=6, color="#444"),
        hoverinfo="skip", showlegend=False,
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=11),
        margin=dict(l=0, r=30, t=0, b=0),
        uirevision=f"{selected or 'none'}-{map_mode}",
        geo=dict(
            showland=True,landcolor="#e8ecf0",
            showocean=True,oceancolor="#d6e8f7",
            showcountries=True,countrycolor="#ffffff",
            bgcolor="rgba(0,0,0,0)",
            projection_type="equirectangular",
            showframe=False,
            lataxis=dict(range=[-60, 85]),
            lonaxis=dict(range=[-180, 180]),
        ),
    )
    return fig


def _build_bar(data: pd.DataFrame, bar_color: str) -> go.Figure:
    fig = go.Figure(go.Bar(
        y=data["short"],
        x=data["value"],
        orientation="h",
        customdata=data["country"],
        marker=dict(color=bar_color, line=dict(width=0)),
        text=data["value"].apply(fmt_m),
        textposition="outside",
        textfont=dict(color=MUTED, size=9),
        hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        xaxis=dict(
            tickformat=",.0s", gridcolor=GRID,
            tickfont=dict(size=8, color=MUTED),
            range=[0, float(data["value"].max()) * 1.38],
            showline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0)",
            autorange="reversed",
            tickfont=dict(color=TEXT, size=10),
        ),
        margin=dict(l=0, r=45, t=2, b=0),
        showlegend=False,
        bargap=0.3,
    )
    return fig


@callback(
    Output("bar-recv",   "figure"),
    Output("label-recv", "children"),
    Input("year-slider", "value"),
    Input("sel-country", "data"),
)
def update_bar_recv(year, selected):
    if selected:
        subset = (
            FLOWS_INTO[
                (FLOWS_INTO["destination"] == selected) & (FLOWS_INTO["year"] == year)
            ][["origin", "migrant_stock"]]
            .nlargest(5, "migrant_stock")
            .rename(columns={"origin": "country", "migrant_stock": "value"})
        )
        label = f"Top 5 Origins for {shorten(selected)}"
    else:
        subset = (
            AGG_DEST[AGG_DEST["year"] == year]
            .nlargest(5, "migrant_stock")
            .rename(columns={"destination": "country", "migrant_stock": "value"})
            .copy()
        )
        label = "Top 5 Receivers Worldwide"

    subset["short"] = subset["country"].apply(shorten)
    return _build_bar(subset, ACCENT), label


@callback(
    Output("bar-send",   "figure"),
    Output("label-send", "children"),
    Input("year-slider", "value"),
    Input("sel-country", "data"),
)
def update_bar_send(year, selected):
    if selected:
        subset = (
            FLOWS_FROM[
                (FLOWS_FROM["origin"] == selected) & (FLOWS_FROM["year"] == year)
            ][["destination", "migrant_stock"]]
            .nlargest(5, "migrant_stock")
            .rename(columns={"destination": "country", "migrant_stock": "value"})
        )
        label = f"{shorten(selected)} Emigrates to"
    else:
        subset = (
            AGG_ORIG[AGG_ORIG["year"] == year]
            .nlargest(5, "migrant_stock")
            .rename(columns={"origin": "country", "migrant_stock": "value"})
            .copy()
        )
        label = "Top 5 Senders Worldwide"

    subset["short"] = subset["country"].apply(shorten)
    return _build_bar(subset, SENDER), label


@callback(
    Output("gender-chart", "figure"),
    Output("gender-title", "children"),
    Input("year-slider",   "value"),
    Input("sel-country",   "data"),
)
def update_gender(year, selected):
    from plotly.subplots import make_subplots

    def get_val(df, sex_val):
        row = df[df["sex"] == sex_val]
        return float(row["migrant_stock"].values[0]) if not row.empty else 0.0

    def _build_gender_fig(male_recv, female_recv, male_sent, female_sent, subtitle):
        recv_total = male_recv + female_recv or 1
        sent_total = male_sent + female_sent or 1

        recv_texts = [
            f"{female_recv / recv_total * 100:.1f}%",
            f"{male_recv   / recv_total * 100:.1f}%",
        ]
        sent_texts = [
            f"{female_sent / sent_total * 100:.1f}%",
            f"{male_sent   / sent_total * 100:.1f}%",
        ]

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Received", "Sent"],
            shared_xaxes=True,
            vertical_spacing=0.18,
        )

        max_val = max(male_recv, female_recv, male_sent, female_sent) * 1.45 or 1

        fig.add_trace(go.Bar(
            name="Received",
            y=["Female", "Male"],
            x=[female_recv, male_recv],
            orientation="h",
            marker_color=[FEMALE_COLOR, MALE_COLOR],
            text=recv_texts,
            textposition="outside",
            textfont=dict(size=9, color=MUTED),
            hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
            showlegend=False,
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            name="Sent",
            y=["Female", "Male"],
            x=[female_sent, male_sent],
            orientation="h",
            marker_color=[FEMALE_COLOR, MALE_COLOR],
            text=sent_texts,
            textposition="outside",
            textfont=dict(size=9, color=MUTED),
            hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
            showlegend=False,
        ), row=2, col=1)

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT, size=10),
            margin=dict(l=0, r=0, t=28, b=0),
            showlegend=False,
            bargap=0.35,
        )
        for row_idx in [1, 2]:
            fig.update_xaxes(
                range=[0, max_val], tickformat=",.0s",
                tickfont=dict(size=8, color=MUTED),
                gridcolor=GRID, showline=False,
                row=row_idx, col=1,
            )
            fig.update_yaxes(
                tickfont=dict(size=10, color=TEXT),
                showline=False, gridcolor="rgba(0,0,0,0)",
                row=row_idx, col=1,
            )
        for ann in fig.layout.annotations:
            ann.font.size  = 10
            ann.font.color = MUTED
        return fig

    if not selected:
        yr_global = GENDER_GLOBAL[GENDER_GLOBAL["year"] == year]
        if yr_global.empty:
            return go.Figure(layout=go.Layout(paper_bgcolor="rgba(0,0,0,0)")), ""

        g_male   = get_val(yr_global, "male")
        g_female = get_val(yr_global, "female")
        total    = g_male + g_female or 1

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=["Female", "Male"],
            x=[g_female, g_male],
            orientation="h",
            marker_color=[FEMALE_COLOR, MALE_COLOR],
            text=[
                f"{g_female / total * 100:.1f}%",
                f"{g_male   / total * 100:.1f}%",
            ],
            textposition="outside",
            textfont=dict(size=9, color=MUTED),
            hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
            showlegend=False,
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT, size=10),
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
            bargap=0.4,
            xaxis=dict(
                range=[0, max(g_male, g_female) * 1.45],
                tickformat=",.0s", tickfont=dict(size=8, color=MUTED),
                gridcolor=GRID, showline=False,
            ),
            yaxis=dict(
                tickfont=dict(size=10, color=TEXT),
                showline=False, gridcolor="rgba(0,0,0,0)",
            ),
        )
        return fig, f"Global Gender Breakdown  ·  {year}"

    recv_yr = GENDER_RECV[
        (GENDER_RECV["destination"] == selected) & (GENDER_RECV["year"] == year)
    ].copy()
    sent_yr = GENDER_SENT[
        (GENDER_SENT["origin"] == selected) & (GENDER_SENT["year"] == year)
    ].copy()

    if recv_yr.empty and sent_yr.empty:
        return go.Figure(layout=go.Layout(paper_bgcolor="rgba(0,0,0,0)")), \
               f"No gender data  ·  {shorten(selected)}"

    recv_male   = get_val(recv_yr, "male")
    recv_female = get_val(recv_yr, "female")
    sent_male   = get_val(sent_yr, "male")
    sent_female = get_val(sent_yr, "female")

    fig = _build_gender_fig(recv_male, recv_female, sent_male, sent_female, selected)
    return fig, f"Gender Breakdown  ·  {shorten(selected)}  ·  {year}"


def _nearest_countries(selected: str, n: int = 4) -> list:
    if selected not in CENTROIDS:
        return []
    lat1, lon1 = CENTROIDS[selected]
    distances = []
    for country, (lat2, lon2) in CENTROIDS.items():
        if country == selected:
            continue
        dist = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5
        distances.append((dist, country))
    distances.sort()
    return [c for _, c in distances[:n]]


@callback(
    Output("timeseries",       "figure"),
    Output("label-timeseries", "children"),
    Input("year-slider",       "value"),
    Input("sel-country",       "data"),
)
def update_timeseries(year, selected):
    fig = go.Figure()

    if selected:
        neighbours = _nearest_countries(selected, n=4)
        countries  = [selected] + neighbours
        agg = AGG_DEST[AGG_DEST["destination"].isin(countries)].copy()

        for i, country in enumerate(countries):
            d = agg[agg["destination"] == country].sort_values("year")
            is_sel = country == selected
            fig.add_trace(go.Scatter(
                x=d["year"], y=d["migrant_stock"],
                name=shorten(country),
                mode="lines+markers",
                line=dict(
                    color=ACCENT if is_sel else COUNTRY_COLOR.get(country, MUTED),
                    width=3 if is_sel else 1.5,
                ),
                marker=dict(size=6 if is_sel else 3),
                opacity=1.0 if is_sel else 0.55,
                hovertemplate=f"{shorten(country)}: %{{y:,.0f}}<extra></extra>",
            ))
        label = f"Migration Trends  ·  {shorten(selected)}  vs. nearest neighbours"

    else:
        fig.add_trace(go.Scatter(
            x=GLOBAL_TREND["year"], y=GLOBAL_TREND["migrant_stock"],
            name="Global",
            mode="lines+markers",
            line=dict(color=ACCENT, width=2.5),
            marker=dict(size=4, color=ACCENT),
            hovertemplate="Global: %{y:,.0f}<extra></extra>",
        ))
        label = "Migration Trends  ·  Global Total  ·  1990–2024"

    fig.add_vline(
        x=year,
        line_width=1,
        line_dash="dot",
        line_color=MUTED,
        annotation_text=str(year),
        annotation_font=dict(size=9, color=MUTED),
        annotation_position="top",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=10),
        xaxis=dict(
            gridcolor=GRID, tickvals=YEARS,
            tickfont=dict(size=9, color=MUTED), showline=False,
        ),
        yaxis=dict(
            gridcolor=GRID, tickformat=",.0s",
            tickfont=dict(size=9, color=MUTED), showline=False,
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        margin=dict(l=0, r=0, t=16, b=0),
    )
    return fig, label

if __name__ == "__main__":
    app.run(debug=True)