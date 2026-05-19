import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback

# colors we use throughout the app
BG_COLOR = "#f7f8fa"
WHITE = "#ffffff"
BLUE = "#0969da"      # for receivers
ORANGE = "#e07b39"    # for senders
GREY = "#b8bfc9"
TEXT_COLOR = "#1c2230"
LIGHT_TEXT = "#5a6474"

# TODO maybe make these nicer later
COLORS_LIST = [
    "#0072B2", "#E69F00", "#56B4E9", "#D55E00",
    "#CC79A7", "#009999", "#7B2D8B", "#F0E442",
    "#3D5A80", "#CC8833", "#5B8DB8", "#E0A040",
]

MALE_COLOR = "#0072B2"
FEMALE_COLOR = "#CC79A7"

# load data - make sure the parquet file is in the same folder
df_all = pd.read_parquet("data_all_sexes.parquet", engine="fastparquet")

# filter out region codes (codes above 900 are world regions not countries)
df_all = df_all[(df_all["destination_code"] < 900) & (df_all["origin_code"] < 900)].copy()

# clean up the asterisks from some country names
df_all["destination"] = df_all["destination"].str.replace("*", "", regex=False)
df_all["origin"] = df_all["origin"].str.replace("*", "", regex=False)
# fix Turkey - dataset uses Türkiye (with ü) but Plotly only recognizes Turkey
df_all["destination"] = df_all["destination"].str.replace("Türkiye", "Turkey", regex=False)
df_all["origin"] = df_all["origin"].str.replace("Türkiye", "Turkey", regex=False)
df_all["migrant_stock"] = pd.to_numeric(df_all["migrant_stock"], errors="coerce")
df_all["year"] = df_all["year"].astype(int)

# main dataframe is both sexes combined
df = df_all[df_all["sex"] == "both_sexes"].copy()

YEARS = sorted(df["year"].unique())
COUNTRIES = sorted(df["destination"].dropna().unique())

print("data loaded, years:", YEARS)
print("number of countries:", len(COUNTRIES))

# precompute aggregations so the callbacks are faster
AGG_DEST = df.groupby(["destination", "year"], as_index=False)["migrant_stock"].sum().dropna()
AGG_ORIG = df.groupby(["origin", "year"], as_index=False)["migrant_stock"].sum().dropna()

FLOWS_INTO = df.groupby(["destination", "year", "origin"], as_index=False)["migrant_stock"].sum().dropna()
FLOWS_FROM = df.groupby(["origin", "year", "destination"], as_index=False)["migrant_stock"].sum().dropna()

# gender data (male/female split)
gender_df = df_all[df_all["sex"].isin(["male", "female"])].dropna(subset=["migrant_stock"])
GENDER_RECV = gender_df.groupby(["destination", "sex", "year"], as_index=False)["migrant_stock"].sum()
GENDER_SENT = gender_df.groupby(["origin", "sex", "year"], as_index=False)["migrant_stock"].sum()

GENDER_GLOBAL = GENDER_RECV.groupby(["sex", "year"], as_index=False)["migrant_stock"].sum()
GLOBAL_TREND = AGG_DEST.groupby("year", as_index=False)["migrant_stock"].sum()

# assign a color to each country for the time series chart
COUNTRY_COLOR = {}
for i, c in enumerate(COUNTRIES):
    COUNTRY_COLOR[c] = COLORS_LIST[i % len(COLORS_LIST)]

# country centroids for the map labels and the nearest-neighbour feature
# these are approximate lat/lon positions
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
    "Turkey": (38.9, 35.2), "Uganda": (1.4, 32.3),
    "Ukraine": (48.4, 31.2), "United Arab Emirates": (23.4, 53.8),
    "United Kingdom": (55.4, -3.4),
    "United States of America": (37.1, -95.7),
    "Uruguay": (-32.5, -55.8),
    "Venezuela (Bolivarian Republic of)": (6.4, -66.6),
    "Viet Nam": (14.1, 108.3), "Yemen": (15.6, 48.5),
    "Zambia": (-13.1, 27.8), "Zimbabwe": (-19.0, 29.2),
}

# only show labels on big countries so the map doesnt get too cluttered
LARGE_COUNTRIES = {
    "United States of America", "Canada", "Brazil", "Argentina",
    "Russian Federation", "Australia", "China", "India",
    "Saudi Arabia", "Mexico", "Iran (Islamic Republic of)",
    "Kazakhstan", "Peru", "South Africa", "Ethiopia",
    "Egypt", "Nigeria", "Algeria", "Sudan",
    "Venezuela (Bolivarian Republic of)", "Colombia",
    "Angola", "Mozambique", "Zambia",
}


# helper to shorten long country names for display
def shorten(name):
    # TODO: could use a dict instead of replace chain
    name = name.replace("United States of America", "USA")
    name = name.replace("United Kingdom", "UK")
    name = name.replace("Russian Federation", "Russia")
    name = name.replace("Syrian Arab Republic", "Syria")
    name = name.replace("Venezuela (Bolivarian Republic of)", "Venezuela")
    name = name.replace("Iran (Islamic Republic of)", "Iran")
    name = name.replace("China, Hong Kong SAR", "Hong Kong")
    name = name.replace("Viet Nam", "Vietnam")
    name = name.replace("Turkiye", "Turkey")
    name = name.replace("Türkiye", "Turkey")
    return name


# format big numbers nicely (e.g. 1500000 -> 1.5 M)
def fmt_m(v):
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f} M"
    elif v >= 1000:
        return f"{v / 1000:.0f} K"
    else:
        return str(int(v))


app = Dash(__name__, suppress_callback_exceptions=True)

# custom CSS - mostly for the slider and dropdown styling
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

# reusable card style
card_style = {
    "background": WHITE,
    "border": "1px solid #dde1e7",
    "borderRadius": "6px",
    "padding": "10px 12px",
    "display": "flex",
    "flexDirection": "column",
    "overflow": "hidden",
}

app.layout = html.Div(style={
    "background": BG_COLOR,
    "height": "100vh",
    "width": "100vw",
    "overflow": "hidden",
    "display": "flex",
    "flexDirection": "column",
    "fontFamily": "system-ui, -apple-system, sans-serif",
    "color": TEXT_COLOR,
    "boxSizing": "border-box",
}, children=[

    dcc.Store(id="sel-country", data=None),
    dcc.Store(id="map-zoom", data=1.0),
    dcc.Store(id="map-mode", data="global"),

    # header bar
    html.Div(style={
        "background": WHITE,
        "borderBottom": "1px solid #dde1e7",
        "padding": "6px 20px",
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "flexShrink": "0",
    }, children=[
        html.Div(style={"display": "flex", "flexDirection": "column", "gap": "1px"}, children=[
            html.Div("Global Migration Flows",
                     style={"fontSize": "15px", "fontWeight": "700", "color": TEXT_COLOR}),
            html.Div("Noman Shahzad · Stepan Pshenichnyi · Vasiliki Korai",
                     style={"fontSize": "10px", "color": LIGHT_TEXT}),
        ]),
        html.Div(id="kpi-pills", style={"display": "flex", "gap": "8px", "alignItems": "center"}),
        html.Div(style={"textAlign": "right"}, children=[
            html.Div("UN International Migrant Stock · 1990–2024",
                     style={"fontSize": "10px", "color": LIGHT_TEXT}),
            html.Div("Source: UN DESA · data.un.org",
                     style={"fontSize": "10px", "color": LIGHT_TEXT}),
        ]),
    ]),

    # year slider
    html.Div(style={
        "background": WHITE,
        "borderBottom": "1px solid #dde1e7",
        "padding": "4px 20px 8px",
        "flexShrink": "0",
    }, children=[
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "16px"}, children=[
            html.Div("Year", style={"color": TEXT_COLOR, "fontSize": "12px", "fontWeight": "600", "whiteSpace": "nowrap"}),
            dcc.Slider(
                id="year-slider",
                min=YEARS[0], max=YEARS[-1], step=None,
                marks={int(y): {"label": str(y), "style": {"fontSize": "11px", "color": LIGHT_TEXT}} for y in YEARS},
                value=2024,
                included=False,
            ),
        ]),
    ]),

    html.Div(style={
        "flex": "1",
        "overflow": "hidden",
        "display": "flex",
        "flexDirection": "column",
        "gap": "6px",
        "padding": "6px",
    }, children=[
        # top row: map + bar charts
        html.Div(style={"flex": "3", "display": "flex", "gap": "6px", "overflow": "hidden"}, children=[

            # map panel
            html.Div(style=dict(flex="3", **card_style), children=[
                html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "6px"}, children=[
                    html.Div(id="map-label", style={
                        "color": LIGHT_TEXT, "fontSize": "10px",
                        "letterSpacing": "0.5px", "textTransform": "uppercase",
                    }, children="Global Migrant Stock  ·  Click a country to explore"),
                    html.Div(id="map-toggles", style={"display": "none"}, children=[
                        html.Button("Origins", id="btn-origins", n_clicks=0,
                                    style={"background": BLUE, "color": "#fff", "border": "none",
                                           "borderRadius": "4px", "padding": "2px 10px",
                                           "cursor": "pointer", "fontSize": "10px"}),
                        html.Button("Destinations", id="btn-dest", n_clicks=0,
                                    style={"background": BG_COLOR, "color": ORANGE,
                                           "border": f"1px solid {ORANGE}",
                                           "borderRadius": "4px", "padding": "2px 10px",
                                           "cursor": "pointer", "fontSize": "10px"}),
                    ]),
                ]),
                dcc.Graph(
                    id="choropleth",
                    style={"flex": "1", "minHeight": "0"},
                    config={"displayModeBar": False, "responsive": True},
                ),
                html.Div(id="map-footer", style={"display": "none", "justifyContent": "flex-end", "marginTop": "4px"}, children=[
                    html.Button("Clear selection", id="clear-btn", n_clicks=0,
                                style={"background": BG_COLOR, "color": LIGHT_TEXT,
                                       "border": "1px solid #dde1e7", "borderRadius": "4px",
                                       "padding": "2px 8px", "cursor": "pointer", "fontSize": "10px"}),
                ]),
            ]),

            # right column: top receivers + top senders bars
            html.Div(style={"flex": "2", "display": "flex", "flexDirection": "column", "gap": "6px", "overflow": "hidden"}, children=[
                html.Div(style=dict(flex="1", **card_style), children=[
                    html.Div(id="label-recv", style={
                        "color": LIGHT_TEXT, "fontSize": "10px",
                        "textTransform": "uppercase", "marginBottom": "6px",
                    }, children="Top 5 Receivers Worldwide"),
                    dcc.Graph(id="bar-recv", style={"flex": "1", "minHeight": "0"}, config={"displayModeBar": False}),
                ]),
                html.Div(style=dict(flex="1", **card_style), children=[
                    html.Div(id="label-send", style={
                        "color": LIGHT_TEXT, "fontSize": "10px",
                        "textTransform": "uppercase", "marginBottom": "6px",
                    }, children="Top 5 Senders Worldwide"),
                    dcc.Graph(id="bar-send", style={"flex": "1", "minHeight": "0"}, config={"displayModeBar": False}),
                ]),
            ]),
        ]),

        # bottom row: gender chart + time series (hidden until a country is selected)
        html.Div(id="detail-panel", style={"display": "none"}, children=[
            html.Div(style=dict(flex="1", **card_style), children=[
                html.Div(id="gender-title", style={
                    "color": LIGHT_TEXT, "fontSize": "10px",
                    "textTransform": "uppercase", "marginBottom": "6px",
                }),
                dcc.Graph(id="gender-chart", style={"flex": "1", "minHeight": "0"}, config={"displayModeBar": False}),
            ]),
            html.Div(style=dict(flex="1", **card_style), children=[
                html.Div(id="label-timeseries", style={
                    "color": LIGHT_TEXT, "fontSize": "10px",
                    "textTransform": "uppercase", "marginBottom": "6px",
                }, children="Migration Trends  ·  1990–2024"),
                dcc.Graph(id="timeseries", style={"flex": "1", "minHeight": "0"}, config={"displayModeBar": False}),
            ]),
        ]),
    ]),
])


# -- CALLBACKS --

@callback(
    Output("sel-country", "data"),
    Input("choropleth", "clickData"),
    Input("clear-btn", "n_clicks"),
    State("sel-country", "data"),
)
def update_selection(map_click, clear_n, current):
    from dash import ctx
    if ctx.triggered_id == "clear-btn":
        return None
    if ctx.triggered_id == "choropleth" and map_click:
        clicked = map_click["points"][0].get("location")
        # clicking same country again deselects it
        if clicked == current:
            return None
        return clicked
    return current


@callback(
    Output("detail-panel", "style"),
    Input("sel-country", "data"),
)
def toggle_detail_panel(country):
    # always show the bottom panels regardless of selection
    # TODO: maybe hide when nothing is selected?
    return {"flex": "2", "display": "flex", "gap": "6px", "overflow": "hidden"}


@callback(
    Output("kpi-pills", "children"),
    Input("year-slider", "value"),
    Input("sel-country", "data"),
)
def update_kpi(year, country):
    yr_dest = AGG_DEST[AGG_DEST["year"] == year]
    yr_orig = AGG_ORIG[AGG_ORIG["year"] == year]

    # pill style helper - inline because its simple
    def make_pill(text, color):
        return html.Div(text, style={
            "background": BG_COLOR,
            "border": f"1px solid {color}",
            "borderRadius": "4px",
            "padding": "3px 10px",
            "fontSize": "11px",
            "color": color,
            "whiteSpace": "nowrap",
        })

    if country:
        recv = float(yr_dest.loc[yr_dest["destination"] == country, "migrant_stock"].sum())
        sent = float(yr_orig.loc[yr_orig["origin"] == country, "migrant_stock"].sum())
        return [
            make_pill(str(year), LIGHT_TEXT),
            make_pill(shorten(country), BLUE),
            make_pill(f"Received  {fmt_m(recv)}", BLUE),
            make_pill(f"Sent  {fmt_m(sent)}", ORANGE),
        ]

    total = float(yr_dest["migrant_stock"].sum())
    top_r = shorten(yr_dest.nlargest(1, "migrant_stock")["destination"].values[0])
    top_s = shorten(yr_orig.nlargest(1, "migrant_stock")["origin"].values[0])
    return [
        make_pill(str(year), LIGHT_TEXT),
        make_pill(f"Global total  {fmt_m(total)}", LIGHT_TEXT),
        make_pill(f"Top receiver  {top_r}", BLUE),
        make_pill(f"Top sender  {top_s}", ORANGE),
    ]


@callback(
    Output("map-zoom", "data"),
    Input("choropleth", "relayoutData"),
    State("map-zoom", "data"),
    prevent_initial_call=True,
)
def track_zoom(relayout, current_zoom):
    if relayout and "geo.projection.scale" in relayout:
        return float(relayout["geo.projection.scale"])
    return current_zoom


@callback(
    Output("map-mode", "data"),
    Output("map-toggles", "style"),
    Output("btn-origins", "style"),
    Output("btn-dest", "style"),
    Output("map-label", "children"),
    Output("map-footer", "style"),
    Input("btn-origins", "n_clicks"),
    Input("btn-dest", "n_clicks"),
    Input("sel-country", "data"),
    State("map-mode", "data"),
)
def update_map_controls(orig_n, dest_n, selected, current_mode):
    from dash import ctx

    # button styles depending on which is active
    def btn_style_origins(active):
        if active:
            return {"background": BLUE, "color": "#fff", "border": "none",
                    "borderRadius": "4px", "padding": "2px 10px", "cursor": "pointer", "fontSize": "10px"}
        return {"background": BG_COLOR, "color": BLUE, "border": f"1px solid {BLUE}",
                "borderRadius": "4px", "padding": "2px 10px", "cursor": "pointer", "fontSize": "10px"}

    def btn_style_dest(active):
        if active:
            return {"background": ORANGE, "color": "#fff", "border": "none",
                    "borderRadius": "4px", "padding": "2px 10px", "cursor": "pointer", "fontSize": "10px"}
        return {"background": BG_COLOR, "color": ORANGE, "border": f"1px solid {ORANGE}",
                "borderRadius": "4px", "padding": "2px 10px", "cursor": "pointer", "fontSize": "10px"}

    if not selected:
        return (
            "global",
            {"display": "none"},
            btn_style_origins(True), btn_style_dest(False),
            "Global Migrant Stock  ·  Click a country to explore",
            {"display": "none"},
        )

    if ctx.triggered_id == "btn-origins":
        mode = "origins"
    elif ctx.triggered_id == "btn-dest":
        mode = "destinations"
    elif ctx.triggered_id == "sel-country":
        mode = "origins"
    else:
        mode = current_mode

    label = f"Migration map  ·  {shorten(selected)}"
    return (
        mode,
        {"display": "flex", "gap": "4px"},
        btn_style_origins(mode == "origins"),
        btn_style_dest(mode == "destinations"),
        label,
        {"display": "flex", "justifyContent": "flex-end", "marginTop": "4px"},
    )


@callback(
    Output("choropleth", "figure"),
    Input("year-slider", "value"),
    Input("sel-country", "data"),
    Input("map-zoom", "data"),
    Input("map-mode", "data"),
)
def update_map(year, selected, zoom_scale, map_mode):
    agg = AGG_DEST[AGG_DEST["year"] == year].copy()
    fig = go.Figure()

    # colorbar settings
    log_ticks = [3, 4, 5, 6, 7]
    log_text = ["1 K", "10 K", "100 K", "1 M", "10 M"]

    def make_colorbar(title, tickvals, ticktext, x=1.0):
        return dict(
            title=dict(text=title, font=dict(size=9, color=LIGHT_TEXT)),
            tickvals=tickvals,
            ticktext=ticktext,
            len=0.45,
            thickness=9,
            x=x,
            tickfont=dict(size=8, color=LIGHT_TEXT),
        )

    if map_mode == "global" or not selected:
        agg["log_stock"] = np.log10(agg["migrant_stock"].clip(lower=1))
        fig.add_trace(go.Choropleth(
            locations=agg["destination"],
            locationmode="country names",
            z=agg["log_stock"],
            customdata=agg["migrant_stock"],
            colorscale="Blues",
            zmin=3, zmax=np.log10(5e7),
            showscale=True,
            colorbar=make_colorbar("Migrants", log_ticks, log_text),
            marker=dict(line=dict(color="#ffffff", width=0.4)),
            hovertemplate="<b>%{location}</b><br>%{customdata:,.0f}<extra></extra>",
        ))

        # highlight selected country with a black border
        if selected and selected in agg["destination"].values:
            sel_log = float(agg.loc[agg["destination"] == selected, "log_stock"].values[0])
            fig.add_trace(go.Choropleth(
                locations=[selected], locationmode="country names",
                z=[sel_log], colorscale=[[0, BLUE], [1, BLUE]],
                zmin=3, zmax=np.log10(5e7),
                showscale=False, showlegend=False,
                marker=dict(line=dict(color="#000000", width=2.5)),
                hoverinfo="skip",
            ))

    elif map_mode == "origins":
        # show where migrants come from to the selected country
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
                colorbar=make_colorbar(f"Into {shorten(selected)}", log_ticks, log_text),
                marker=dict(line=dict(color="#ffffff", width=0.4)),
                hovertemplate="<b>%{location}</b><br>%{customdata:,.0f}<extra></extra>",
            ))

        fig.add_trace(go.Choropleth(
            locations=[selected], locationmode="country names",
            z=[1], colorscale=[[0, BLUE], [1, BLUE]],
            zmin=0, zmax=1,
            showscale=False, showlegend=False,
            marker=dict(line=dict(color="#000000", width=2.5)),
            hovertemplate=f"<b>{shorten(selected)}</b><extra></extra>",
        ))

    else:
        # show where migrants from the selected country go to
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
                colorbar=make_colorbar(f"From {shorten(selected)}", log_ticks, log_text),
                marker=dict(line=dict(color="#ffffff", width=0.4)),
                hovertemplate="<b>%{location}</b><br>%{customdata:,.0f}<extra></extra>",
            ))

        fig.add_trace(go.Choropleth(
            locations=[selected], locationmode="country names",
            z=[1], colorscale=[[0, BLUE], [1, BLUE]],
            zmin=0, zmax=1,
            showscale=False, showlegend=False,
            marker=dict(line=dict(color="#000000", width=2.5)),
            hovertemplate=f"<b>{shorten(selected)}</b><extra></extra>",
        ))

    # add country name labels on the map
    # only show labels for large countries to avoid clutter, or all if zoomed in
    if (zoom_scale or 1.0) > 2.5:
        visible = set(CENTROIDS.keys())
    else:
        visible = LARGE_COUNTRIES

    lats, lons, lbl = [], [], []
    for c in agg["destination"].tolist():
        if c in CENTROIDS and c in visible:
            lat, lon = CENTROIDS[c]
            lats.append(lat)
            lons.append(lon)
            lbl.append(shorten(c))

    fig.add_trace(go.Scattergeo(
        lat=lats, lon=lons, text=lbl, mode="text",
        textfont=dict(size=6, color="#444"),
        hoverinfo="skip", showlegend=False,
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR, size=11),
        margin=dict(l=0, r=30, t=0, b=0),
        uirevision=f"{selected or 'none'}-{map_mode}",
        geo=dict(
            showland=True, landcolor="#e8ecf0",
            showocean=True, oceancolor="#d6e8f7",
            showcountries=True, countrycolor="#ffffff",
            bgcolor="rgba(0,0,0,0)",
            projection_type="equirectangular",
            showframe=False,
            lataxis=dict(range=[-60, 85]),
            lonaxis=dict(range=[-180, 180]),
        ),
    )
    return fig


def build_bar_chart(data, bar_color):
    """builds a horizontal bar chart for top 5 countries"""
    fig = go.Figure(go.Bar(
        y=data["short"],
        x=data["value"],
        orientation="h",
        customdata=data["country"],
        marker=dict(color=bar_color, line=dict(width=0)),
        text=data["value"].apply(fmt_m),
        textposition="outside",
        textfont=dict(color=LIGHT_TEXT, size=9),
        hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR),
        xaxis=dict(
            tickformat=",.0s",
            gridcolor="#edf0f3",
            tickfont=dict(size=8, color=LIGHT_TEXT),
            range=[0, float(data["value"].max()) * 1.38],
            showline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0)",
            autorange="reversed",
            tickfont=dict(color=TEXT_COLOR, size=10),
        ),
        margin=dict(l=0, r=45, t=2, b=0),
        showlegend=False,
        bargap=0.3,
    )
    return fig


@callback(
    Output("bar-recv", "figure"),
    Output("label-recv", "children"),
    Input("year-slider", "value"),
    Input("sel-country", "data"),
)
def update_bar_recv(year, selected):
    if selected:
        # show top origins for this country
        subset = FLOWS_INTO[
            (FLOWS_INTO["destination"] == selected) & (FLOWS_INTO["year"] == year)
        ][["origin", "migrant_stock"]].nlargest(5, "migrant_stock").rename(
            columns={"origin": "country", "migrant_stock": "value"}
        )
        label = f"Top 5 Origins for {shorten(selected)}"
    else:
        subset = AGG_DEST[AGG_DEST["year"] == year].nlargest(5, "migrant_stock").rename(
            columns={"destination": "country", "migrant_stock": "value"}
        ).copy()
        label = "Top 5 Receivers Worldwide"

    subset["short"] = subset["country"].apply(shorten)
    return build_bar_chart(subset, BLUE), label


@callback(
    Output("bar-send", "figure"),
    Output("label-send", "children"),
    Input("year-slider", "value"),
    Input("sel-country", "data"),
)
def update_bar_send(year, selected):
    if selected:
        subset = FLOWS_FROM[
            (FLOWS_FROM["origin"] == selected) & (FLOWS_FROM["year"] == year)
        ][["destination", "migrant_stock"]].nlargest(5, "migrant_stock").rename(
            columns={"destination": "country", "migrant_stock": "value"}
        )
        label = f"{shorten(selected)} Emigrates to"
    else:
        subset = AGG_ORIG[AGG_ORIG["year"] == year].nlargest(5, "migrant_stock").rename(
            columns={"origin": "country", "migrant_stock": "value"}
        ).copy()
        label = "Top 5 Senders Worldwide"

    subset["short"] = subset["country"].apply(shorten)
    return build_bar_chart(subset, ORANGE), label


@callback(
    Output("gender-chart", "figure"),
    Output("gender-title", "children"),
    Input("year-slider", "value"),
    Input("sel-country", "data"),
)
def update_gender(year, selected):
    from plotly.subplots import make_subplots

    def get_val(df, sex_val):
        row = df[df["sex"] == sex_val]
        return float(row["migrant_stock"].values[0]) if not row.empty else 0.0

    if not selected:
        # global gender split
        yr_global = GENDER_GLOBAL[GENDER_GLOBAL["year"] == year]
        if yr_global.empty:
            return go.Figure(layout=go.Layout(paper_bgcolor="rgba(0,0,0,0)")), ""

        g_male = get_val(yr_global, "male")
        g_female = get_val(yr_global, "female")
        total = g_male + g_female if (g_male + g_female) > 0 else 1

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=["Female", "Male"],
            x=[g_female, g_male],
            orientation="h",
            marker_color=[FEMALE_COLOR, MALE_COLOR],
            text=[f"{g_female / total * 100:.1f}%", f"{g_male / total * 100:.1f}%"],
            textposition="outside",
            textfont=dict(size=9, color=LIGHT_TEXT),
            hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
            showlegend=False,
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_COLOR, size=10),
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
            bargap=0.4,
            xaxis=dict(
                range=[0, max(g_male, g_female) * 1.45],
                tickformat=",.0s",
                tickfont=dict(size=8, color=LIGHT_TEXT),
                gridcolor="#edf0f3",
                showline=False,
            ),
            yaxis=dict(
                tickfont=dict(size=10, color=TEXT_COLOR),
                showline=False,
                gridcolor="rgba(0,0,0,0)",
            ),
        )
        return fig, f"Global Gender Breakdown  ·  {year}"

    # country-level: show received vs sent by gender
    recv_yr = GENDER_RECV[(GENDER_RECV["destination"] == selected) & (GENDER_RECV["year"] == year)].copy()
    sent_yr = GENDER_SENT[(GENDER_SENT["origin"] == selected) & (GENDER_SENT["year"] == year)].copy()

    if recv_yr.empty and sent_yr.empty:
        return go.Figure(layout=go.Layout(paper_bgcolor="rgba(0,0,0,0)")), f"No gender data  ·  {shorten(selected)}"

    recv_male = get_val(recv_yr, "male")
    recv_female = get_val(recv_yr, "female")
    sent_male = get_val(sent_yr, "male")
    sent_female = get_val(sent_yr, "female")

    recv_total = recv_male + recv_female if (recv_male + recv_female) > 0 else 1
    sent_total = sent_male + sent_female if (sent_male + sent_female) > 0 else 1

    fig = make_subplots(rows=2, cols=1, subplot_titles=["Received", "Sent"],
                        shared_xaxes=True, vertical_spacing=0.18)

    max_val = max(recv_male, recv_female, sent_male, sent_female) * 1.45
    if max_val == 0:
        max_val = 1

    fig.add_trace(go.Bar(
        y=["Female", "Male"],
        x=[recv_female, recv_male],
        orientation="h",
        marker_color=[FEMALE_COLOR, MALE_COLOR],
        text=[f"{recv_female / recv_total * 100:.1f}%", f"{recv_male / recv_total * 100:.1f}%"],
        textposition="outside",
        textfont=dict(size=9, color=LIGHT_TEXT),
        hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=["Female", "Male"],
        x=[sent_female, sent_male],
        orientation="h",
        marker_color=[FEMALE_COLOR, MALE_COLOR],
        text=[f"{sent_female / sent_total * 100:.1f}%", f"{sent_male / sent_total * 100:.1f}%"],
        textposition="outside",
        textfont=dict(size=9, color=LIGHT_TEXT),
        hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
        showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR, size=10),
        margin=dict(l=0, r=0, t=28, b=0),
        showlegend=False,
        bargap=0.35,
    )
    for row_idx in [1, 2]:
        fig.update_xaxes(range=[0, max_val], tickformat=",.0s",
                         tickfont=dict(size=8, color=LIGHT_TEXT),
                         gridcolor="#edf0f3", showline=False, row=row_idx, col=1)
        fig.update_yaxes(tickfont=dict(size=10, color=TEXT_COLOR),
                         showline=False, gridcolor="rgba(0,0,0,0)", row=row_idx, col=1)
    for ann in fig.layout.annotations:
        ann.font.size = 10
        ann.font.color = LIGHT_TEXT

    return fig, f"Gender Breakdown  ·  {shorten(selected)}  ·  {year}"


def get_nearest_countries(selected, n=4):
    """returns n closest countries to selected based on lat/lon distance"""
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
    Output("timeseries", "figure"),
    Output("label-timeseries", "children"),
    Input("year-slider", "value"),
    Input("sel-country", "data"),
)
def update_timeseries(year, selected):
    fig = go.Figure()

    if selected:
        neighbours = get_nearest_countries(selected, n=4)
        countries = [selected] + neighbours
        agg = AGG_DEST[AGG_DEST["destination"].isin(countries)].copy()

        for country in countries:
            d = agg[agg["destination"] == country].sort_values("year")
            is_selected = country == selected
            fig.add_trace(go.Scatter(
                x=d["year"], y=d["migrant_stock"],
                name=shorten(country),
                mode="lines+markers",
                line=dict(
                    color=BLUE if is_selected else COUNTRY_COLOR.get(country, GREY),
                    width=3 if is_selected else 1.5,
                ),
                marker=dict(size=6 if is_selected else 3),
                opacity=1.0 if is_selected else 0.55,
                hovertemplate=f"{shorten(country)}: %{{y:,.0f}}<extra></extra>",
            ))
        label = f"Migration Trends  ·  {shorten(selected)}  vs. nearest neighbours"

    else:
        fig.add_trace(go.Scatter(
            x=GLOBAL_TREND["year"], y=GLOBAL_TREND["migrant_stock"],
            name="Global",
            mode="lines+markers",
            line=dict(color=BLUE, width=2.5),
            marker=dict(size=4, color=BLUE),
            hovertemplate="Global: %{y:,.0f}<extra></extra>",
        ))
        label = "Migration Trends  ·  Global Total  ·  1990–2024"

    # add vertical line for the currently selected year
    fig.add_vline(
        x=year,
        line_width=1,
        line_dash="dot",
        line_color=LIGHT_TEXT,
        annotation_text=str(year),
        annotation_font=dict(size=9, color=LIGHT_TEXT),
        annotation_position="top",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR, size=10),
        xaxis=dict(gridcolor="#edf0f3", tickvals=YEARS,
                   tickfont=dict(size=9, color=LIGHT_TEXT), showline=False),
        yaxis=dict(gridcolor="#edf0f3", tickformat=",.0s",
                   tickfont=dict(size=9, color=LIGHT_TEXT), showline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        margin=dict(l=0, r=0, t=16, b=0),
    )
    return fig, label


if __name__ == "__main__":
    app.run(debug=True)