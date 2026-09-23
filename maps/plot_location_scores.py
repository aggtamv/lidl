"""Plot Lidl demographics, road network, and equal-weight demand scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
from adjustText import adjust_text
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "lidl_location_scores.gpkg"
DEFAULT_OUTPUT = SCRIPT_DIR / "lidl_demand_scores.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Lidl demand scores, demographics, and road network."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the interactive Matplotlib window after saving.",
    )
    return parser.parse_args()


def read_optional_layer(path: Path, layer: str) -> gpd.GeoDataFrame | None:
    layers = set(gpd.list_layers(path)["name"])
    return gpd.read_file(path, layer=layer) if layer in layers else None


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Score dataset not found: {args.input}. Run "
            "calculate_location_score.py first."
        )

    scores = gpd.read_file(args.input, layer="lidl_location_scores")
    bus_stops = read_optional_layer(args.input, "bus_stops")
    metro_stations = read_optional_layer(args.input, "metro_stations")

    required = {
        "store_label",
        "store_x_m",
        "store_y_m",
        "population",
        "population_density_km2",
        "demand_score",
    }
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError("Score layer is missing: " + ", ".join(sorted(missing)))

    boundary_wgs84 = (
        gpd.GeoSeries([scores.geometry.union_all()], crs=scores.crs)
        .to_crs("EPSG:4326")
        .iloc[0]
    )
    ox.settings.use_cache = True
    ox.settings.cache_folder = SCRIPT_DIR / "cache"
    road_graph = ox.graph_from_polygon(
        boundary_wgs84,
        network_type="drive",
        simplify=True,
        retain_all=True,
    )
    road_graph = ox.project_graph(road_graph, to_crs=scores.crs)
    fig, ax = ox.plot_graph(
        road_graph,
        show=False,
        close=False,
        node_size=0,
        edge_color="#999999",
        edge_linewidth=0.4,
        bgcolor="white",
        figsize=(15, 12),
    )

    scores.plot(
        ax=ax,
        column=scores["demand_score"] * 100,
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        alpha=0.42,
        edgecolor="#333333",
        linewidth=1.1,
        legend=True,
        legend_kwds={"label": "Demand score (0–100)", "shrink": 0.72},
        zorder=2,
    )

    store_points = gpd.GeoDataFrame(
        scores[["store_label"]].copy(),
        geometry=gpd.points_from_xy(scores["store_x_m"], scores["store_y_m"]),
        crs=scores.crs,
    )
    store_points.plot(
        ax=ax,
        color="#0050aa",
        edgecolor="white",
        linewidth=0.9,
        markersize=75,
        zorder=6,
    )

    if bus_stops is not None and not bus_stops.empty:
        bus_stops.to_crs(scores.crs).plot(
            ax=ax,
            color="#20a65a",
            markersize=7,
            alpha=0.65,
            zorder=4,
        )
    if metro_stations is not None and not metro_stations.empty:
        metro_stations.to_crs(scores.crs).plot(
            ax=ax,
            color="#6a1b9a",
            marker="s",
            edgecolor="white",
            markersize=42,
            zorder=5,
        )

    texts = []
    for store in scores.itertuples():
        label = (
            f"{store.store_label}\n"
            f"Demand: {store.demand_score * 100:.1f} | "
            f"Population: {int(store.population):,}\n"
            f"Density: {store.population_density_km2:,.0f}/km²"
        )
        texts.append(
            ax.text(
                store.store_x_m,
                store.store_y_m,
                label,
                fontsize=7,
                color="#102a43",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#0050aa",
                    "alpha": 0.88,
                    "linewidth": 0.55,
                },
                zorder=7,
            )
        )

    adjust_text(
        texts,
        ax=ax,
        arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.5},
        expand=(1.05, 1.15),
    )

    legend_items = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="#0050aa",
            markeredgecolor="white", markersize=8, label="Lidl"
        ),
        Line2D(
            [0], [0], color="#9b9b9b", linewidth=1, label="Drive network"
        ),
    ]
    if bus_stops is not None and not bus_stops.empty:
        legend_items.append(
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor="#20a65a",
                markersize=5, label="Bus stop"
            )
        )
    if metro_stations is not None and not metro_stations.empty:
        legend_items.append(
            Line2D(
                [0], [0], marker="s", color="none", markerfacecolor="#6a1b9a",
                markersize=7, label="Metro station"
            )
        )
    ax.legend(handles=legend_items, loc="lower left")
    ax.set_title(
        "Lidl catchment demand proxy, demographics, and transport network",
        fontsize=16,
    )
    ax.set_axis_off()
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=250, bbox_inches="tight")
    print(f"Saved: {args.output.resolve()}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
