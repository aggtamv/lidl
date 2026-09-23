from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox

INPUT = Path(__file__).with_name("lidl_catchments_demographics.gpkg")
OUTPUT = Path(__file__).with_name("population_by_lidl_catchment.png")

# Change this to another demographic column when required.
PLOT_FIELD = "population_density_km2"
PLOT_LABEL = "Estimated population per km²"

catchments = gpd.read_file(
    INPUT,
    layer="lidl_catchments_demographics",
)

# Download roads covering the complete catchment area.
boundary_wgs84 = (
    gpd.GeoSeries(
        [catchments.geometry.union_all()],
        crs=catchments.crs,
    )
    .to_crs("EPSG:4326")
    .iloc[0]
)

ox.settings.use_cache = True
ox.settings.cache_folder = Path(__file__).with_name("cache")

roads = ox.graph_from_polygon(
    boundary_wgs84,
    network_type="drive",
    simplify=True,
    retain_all=True,
)

roads = ox.project_graph(roads, to_crs=catchments.crs)

_, ax = ox.plot_graph(
    roads,
    show=False,
    close=False,
    node_size=0,
    edge_color="#999999",
    edge_linewidth=0.4,
    bgcolor="white",
    figsize=(15, 12),
)

catchments.plot(
    ax=ax,
    column=PLOT_FIELD,
    cmap="YlOrRd",
    alpha=0.55,
    edgecolor="#333333",
    linewidth=1.2,
    legend=True,
    legend_kwds={"label": PLOT_LABEL},
    zorder=2,
)

# Draw and label Lidl locations using coordinates saved in the catchment file.
for store in catchments.itertuples():
    ax.scatter(
        store.store_x_m,
        store.store_y_m,
        color="#0050aa",
        edgecolor="white",
        linewidth=0.8,
        s=65,
        zorder=4,
    )

    ax.annotate(
        store.store_label,
        (store.store_x_m, store.store_y_m),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=7,
        fontweight="bold",
        color="#003b73",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "#0050aa",
            "alpha": 0.8,
        },
        zorder=5,
    )

ax.set_title("Estimated population density by Lidl proxy catchment")
ax.set_axis_off()

plt.tight_layout()
plt.savefig(OUTPUT, dpi=250, bbox_inches="tight")
plt.show()

print(f"Saved map: {OUTPUT}")