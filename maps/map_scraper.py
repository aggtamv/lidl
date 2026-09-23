import geopandas as gpd
import pandas as pd
import osmnx as ox
import folium

tags = {"shop": "supermarket"}

stores = ox.features_from_place(
    "Thessaloniki, Greece",
    tags=tags
)

stores = stores[
    ["name", "brand", "geometry"]
].copy()

print(stores.head())

"""lidl = stores[
    stores["name"].str.contains("Lidl", case=False, na=False)
]

ab = stores[
    stores["name"].str.contains(
        "AB|Βασιλόπουλος|Vassilopoulos",
        case=False,
        na=False
    )
]

masoutis = stores[
    stores["name"].str.contains(
        "Μασούτης|Masoutis",
        case=False,
        na=False
    )
]"""