"""Télécharge la géométrie de Spa (GeoJSON).

Source : dépôt libre bacinger/f1-circuits (licence MIT).
"""
import os
import urllib.request

URL = ("https://raw.githubusercontent.com/bacinger/f1-circuits/"
       "master/circuits/be-1925.geojson")
OUT = "data/raw/spa.geojson"


def main() -> None:
    os.makedirs("data/raw", exist_ok=True)
    urllib.request.urlretrieve(URL, OUT)
    print(f"Téléchargé : {OUT}")


if __name__ == "__main__":
    main()