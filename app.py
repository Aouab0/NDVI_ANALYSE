import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import pandas as pd
import json
import os
import io                     ### AJOUT ###
import numpy as np            ### AJOUT ###
import rasterio               ### AJOUT ###
from rasterio.plot import reshape_as_image  ### AJOUT ###
import matplotlib.pyplot as plt   ### AJOUT ###
from matplotlib import cm         ### AJOUT ###
from PIL import Image             ### AJOUT ###
import tempfile
from processing import init_gee, get_moisture_map_layers

# Configuration de la page
st.set_page_config(page_title="PFE - Cartographie Humidité", layout="wide")

CENTER_LAT, CENTER_LON = 34.3, -6.1 

st.title("💧 Cartographie Spatiale : NDMI & NDWI")
st.info("Analysez visuellement la saturation hydrique des parcelles. Le **NDMI** montre l'humidité interne du sol et des plantes (irrigation), tandis que le **NDWI** met en évidence l'eau libre en surface.")

with st.spinner("Connexion sécurisée à Google Earth Engine..."):
    init_gee()

# Mémoire de l'application pour conserver les couches affichées
if "map_layers" not in st.session_state:
    st.session_state.map_layers = None

col_parametres, col_carte = st.columns([1, 3])
def get_cumul_overlay(tif_path):
    """
    Charge le raster Cumul.tif (WGS84), applique la palette 'turbo'
    avec les valeurs négatives considérées comme maximales (inversion de l'échelle),
    et retourne le chemin du fichier PNG temporaire et les coordonnées de la bounding box.
    """
    with rasterio.open(tif_path) as src:
        data = src.read(1)  # bande unique
        mask = src.read_masks(1) > 0
        data_masked = np.ma.array(data, mask=~mask)
        vmin = data_masked.min()
        vmax = data_masked.max()
        
        # Normalisation classique : 0 (vmin) -> 1 (vmax)
        norm_data = (data_masked - vmin) / (vmax - vmin)
        # Inversion : les valeurs négatives (proches de vmin) auront 1,
        # les positives (proches de vmax) auront 0
        inv_norm = 1.0 - norm_data
        
        # Appliquer la palette 'turbo' (retourne RGBA)
        colored = cm.turbo(inv_norm)   # shape (H, W, 4)
        colored[..., 3] = np.where(mask, 1.0, 0.0)  # transparence pour nodata
        
        # Convertir en image PIL 8 bits
        img_array = (colored * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # Sauvegarder dans un fichier temporaire (persiste dans le conteneur)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp, format='PNG')
            tmp_path = tmp.name
        
        # Coordonnées de la bounding box (WGS84)
        bounds = [[src.bounds.bottom, src.bounds.left],
                  [src.bounds.top, src.bounds.right]]
        
        return tmp_path, bounds

with col_parametres:
    st.subheader("1. Paramètres")
    start_date = st.date_input("Date de début", value=pd.to_datetime("2020-06-01"))
    end_date = st.date_input("Date de fin", value=pd.to_datetime("2020-08-31"))
    
    st.markdown("---")
    st.subheader("2. Action")
    st.write("Dessinez un polygone sur la carte, puis lancez le calcul.")
    
    lancer_calcul = st.button("🚀 Afficher les couches", type="primary", use_container_width=True)
    
    if st.session_state.map_layers:
        st.success("✅ Couches générées avec succès !")
        st.markdown("""
        **Légendes :**
        - **NDMI (Vert/Bleu-vert) :** Saturation en eau du sol/plantes.
        - **NDWI (Bleu foncé) :** Eau libre en surface.
        - **Points rouges :** Points de mission de terrain.
        """)

with col_carte:
    m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=11)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Satellite Esri (Statique)'
    ).add_to(m)
    
    # 1. Chargement et ajout du fichier mission.geojson (Layer Points de terrain)
    if os.path.exists("mission.geojson"):
        try:
            with open("mission.geojson", "r", encoding="utf-8") as f:
                geo_data = json.load(f)
            
            folium.GeoJson(
                geo_data,
                name="Points de Mission",
                tooltip=folium.GeoJsonTooltip(
                    fields=["nom_point"],
                    aliases=["Point :"],
                    localize=True
                ),
                popup=folium.GeoJsonPopup(fields=["nom_point"]),
                marker=folium.Marker(icon=folium.Icon(color="red", icon="info-sign"))
            ).add_to(m)
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier GeoJSON : {e}")
    if os.path.exists("Cumul.tif"):
        cumul_path, cumul_bounds = get_cumul_overlay("Cumul.tif")
        folium.raster_layers.ImageOverlay(
            image=cumul_path,          # Chemin du fichier temporaire
            bounds=cumul_bounds,
            opacity=1.0,
            name='Cumul (palette turbo inversée)',
            show=True,
            mercator_project=False
        ).add_to(m)

    # 2. Ajout des couches dynamiques GEE
    if st.session_state.map_layers:
        folium.TileLayer(
            tiles=st.session_state.map_layers['rgb'],
            attr='Google Earth Engine',
            name='Couche RGB Moyen',
            overlay=True,
            show=True
        ).add_to(m)

        folium.TileLayer(
            tiles=st.session_state.map_layers['ndmi'],
            attr='Google Earth Engine',
            name='Couche NDMI (Humidité)',
            overlay=True,
            show=False
        ).add_to(m)
        
        folium.TileLayer(
            tiles=st.session_state.map_layers['ndwi'],
            attr='Google Earth Engine',
            name='Couche NDWI (Eau)',
            overlay=True,
            show=False
        ).add_to(m)
        
    folium.LayerControl(position='topright').add_to(m)
    
    Draw(
        export=False, position='topleft',
        draw_options={'polyline': False, 'rectangle': True, 'polygon': True, 'circle': False, 'marker': False, 'circlemarker': False},
        edit_options={'edit': False}
    ).add_to(m)
    
    st_data = st_folium(m, height=600, use_container_width=True)

# Logique de déclenchement
if lancer_calcul:
    drawn_geometry = None
    if st_data and st_data.get("last_active_drawing"):
        drawn_geometry = st_data["last_active_drawing"]["geometry"] 

    if not drawn_geometry:
        st.error("⚠️ Veuillez dessiner un polygone sur la carte.")
    else:
        with st.spinner("Calcul des couches en cours..."):
            try:
                layers = get_moisture_map_layers(drawn_geometry, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                st.session_state.map_layers = layers
                st.rerun() 
            except Exception as e:
                st.error(f"Erreur GEE : {e}")
