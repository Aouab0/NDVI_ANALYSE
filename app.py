import streamlit as st
import rasterio
from rasterio.features import shapes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

from processing import init_gee, get_ndvi_series, get_summer_ndvi_thumbs

st.set_page_config(page_title="PFE - Analyse NDVI par Cluster", layout="wide")

# Couleurs des clusters (identiques à ton code local)
CLUSTER_COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e'] # Rouge, Bleu, Vert, Orange
# Coordonnées centrales approximatives du Gharb (à ajuster selon ta zone)
CENTER_LAT, CENTER_LON = 34.3, -6.1 

st.title("🌱 Analyse Spatiale du NDVI et Subsidence (InSAR)")

with st.spinner("Connexion à Google Earth Engine..."):
    init_gee()

@st.cache_data
def load_cluster_polygons(tif_path, target_cluster):
    # (Garde ta fonction load_cluster_polygons exacte ici comme dans le message précédent)
    pass 

@st.cache_data
def get_raster_overlay(tif_path):
    """Crée une image RGBA à partir du TIF pour l'afficher sur Folium"""
    with rasterio.open(tif_path) as src:
        image = src.read(1)
        bounds = src.bounds
        
        # Convertir les coordonnées du Bounding Box
        # Attention : Folium utilise [lat_min, lon_min], [lat_max, lon_max]
        # rasterio bounds = (left/lon_min, bottom/lat_min, right/lon_max, top/lat_max)
        bbox = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
        
        # Créer une image RGBA
        rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        
        # Appliquer les couleurs
        for c_id, hex_col in enumerate(CLUSTER_COLORS):
            # Convertir HEX en RGB
            h = hex_col.lstrip('#')
            rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            
            mask = (image == c_id)
            rgba[mask, 0] = rgb[0]
            rgba[mask, 1] = rgb[1]
            rgba[mask, 2] = rgb[2]
            rgba[mask, 3] = 128 # Transparence (Alpha) à 50%
            
        return rgba, bbox

# =====================================================================
# --- CRÉATION DES ONGLETS ---
# =====================================================================
tab1, tab2 = st.tabs(["📊 Analyse par Cluster (Globale)", "✍️ Analyse Interactive sur Carte (Locale)"])

# ---------------------------------------------------------
# ONGLET 1 : ANALYSE GLOBALE (Ton code précédent va ici)
# ---------------------------------------------------------
with tab1:
    st.write("Sélectionnez un cluster pour voir la dynamique végétale de toute la zone.")
    # ... Mets ici ton code précédent avec col1, col2, le selectbox et st.pyplot() ...

# ---------------------------------------------------------
# ONGLET 2 : ANALYSE INTERACTIVE (Nouveau)
# ---------------------------------------------------------
with tab2:
    st.markdown("### Dessinez une zone (polygone) sur la carte pour analyser la dynamique NDVI locale.")
    
    col_map, col_results = st.columns([1, 1])
    
    with col_map:
        # Création de la carte Folium
        m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=11)
        
        # Ajout du fond de carte Satellite
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Ajout de la couche Clusters (depuis le .tif)
        try:
            rgba_image, bbox = get_raster_overlay("clusters.tif")
            folium.raster_layers.ImageOverlay(
                image=rgba_image,
                bounds=bbox,
                opacity=0.6,
                name='Clusters InSAR',
                interactive=True,
                cross_origin=False,
            ).add_to(m)
        except Exception as e:
            st.warning("Le fichier clusters.tif n'a pas pu être chargé sur la carte.")
        
        # Ajout des outils de dessin (uniquement Polygone)
        Draw(
            export=False,
            position='topleft',
            draw_options={'polyline': False, 'rectangle': True, 'circle': False, 'marker': False, 'circlemarker': False},
            edit_options={'edit': False}
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        # Affichage de la carte dans Streamlit
        st_data = st_folium(m, height=500, use_container_width=True)

    # Récupération de la zone dessinée
    drawn_polygon = None
    if st_data["last_active_drawing"]:
        drawn_polygon = st_data["last_active_drawing"]["geometry"]["coordinates"]

    with col_results:
        if drawn_polygon:
            st.success("Zone sélectionnée ! Extraction des données en cours...")
            
            start_yr, end_yr = 2018, 2021 # Ajuste selon tes années d'étude
            
            with st.spinner("1. Extraction de la Série Temporelle (15 jours)..."):
                try:
                    df_local = get_ndvi_series(drawn_polygon, f'{start_yr}-01-01', f'{end_yr}-12-31')
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(df_local.index, df_local['NDVI'], color='green', linewidth=2)
                    ax.set_title(f"Série temporelle NDVI sur la parcelle dessinée")
                    ax.set_ylabel("NDVI Moyen")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Erreur série temporelle : {e}")

            with st.spinner("2. Génération des Cartes NDVI Estivales..."):
                try:
                    thumbs = get_summer_ndvi_thumbs(drawn_polygon, start_yr, end_yr)
                    
                    st.markdown("#### Évolution Spatiale du NDVI en Été (Juin - Août)")
                    # Création de colonnes dynamiques pour afficher les images côte à côte
                    cols_img = st.columns(len(thumbs))
                    for idx, (year, url) in enumerate(thumbs.items()):
                        with cols_img[idx]:
                            st.markdown(f"**{year}**")
                            if url:
                                st.image(url, use_container_width=True)
                            else:
                                st.write("Aucune donnée")
                except Exception as e:
                    st.error(f"Erreur cartes estivales : {e}")
                    
        else:
            st.info("👈 Utilisez l'outil de dessin (carré/polygone) sur la carte de gauche pour commencer.")
