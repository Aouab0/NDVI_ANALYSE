import streamlit as st
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

from processing import init_gee, get_ndvi_series, get_summer_ndvi_thumbs

st.set_page_config(page_title="PFE - Analyse Hydro-Agricole", layout="wide")

CLUSTER_COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']
CENTER_LAT, CENTER_LON = 34.3, -6.1 

st.title("🌾 Analyse Couplée : Subsidence InSAR & Dynamique Agricole (NDVI)")

with st.spinner("Connexion sécurisée à Google Earth Engine..."):
    init_gee()

@st.cache_data
def load_cluster_polygons(tif_path, target_cluster):
    with rasterio.open(tif_path) as src:
        image = src.read(1)
        transform = src.transform
        
    mask = (image == target_cluster)
    if not mask.any(): return None
        
    results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) in enumerate(shapes(image, mask=mask, transform=transform))
    )
    
    polygons = []
    for res in results:
        geom = shape(res['geometry'])
        # FILTRE : On ignore les minuscules pixels orphelins pour ne pas faire crasher GEE
        # (0.00005 degrés carré élimine le bruit InSAR isolé)
        if geom.area > 0.00005: 
            polygons.append(geom)
            
    if not polygons:
        return None
        
    # FUSION : On combine tout en un seul objet propre et on simplifie les bordures
    merged_geom = unary_union(polygons).simplify(0.001)
    
    # mapping() convertit l'objet Shapely en un dictionnaire GeoJSON parfait pour GEE
    return mapping(merged_geom)

@st.cache_data
def get_raster_overlay(tif_path):
    with rasterio.open(tif_path) as src:
        image = src.read(1)
        bounds = src.bounds
        bbox = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
        rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        
        for c_id, hex_col in enumerate(CLUSTER_COLORS):
            h = hex_col.lstrip('#')
            rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            mask = (image == c_id)
            rgba[mask, 0], rgba[mask, 1], rgba[mask, 2], rgba[mask, 3] = rgb[0], rgb[1], rgb[2], 128
            
        return rgba, bbox

# =====================================================================
# --- ONGLETS DE L'APPLICATION ---
# =====================================================================
tab1, tab2 = st.tabs(["📊 1. Analyse Globale par Cluster (Extraction PFE)", "✍️ 2. Analyse Interactive (Ciblage Parcelle)"])

# ---------------------------------------------------------------------
# ONGLET 1 : ANALYSE GLOBALE PAR CLUSTER (Maintenant Rempli !)
# ---------------------------------------------------------------------
with tab1:
    st.markdown("### Comportement Agricole des Zones de Déformation")
    st.info("💡 **Aide PFE :** Utilisez cet onglet pour identifier la signature agricole de chaque cluster InSAR. Un NDVI élevé en été (juin-août) indique une culture intensive irriguée, ce qui justifie l'épuisement de la nappe et la subsidence inélastique.")
    
    col_param, col_graph = st.columns([1, 3])
    
    with col_param:
        st.subheader("Paramètres Globaux")
        cluster_choisi = st.selectbox("Sélectionnez le Cluster InSAR à analyser :", [0, 1, 2, 3])
        start_date_g = st.date_input("Date de début (Global)", value=pd.to_datetime("2016-01-01"))
        end_date_g = st.date_input("Date de fin (Global)", value=pd.to_datetime("2022-12-31"))
        lancer_global = st.button("Lancer l'Extraction du Cluster", type="primary")

    with col_graph:
        if lancer_global:
            st.info(f"Création de la géométrie pour le Cluster {cluster_choisi}...")
            polygons = load_cluster_polygons("clusters.tif", cluster_choisi)
            
            if polygons is None:
                st.error("Cluster vide.")
            else:
                with st.spinner("Extraction de la série GEE (cela peut prendre du temps vue la taille du cluster)..."):
                    try:
                        df_global = get_ndvi_series(polygons, start_date_g.strftime('%Y-%m-%d'), end_date_g.strftime('%Y-%m-%d'))
                        
                        # --- Graphique 1 : Série Temporelle Complète ---
                        fig1, ax1 = plt.subplots(figsize=(12, 4))
                        ax1.plot(df_global.index, df_global['NDVI'], color=CLUSTER_COLORS[cluster_choisi], linewidth=1.5)
                        ax1.fill_between(df_global.index, df_global['NDVI'], color=CLUSTER_COLORS[cluster_choisi], alpha=0.2)
                        ax1.set_title(f"Évolution NDVI Continue - Cluster {cluster_choisi}")
                        ax1.grid(True, alpha=0.3)
                        st.pyplot(fig1)

                        # --- Graphique 2 & 3 : Phénologie et Tendances Estivales (Idéal pour le PFE) ---
                        col_g1, col_g2 = st.columns(2)
                        
                        with col_g1:
                            fig2, ax2 = plt.subplots(figsize=(6, 4))
                            sns.boxplot(data=df_global, x='Mois', y='NDVI', ax=ax2, color="lightgreen")
                            ax2.set_title("Profil Phénologique (Cycle Agricole Moyen)")
                            ax2.set_xlabel("Mois de l'année")
                            ax2.grid(True, alpha=0.3)
                            st.pyplot(fig2)
                            
                            with st.expander("🧠 Interprétation PFE (Phénologie)"):
                                st.write("""
                                - **Si le pic est en Février/Mars :** Cultures d'hiver (pluviales). Faible impact sur la nappe.
                                - **Si le pic est en Juillet/Août :** Cultures d'été (maraîchage, maïs, arboriculture). **Forte dépendance au pompage**. La subsidence (IC2) devrait y être la plus forte.
                                """)

                        with col_g2:
                            # Calcul de l'anomalie estivale moyenne
                            summer_ndvi = df_global[df_global['Saison'] == 'Été (Irrigation)'].groupby('Année')['NDVI'].mean()
                            
                            fig3, ax3 = plt.subplots(figsize=(6, 4))
                            summer_ndvi.plot(kind='bar', ax=ax3, color='orange', edgecolor='black')
                            ax3.set_title("Moyenne du NDVI Estival (Juin-Août) par Année")
                            ax3.set_ylabel("NDVI Moyen Estival")
                            ax3.set_ylim(0, max(summer_ndvi)*1.2)
                            ax3.grid(axis='y', alpha=0.3)
                            st.pyplot(fig3)
                            
                            with st.expander("🧠 Interprétation PFE (Pression Hydrique)"):
                                st.write("""
                                - Les variations annuelles de l'histogramme montrent les **années de stress hydrique**.
                                - **Corrélation à chercher :** Une année avec un fort NDVI estival implique un pompage massif. Vérifie si la chute de ta courbe InSAR s'accélère durant ces années spécifiques.
                                """)
                                
                        st.download_button("📥 Exporter les données du Cluster (CSV)", data=df_global.to_csv().encode('utf-8'), file_name=f'ndvi_cluster_{cluster_choisi}.csv', mime='text/csv')

                    except Exception as e:
                        st.error(f"Erreur d'extraction GEE : {e}")

# ---------------------------------------------------------------------
# ONGLET 2 : ANALYSE INTERACTIVE SUR CARTE
# ---------------------------------------------------------------------
with tab2:
    st.markdown("### Analyse Ciblée par Parcelle Agricole")
    st.info("Utilisez l'outil polygone (carré noir sur la carte) pour dessiner sur une parcelle. Identifiez visuellement si la zone a été irriguée pendant la sécheresse.")
    
    col_map, col_results = st.columns([1, 1])
    
    with col_map:
        m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=11)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri', name='Esri Satellite'
        ).add_to(m)
        
        try:
            rgba_image, bbox = get_raster_overlay("clusters.tif")
            folium.raster_layers.ImageOverlay(
                image=rgba_image, bounds=bbox, opacity=0.5, name='Clusters InSAR'
            ).add_to(m)
        except: pass
        
        Draw(
            export=False, position='topleft',
            draw_options={'polyline': False, 'rectangle': True, 'circle': False, 'marker': False, 'circlemarker': False},
            edit_options={'edit': False}
        ).add_to(m)
        
        st_data = st_folium(m, height=500, use_container_width=True)

    drawn_geometry = None
    if st_data["last_active_drawing"]:
        drawn_geometry = st_data["last_active_drawing"]["geometry"] 

    with col_results:
        if drawn_geometry: # <--- Modification ici
            st.success("Parcelle capturée ! Analyse en cours...")
            start_yr, end_yr = 2018, 2022
            
            with st.spinner("Extraction de la dynamique locale..."):
                try:
                    # On passe "drawn_geometry" directement à GEE
                    df_local = get_ndvi_series(drawn_geometry, f'{start_yr}-01-01', f'{end_yr}-12-31')
                    
                    fig, ax = plt.subplots(figsize=(10, 3.5))
                    ax.plot(df_local.index, df_local['NDVI'], color='darkgreen', linewidth=2)
                    ax.set_title("Dynamique NDVI de la parcelle ciblée")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    st.download_button("📥 Exporter série parcelle", data=df_local.to_csv().encode('utf-8'), file_name='parcelle_locale.csv')
                except Exception as e:
                    st.error(f"Erreur : {e}")

            with st.spinner("Génération des Cartes NDVI Estivales..."):
                try:
                    thumbs = get_summer_ndvi_thumbs(drawn_polygon, start_yr, end_yr)
                    st.markdown("#### Couvert Végétal en Été (Preuve d'Irrigation)")
                    
                    cols_img = st.columns(len(thumbs))
                    for idx, (year, url) in enumerate(thumbs.items()):
                        with cols_img[idx]:
                            st.markdown(f"**Été {year}**")
                            if url: st.image(url, use_container_width=True)
                            else: st.write("ND")
                            
                    with st.expander("🧠 Comment insérer ceci dans le PFE ?"):
                        st.write("Faites des captures d'écran de ces vignettes estivales. Si la parcelle est rouge vif en 2018 et vert foncé en 2020, cela indique un changement de pratique agricole (passage au maraîchage d'été). Montrez dans votre rapport que **l'apparition de ces tâches vertes en été coïncide spatialement avec l'aggravation de la subsidence locale**.")
                except Exception as e:
                    st.error(f"Erreur images : {e}")
        else:
            st.info("👈 Dessinez une zone sur la carte pour générer les graphiques locaux.")
