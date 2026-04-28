import streamlit as st
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

from processing import init_gee, get_ndvi_series, get_summer_ndvi_thumbs, create_ndvi_gif, get_soil_moisture_series, get_summer_sm_thumbs, create_sm_gif

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
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Analyse Globale par Cluster", 
    "✍️ 2. Analyse Interactive (NDVI)", 
    "💧 3. Analyse Humidité du Sol (NMDI)",
    "💧 4.Analyse NMDI global par cluster"
])

# ---------------------------------------------------------------------
# ONGLET 1 : ANALYSE GLOBALE PAR CLUSTER
# ---------------------------------------------------------------------
with tab1:
    st.markdown("### Comportement Agricole des Zones de Déformation")
    st.info("💡 **Aide PFE :** Utilisez cet onglet pour identifier la signature agricole de chaque cluster InSAR. Un NDVI élevé en été (juin-août) indique une culture intensive irriguée, ce qui justifie l'épuisement de la nappe et la subsidence inélastique.")
    
    # Initialisation de la mémoire de l'application
    if "extraction_lancee" not in st.session_state:
        st.session_state.extraction_lancee = False

    col_param, col_graph = st.columns([1, 3])
    
    with col_param:
        st.subheader("Paramètres Globaux")
        cluster_choisi = st.selectbox("Sélectionnez le Cluster InSAR à analyser :", [0, 1, 2, 3])
        start_date_g = st.date_input("Date de début (Global)", value=pd.to_datetime("2016-01-01"))
        end_date_g = st.date_input("Date de fin (Global)", value=pd.to_datetime("2022-12-31"))
        lancer_global = st.button("Lancer l'Extraction du Cluster", type="primary")

    # Si on clique sur le bouton principal, on active l'état
    if lancer_global:
        st.session_state.extraction_lancee = True

    with col_graph:
        if st.session_state.extraction_lancee:
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

                        # --- Graphique 2 & 3 : Phénologie et Tendances Estivales ---
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
                        
                        st.markdown("---")
                        st.markdown("### 🎬 Animation Spatio-Temporelle du Cluster")
                        st.info("Générez un GIF pour visualiser l'évolution agricole tous les 10 jours durant la saison chaude (Avril-Octobre).")
                        
                        col_gif1, col_gif2 = st.columns([1, 2])
                        with col_gif1:
                            annee_gif = st.selectbox("Sélectionnez l'année :", range(2018, 2023), index=2)
                            generer_gif = st.button("🪄 Générer le GIF Animé (2 FPS)", type="secondary")
                            
                        with col_gif2:
                            if generer_gif:
                                with st.spinner(f"Création de l'animation pour {annee_gif} (Téléchargement des images de {annee_gif}-04 à {annee_gif}-10)..."):
                                    try:
                                        gif_bytes = create_ndvi_gif(polygons, annee_gif)
                                        
                                        if gif_bytes:
                                            st.image(gif_bytes, use_container_width=True)
                                            st.download_button(
                                                label="📥 Télécharger le GIF pour la soutenance",
                                                data=gif_bytes,
                                                file_name=f'NDVI_Cluster{cluster_choisi}_{annee_gif}.gif',
                                                mime='image/gif'
                                            )
                                        else:
                                            st.warning("Pas assez de données valides (ou trop de nuages) pour créer un GIF sur cette période.")
                                    except Exception as e:
                                        st.error(f"Erreur lors de la création du GIF : {e}")

                        with st.expander("🧠 Comment utiliser ce GIF lors de la soutenance ?"):
                            st.write("""
                            **Mettez ce GIF dans votre présentation PowerPoint !**
                            Il permet de démontrer visuellement ce que les équations expriment.
                            - **La couleur :** Une saturation rapide vers le vert foncé avec des valeurs supérieures à 0.35 confirme l'irrigation intensive en plein été.
                            - **Le timing :** Le jury verra concrètement que le pic de verdissement sur le GIF correspond très exactement au Lag de subsidence (le "creux" InSAR) sur les graphes de corrélation croisée.
                            """)
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
        if drawn_geometry:
            st.success("Parcelle capturée ! Analyse en cours...")
            start_yr, end_yr = 2018, 2022
            
            with st.spinner("Extraction de la dynamique locale..."):
                try:
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
                    # Correction appliquée ici : utilisation de drawn_geometry
                    thumbs = get_summer_ndvi_thumbs(drawn_geometry, start_yr, end_yr)
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
# ---------------------------------------------------------------------
# ONGLET 3 : ANALYSE INTERACTIVE D'HUMIDITÉ DU SOL (NMDI)
# ---------------------------------------------------------------------
with tab3:
    st.markdown("### Évaluation de l'Humidité du Sol par l'Indice NMDI")
    st.info("💡 **Aide PFE :** Contrairement au NDVI qui mesure la chlorophylle, le NMDI (Normalized Multi-band Drought Index) exploite la différence entre deux bandes Infrarouge (SWIR 1 et SWIR 2) pour isoler l'humidité contenue dans la terre nue et atténuer l'effet de l'eau contenue dans les feuilles. C'est le meilleur indicateur pour prouver qu'un sol est saturé d'eau (irrigation).")
    
    col_map_sm, col_results_sm = st.columns([1, 1])
    
    with col_map_sm:
        m_sm = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=11)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri', name='Esri Satellite'
        ).add_to(m_sm)
        
        try:
            rgba_image, bbox = get_raster_overlay("clusters.tif")
            folium.raster_layers.ImageOverlay(
                image=rgba_image, bounds=bbox, opacity=0.5, name='Clusters InSAR'
            ).add_to(m_sm)
        except: pass
        
        Draw(
            export=False, position='topleft',
            draw_options={'polyline': False, 'rectangle': True, 'circle': False, 'marker': False, 'circlemarker': False},
            edit_options={'edit': False}
        ).add_to(m_sm)
        
        st_data_sm = st_folium(m_sm, height=500, use_container_width=True, key="sm_map")

    drawn_geometry_sm = None
    if st_data_sm["last_active_drawing"]:
        drawn_geometry_sm = st_data_sm["last_active_drawing"]["geometry"] 

    with col_results_sm:
        if drawn_geometry_sm:
            st.success("Parcelle capturée ! Extraction de l'humidité du sol en cours...")
            start_yr, end_yr = 2018, 2022
            
            with st.spinner("Extraction de la dynamique NMDI..."):
                try:
                    df_local_sm = get_soil_moisture_series(drawn_geometry_sm, f'{start_yr}-01-01', f'{end_yr}-12-31')
                    
                    fig, ax = plt.subplots(figsize=(10, 3.5))
                    # La couleur bleue représente l'humidité
                    ax.plot(df_local_sm.index, df_local_sm['NMDI'], color='#01665e', linewidth=2)
                    ax.set_title("Dynamique de l'humidité du sol (Indice NMDI)")
                    ax.set_ylabel("NMDI")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    st.download_button("📥 Exporter la série d'humidité (CSV)", data=df_local_sm.to_csv().encode('utf-8'), file_name='humidite_sol_locale.csv')
                except Exception as e:
                    st.error(f"Erreur d'extraction : {e}")

            with st.spinner("Génération des Cartes Thermiques d'Humidité Estivales..."):
                try:
                    thumbs_sm = get_summer_sm_thumbs(drawn_geometry_sm, start_yr, end_yr)
                    st.markdown("#### Saturation en Eau du Sol (Juin - Août)")
                    
                    cols_img_sm = st.columns(len(thumbs_sm))
                    for idx, (year, url) in enumerate(thumbs_sm.items()):
                        with cols_img_sm[idx]:
                            st.markdown(f"**Été {year}**")
                            if url: st.image(url, use_container_width=True)
                            else: st.write("ND")
                            
                    with st.expander("🧠 Comment exploiter l'humidité dans le PFE ?"):
                        st.write("Si les courbes NDVI (onglet 2) montrent une forte végétation, mais que la courbe NMDI ci-dessus montre **des valeurs très élevées en été** (sols saturés représentés en bleu sur les vignettes), cela confirme qu'il ne s'agit pas de plantes résistantes à la sécheresse, mais bien d'une irrigation artificielle massive. C'est l'argument final pour lier l'agriculture à l'épuisement de l'aquifère et à la déformation (InSAR).")
                except Exception as e:
                    st.error(f"Erreur images : {e}")
        else:
            st.info("👈 Dessinez une zone sur la carte pour évaluer l'humidité du sol.")
            
# ---------------------------------------------------------------------
# ONGLET 4 : ANALYSE GLOBALE DE L'HUMIDITÉ DU SOL PAR CLUSTER (NMDI)
# ---------------------------------------------------------------------
with tab4:
    st.markdown("### Saturation Hydrique des Sols (Indice NMDI)")
    st.info("💡 **Aide PFE :** Contrairement au NDVI qui mesure la verdure des feuilles, le NMDI exploite les bandes Infrarouge (SWIR) pour mesurer directement l'humidité contenue dans la terre. Un sol saturé en plein été est la preuve formelle d'une irrigation massive, confirmant l'épuisement de l'aquifère et la subsidence inélastique du cluster associé.")
    
    if "extraction_sm_lancee" not in st.session_state:
        st.session_state.extraction_sm_lancee = False

    col_param_sm, col_graph_sm = st.columns([1, 3])
    
    with col_param_sm:
        st.subheader("Paramètres NMDI")
        cluster_choisi_sm = st.selectbox("Sélectionnez le Cluster InSAR :", [0, 1, 2, 3], key="sm_select")
        start_date_sm = st.date_input("Date de début", value=pd.to_datetime("2016-01-01"), key="sm_start")
        end_date_sm = st.date_input("Date de fin", value=pd.to_datetime("2022-12-31"), key="sm_end")
        lancer_sm = st.button("Lancer l'Extraction de l'Humidité", type="primary", key="sm_btn")

    if lancer_sm:
        st.session_state.extraction_sm_lancee = True

    with col_graph_sm:
        if st.session_state.extraction_sm_lancee:
            polygons_sm = load_cluster_polygons("clusters.tif", cluster_choisi_sm)
            
            if polygons_sm is None:
                st.error("Cluster vide.")
            else:
                with st.spinner("Calcul de l'indice NMDI (Humidité du sol) sur GEE..."):
                    try:
                        df_sm = get_soil_moisture_series(polygons_sm, start_date_sm.strftime('%Y-%m-%d'), end_date_sm.strftime('%Y-%m-%d'))
                        
                        # --- Graphique 1 : Série Temporelle NMDI ---
                        fig_sm1, ax_sm1 = plt.subplots(figsize=(12, 4))
                        ax_sm1.plot(df_sm.index, df_sm['NMDI'], color='#01665e', linewidth=1.5)
                        ax_sm1.fill_between(df_sm.index, df_sm['NMDI'], color='#01665e', alpha=0.2)
                        ax_sm1.set_title(f"Évolution Humidité du Sol (NMDI) - Cluster {cluster_choisi_sm}")
                        ax_sm1.set_ylabel("NMDI")
                        ax_sm1.grid(True, alpha=0.3)
                        st.pyplot(fig_sm1)

                        # --- Graphique 2 & 3 : Phénologie Hydrique et Anomalies ---
                        col_sg1, col_sg2 = st.columns(2)
                        
                        with col_sg1:
                            fig_sm2, ax_sm2 = plt.subplots(figsize=(6, 4))
                            sns.boxplot(data=df_sm, x='Mois', y='NMDI', ax=ax_sm2, color="lightblue")
                            ax_sm2.set_title("Profil Hydrique Mensuel Moyen")
                            ax_sm2.set_xlabel("Mois de l'année")
                            ax_sm2.grid(True, alpha=0.3)
                            st.pyplot(fig_sm2)
                            
                            with st.expander("🧠 Interprétation PFE (Cycle de l'eau)"):
                                st.write("""
                                - **Comportement Naturel :** L'humidité devrait être haute en hiver (pluies) et s'effondrer en été.
                                - **Comportement Anthropique (Irrigation) :** Si les boîtes des mois d'été (6, 7, 8) restent hautes ou présentent des pics anormaux, cela prouve l'apport artificiel d'eau souterraine.
                                """)

                        with col_sg2:
                            summer_sm = df_sm[df_sm['Saison'] == 'Été (Irrigation)'].groupby('Année')['NMDI'].mean()
                            fig_sm3, ax_sm3 = plt.subplots(figsize=(6, 4))
                            summer_sm.plot(kind='bar', ax=ax_sm3, color='teal', edgecolor='black')
                            ax_sm3.set_title("Humidité Moyenne Estivale par Année")
                            ax_sm3.set_ylabel("NMDI Moyen (Juin-Août)")
                            ax_sm3.set_ylim(0.4, max(summer_sm)*1.1)
                            ax_sm3.grid(axis='y', alpha=0.3)
                            st.pyplot(fig_sm3)
                            
                            with st.expander("🧠 Interprétation PFE (Subsidence)"):
                                st.write("Reliez cet histogramme à vos composantes InSAR (IC). Les années où l'humidité estivale est exceptionnellement forte malgré les sécheresses régionales correspondent aux pics de pompage (et donc aux déformations inélastiques maximales).")

                        st.download_button("📥 Exporter les données NMDI", data=df_sm.to_csv().encode('utf-8'), file_name=f'nmdi_cluster_{cluster_choisi_sm}.csv', mime='text/csv')
                        
                        # --- Section GIF NMDI ---
                        st.markdown("---")
                        st.markdown("### 🎬 Animation Spatio-Temporelle de l'Humidité du Sol")
                        
                        col_sgif1, col_sgif2 = st.columns([1, 2])
                        with col_sgif1:
                            annee_gif_sm = st.selectbox("Année (NMDI) :", range(2018, 2023), index=2, key="sm_yr")
                            generer_gif_sm = st.button("🪄 Générer le GIF Humidité", type="secondary", key="sm_gif_btn")
                            
                        with col_sgif2:
                            if generer_gif_sm:
                                with st.spinner(f"Création de l'animation NMDI pour {annee_gif_sm}..."):
                                    try:
                                        gif_bytes_sm = create_sm_gif(polygons_sm, annee_gif_sm)
                                        if gif_bytes_sm:
                                            st.image(gif_bytes_sm, use_container_width=True)
                                            st.download_button("📥 Télécharger le GIF NMDI", data=gif_bytes_sm, file_name=f'NMDI_Cluster{cluster_choisi_sm}_{annee_gif_sm}.gif', mime='image/gif')
                                        else:
                                            st.warning("Pas assez d'images sans nuages pour cette période.")
                                    except Exception as e:
                                        st.error(f"Erreur : {e}")
                    except Exception as e:
                        st.error(f"Erreur Earth Engine : {e}")
