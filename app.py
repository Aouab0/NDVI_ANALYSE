import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

from processing import init_gee, get_soil_moisture_series, get_summer_sm_thumbs

# Configuration de la page
st.set_page_config(page_title="PFE - Analyse Humidité", layout="wide")

# Coordonnées centrales (ex: Plaine du Gharb)
CENTER_LAT, CENTER_LON = 34.3, -6.1 

st.title("💧 Analyse Interactive de l'Humidité du Sol (Indice NMDI/NDWI)")
st.info("Dessinez une zone agricole sur la carte, choisissez la période de votre étude, puis lancez l'extraction pour lier la saturation hydrique à la surexploitation des aquifères.")

with st.spinner("Connexion sécurisée à Google Earth Engine..."):
    init_gee()

# Initialisation de la mémoire pour figer les résultats
if "resultats_analyse" not in st.session_state:
    st.session_state.resultats_analyse = None

col_map, col_results = st.columns([1, 1])

with col_map:
    st.subheader("1. Zone d'étude et Période")
    
    # Choix de la période
    col_dates1, col_dates2 = st.columns(2)
    with col_dates1:
        start_date = st.date_input("Date de début", value=pd.to_datetime("2018-01-01"))
    with col_dates2:
        end_date = st.date_input("Date de fin", value=pd.to_datetime("2022-12-31"))

    # Initialisation de la carte
    m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=11)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Esri Satellite'
    ).add_to(m)
    
    # Outils de dessin
    Draw(
        export=False, position='topleft',
        draw_options={'polyline': False, 'rectangle': True, 'polygon': True, 'circle': False, 'marker': False, 'circlemarker': False},
        edit_options={'edit': False}
    ).add_to(m)
    
    # Affichage de la carte
    st_data = st_folium(m, height=450, use_container_width=True)

    drawn_geometry = None
    if st_data["last_active_drawing"]:
        drawn_geometry = st_data["last_active_drawing"]["geometry"] 

    # Bouton de lancement explicite
    if st.button("🚀 Lancer le calcul sur cette zone", type="primary", use_container_width=True):
        if not drawn_geometry:
            st.warning("⚠️ Veuillez d'abord dessiner un polygone sur la carte.")
        else:
            with st.spinner("Extraction de la dynamique hydrique depuis les serveurs Google..."):
                try:
                    df_sm = get_soil_moisture_series(drawn_geometry, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                    thumbs = get_summer_sm_thumbs(drawn_geometry, start_date.year, end_date.year)
                    
                    # On stocke les résultats dans la mémoire pour éviter les rechargements intempestifs
                    st.session_state.resultats_analyse = {
                        "df": df_sm,
                        "thumbs": thumbs,
                        "start_yr": start_date.year,
                        "end_yr": end_date.year
                    }
                except Exception as e:
                    st.error(f"Erreur d'extraction : {e}")

with col_results:
    st.subheader("2. Résultats de l'analyse")
    
    # Affichage figé des résultats stockés en mémoire
    if st.session_state.resultats_analyse:
        res = st.session_state.resultats_analyse
        df_local_sm = res["df"]
        thumbs_sm = res["thumbs"]
        s_yr, e_yr = res["start_yr"], res["end_yr"]
        
        if not df_local_sm.empty:
            # Création du graphique principal
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(df_local_sm.index, df_local_sm['NDWI'], color='#01665e', linewidth=2)
            ax.fill_between(df_local_sm.index, df_local_sm['NDWI'], color='#01665e', alpha=0.2)
            ax.set_title("Dynamique de l'humidité du sol (Indice NDWI/NMDI)")
            ax.set_ylabel("Indice")
            ax.grid(True, alpha=0.3)
            
            # Ajout de la barre de valeurs (Colorbar) demandée
            cmap = mpl.colors.LinearSegmentedColormap.from_list('ndwi_cmap', ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e'])
            norm = mpl.colors.Normalize(vmin=-0.2, vmax=0.4)
            sm_cbar = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm_cbar.set_array([])
            cbar = fig.colorbar(sm_cbar, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
            cbar.set_label('Humidité (Sec ➔ Saturé en eau)')
            
            st.pyplot(fig)
            
            st.download_button("📥 Exporter la série d'humidité (CSV)", data=df_local_sm.to_csv().encode('utf-8'), file_name='humidite_sol_locale.csv')
            
            # Affichage des miniatures estivales
            st.markdown("#### Cartes de Saturation en Eau (Juin - Août)")
            cols_img = st.columns(len(thumbs_sm))
            for idx, (year, url) in enumerate(thumbs_sm.items()):
                with cols_img[idx]:
                    st.markdown(f"**Été {year}**")
                    if url: 
                        st.image(url, use_container_width=True)
                    else: 
                        st.write("ND")
        else:
            st.warning("Aucune donnée valide trouvée pour cette période.")
    else:
        st.info("👈 Dessinez une zone sur la carte, ajustez les dates et cliquez sur le bouton pour lancer l'analyse. Les résultats resteront affichés ici tant que vous ne relancez pas un nouveau calcul.")
