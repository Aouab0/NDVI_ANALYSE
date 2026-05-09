import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import pandas as pd

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

with col_parametres:
    st.subheader("1. Paramètres")
    start_date = st.date_input("Date de début", value=pd.to_datetime("2020-06-01"))
    end_date = st.date_input("Date de fin", value=pd.to_datetime("2020-08-31"))
    
    st.markdown("---")
    st.subheader("2. Action")
    st.write("Dessinez un polygone sur la carte, puis lancez le calcul.")
    
    lancer_calcul = st.button("🚀 Afficher les couches", type="primary", use_container_width=True)
    
    if st.session_state.map_layers:
        st.success("✅ Couches générées avec succès ! Utilisez l'icône en haut à droite de la carte pour basculer entre le NDMI et le NDWI.")
        st.markdown("""
        **Légendes :**
        - **NDMI (Vert/Bleu-vert) :** Plus la couleur est foncée, plus la parcelle est saturée en eau (preuve d'irrigation massive en été).
        - **NDWI (Bleu foncé) :** Indique la présence d'eau à la surface même du sol.
        """)

with col_carte:
    # Initialisation de la carte
    m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=11)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Satellite de base'
    ).add_to(m)
    
    # Si les couches existent en mémoire, on les ajoute à la carte AVANT de l'afficher
    if st.session_state.map_layers:
        folium.TileLayer(
            tiles=st.session_state.map_layers['ndmi'],
            attr='Google Earth Engine',
            name='Couche NDMI (Humidité Sol/Plante)',
            overlay=True,
            show=True # Affiché par défaut
        ).add_to(m)
        
        folium.TileLayer(
            tiles=st.session_state.map_layers['ndwi'],
            attr='Google Earth Engine',
            name='Couche NDWI (Eau de surface)',
            overlay=True,
            show=False # Masqué par défaut pour ne pas mélanger les couleurs
        ).add_to(m)
        
    # Ajout du contrôleur de couches (c'est ce qui permet de cocher/décocher les images)
    folium.LayerControl(position='topright').add_to(m)
    
    # Outils de dessin
    Draw(
        export=False, position='topleft',
        draw_options={'polyline': False, 'rectangle': True, 'polygon': True, 'circle': False, 'marker': False, 'circlemarker': False},
        edit_options={'edit': False}
    ).add_to(m)
    
    # Rendu de la carte
    st_data = st_folium(m, height=600, use_container_width=True)

# Logique de déclenchement (Quand on clique sur le bouton)
if lancer_calcul:
    drawn_geometry = None
    if st_data and st_data.get("last_active_drawing"):
        drawn_geometry = st_data["last_active_drawing"]["geometry"] 

    if not drawn_geometry:
        st.error("⚠️ Vous devez d'abord dessiner un polygone (carré ou forme libre) sur la carte.")
    else:
        with st.spinner("Demande de calcul aux serveurs de Google (cela peut prendre quelques secondes)..."):
            try:
                # Récupération des URLs
                layers = get_moisture_map_layers(drawn_geometry, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                
                # Stockage en mémoire
                st.session_state.map_layers = layers
                
                # On force le rechargement de la page pour que la carte intègre les nouvelles couches
                st.rerun() 
            except Exception as e:
                st.error(f"Erreur lors du calcul GEE : {e}")
