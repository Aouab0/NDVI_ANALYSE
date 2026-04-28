import streamlit as st
import rasterio
import pandas as pd
from rasterio.features import shapes
from shapely.geometry import shape
import matplotlib.pyplot as plt
from processing import init_gee, get_ndvi_series

st.set_page_config(page_title="Analyse NDVI par Cluster", layout="wide")

st.title("🌱 Analyse Spatiale du NDVI par Cluster (InSAR)")
st.write("Cette application extrait l'historique NDVI Sentinel-2 pour chaque cluster de déformation.")

# 1. Initialisation GEE
with st.spinner("Connexion à Google Earth Engine..."):
    init_gee()

# 2. Lecture du Raster local
@st.cache_data
def load_cluster_polygons(tif_path, target_cluster):
    """
    Lit le .tif et transforme les pixels d'un cluster spécifique en polygones vectoriels.
    """
    with rasterio.open(tif_path) as src:
        image = src.read(1)
        transform = src.transform
        
    # Masque strict sur le cluster choisi
    mask = (image == target_cluster)
    
    if not mask.any():
        return None
        
    # Vectorisation des pixels en polygones (GeoJSON natif)
    results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) 
        in enumerate(shapes(image, mask=mask, transform=transform))
    )
    
    # Extraction des coordonnées pour GEE
    polygons = []
    for res in results:
        polygons.append(res['geometry']['coordinates'])
        
    return polygons

# --- Interface Utilisateur ---
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Paramètres")
    cluster_choisi = st.selectbox("Sélectionnez un Cluster :", [0, 1, 2, 3])
    start_date = st.date_input("Date de début", value=pd.to_datetime("2015-06-01"))
    end_date = st.date_input("Date de fin", value=pd.to_datetime("2021-12-31"))
    
    lancer = st.button("Lancer l'analyse NDVI", type="primary")

with col2:
    if lancer:
        st.info(f"Vectorisation du Cluster {cluster_choisi} en cours...")
        polygons = load_cluster_polygons("clusters.tif", cluster_choisi)
        
        if polygons is None:
            st.error("Aucun pixel trouvé pour ce cluster.")
        else:
            st.success(f"Cluster vectorisé ! Interrogation de GEE pour la période sélectionnée...")
            
            with st.spinner("GEE calcule l'historique NDVI (cela peut prendre 1 à 2 minutes)..."):
                try:
                    # Conversion des dates pour GEE
                    start_str = start_date.strftime('%Y-%m-%d')
                    end_str = end_date.strftime('%Y-%m-%d')
                    
                    df_ndvi = get_ndvi_series(polygons, start_str, end_str)
                    
                    # --- Affichage du Graphique ---
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(df_ndvi.index, df_ndvi['NDVI'], color='green', linewidth=2, marker='.')
                    ax.set_title(f"Évolution de la vigueur végétale (NDVI) - Cluster {cluster_choisi}")
                    ax.set_ylabel("NDVI Moyen")
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    # Téléchargement des données
                    csv = df_ndvi.to_csv().encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger la série NDVI en CSV",
                        data=csv,
                        file_name=f'ndvi_cluster_{cluster_choisi}.csv',
                        mime='text/csv',
                    )
                    
                except Exception as e:
                    st.error(f"Une erreur GEE est survenue : {e}")
