import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. CONFIGURACIÓN Y CARGA DE DATOS
st.set_page_config(page_title="LoL Draft Predictor", layout="wide")
st.title("🏆 LoL Sequential Draft Picker & Predictor")

@st.cache_resource
def load_models():
    # Reemplaza con las rutas correctas si están en carpetas
    model_low = joblib.load("best_model_lowtier.joblib")
    model_apex = joblib.load("best_model_apex.joblib")
    return model_low, model_apex

@st.cache_data
def load_champ_data():
    # Simulación de datos. Tu CSV debe tener el nombre del campeón y su rol principal
    # Columnas: 'campeon', 'rol' (Top, Jungle, Mid, ADC, Support)
    try:
        df = pd.read_csv("datos_campeones.csv")
    except FileNotFoundError:
        # Dataframe de respaldo para pruebas
        df = pd.DataFrame({
            'campeon': ['Aatrox', 'Ahri', 'Ashe', 'Braum', 'Darius', 'Lee Sin', 'Jinx', 'Thresh', 'Orianna', 'Vayne'],
            'rol': ['Top', 'Mid', 'ADC', 'Support', 'Top', 'Jungle', 'ADC', 'Support', 'Mid', 'ADC']
        })
    return df

model_low, model_apex = load_models()
df_champs = load_champ_data()

# 2. SELECCIÓN DE ELO (INTERFAZ)
elo_side = st.sidebar.selectbox("Selecciona el Elo del Modelo:", ["Low Elo", "Apex (High Elo)"])
modelo_activo = model_apex if elo_side == "Apex (High Elo)" else model_low

# 3. LÓGICA DEL FLUJO DEL DRAFT (1B -> 2R -> 2B -> 2R -> 2B -> 1R)
# Definimos el orden de quién pickea en cada uno de los 10 turnos
DRAFT_ORDER = [
    {"turno": 1, "equipo": "Blue", "label": "Pick 1 Azul"},
    {"turno": 2, "equipo": "Red", "label": "Pick 1 Rojo"},
    {"turno": 3, "equipo": "Red", "label": "Pick 2 Rojo"},
    {"turno": 4, "equipo": "Blue", "label": "Pick 2 Azul"},
    {"turno": 5, "equipo": "Blue", "label": "Pick 3 Azul"},
    {"turno": 6, "equipo": "Red", "label": "Pick 3 Rojo"},
    {"turno": 7, "equipo": "Red", "label": "Pick 4 Rojo"},
    {"turno": 8, "equipo": "Blue", "label": "Pick 4 Azul"},
    {"turno": 9, "equipo": "Blue", "label": "Pick 5 Azul"},
    {"turno": 10, "equipo": "Red", "label": "Pick 5 Rojo"},
]

# Inicializar el estado de la sesión de Streamlit para el flujo
if "current_step" not in st.session_state:
    st.session_state.current_step = 0
    st.session_state.blue_team = []
    st.session_state.red_team = []
    st.session_state.picked_champions = set()

# Botón para reiniciar el draft
if st.sidebar.button("Reiniciar Draft"):
    st.session_state.current_step = 0
    st.session_state.blue_team = []
    st.session_state.red_team = []
    st.session_state.picked_champions = set()
    st.rerun()

# 4. FUNCIONES DE VALIDACIÓN AUXILIARES
def check_role_limit(team_champs, proposed_champ):
    proposed_role = df_champs[df_champs['campeon'] == proposed_champ]['rol'].values[0]
    roles_in_team = df_champs[df_champs['campeon'].isin(team_champs)]['rol'].tolist()
    return roles_in_team.count(proposed_role) < 2

# 5. SIMULACIÓN DE PROBABILIDADES (Aquí conectas tu modelo)
def calcular_metricas(blue_team, red_team, modelo):
    """
    Modifica esta función según cómo reciba los datos tu modelo .joblib.
    Generalmente requiere un vector One-Hot Encoding de los 10 campeones.
    """
    # Placeholder: Simulación de probabilidad basada en el número de campeones
    # Replázalo con: prediccion = modelo.predict_proba(X)
    base_winrate = 50.0
    blue_impact = len(blue_team) * 1.2  # Simulación de valor agregado
    red_impact = len(red_team) * 1.1
    
    winrate_blue = clamp(base_winrate + blue_impact - red_impact, 10, 90)
    winrate_red = 100 - winrate_blue
    
    # Simulación del valor individual que aporta el último campeón seleccionado
    valor_agregado = round(np.random.uniform(-3.5, 4.5), 2) 
    
    return round(winrate_blue, 1), round(winrate_red, 1), valor_agregado

def clamp(n, minn, maxn):
    return max(min(n, maxn), minn)

# 6. INTERFAZ GRÁFICA DEL DRAFT
col_blue, col_status, col_red = st.columns([1, 1.5, 1])

with col_blue:
    st.header("🔵 Equipo Azul")
    for i, champ in enumerate(st.session_state.blue_team):
        rol = df_champs[df_champs['campeon'] == champ]['rol'].values[0]
        st.subheader(f"{i+1}. {champ} ({rol})")

with col_red:
    st.header("🔴 Equipo Rojo")
    for i, champ in enumerate(st.session_state.red_team):
        rol = df_champs[df_champs['campeon'] == champ]['rol'].values[0]
        st.subheader(f"{i+1}. {champ} ({rol})")

# Control central del flujo
with col_status:
    st.header("📊 Estado del Draft")
    
    win_b, win_r, val_agregado = calcular_metricas(st.session_state.blue_team, st.session_state.red_team, modelo_activo)
    
    # Mostrar Porcentajes de Victoria en tiempo real
    st.metric(label="Winrate Equipo Azul", value=f"{win_b}%")
    st.metric(label="Winrate Equipo Rojo", value=f"{win_r}%")
    
    if st.session_state.current_step < 10:
        current_pick_info = DRAFT_ORDER[st.session_state.current_step]
        st.info(f"Turno actual: **{current_pick_info['label']}**")
        
        # Filtrar campeones ya seleccionados
        disponibles = df_champs[~df_champs['campeon'].isin(st.session_state.picked_champions)]['campeon'].tolist()
        
        # Input de selección
        seleccion = st.selectbox("Selecciona un Campeón:", ["-- Selecciona --"] + disponibles)
        
        if st.button("Confirmar Selección"):
            if seleccion != "-- Selecciona --":
                equipo_actual = current_pick_info['equipo']
                team_list = st.session_state.blue_team if equipo_actual == "Blue" else st.session_state.red_team
                
                # Validar restricción de máximo 2 del mismo rol
                if check_role_limit(team_list, seleccion):
                    # Guardar selección
                    if equipo_actual == "Blue":
                        st.session_state.blue_team.append(seleccion)
                    else:
                        st.session_state.red_team.append(seleccion)
                    
                    st.session_state.picked_champions.add(seleccion)
                    st.session_state.current_step += 1
                    
                    # Mostrar el impacto del campeón recién pickeado
                    st.success(f"{seleccion} agregará un {val_agregado}% de probabilidad de victoria al equipo {equipo_actual}")
                    st.button("Continuar al siguiente turno") # Forzar recarga limpia
                else:
                    st.error(f"❌ No puedes elegir a {seleccion}. El equipo {equipo_actual} ya tiene el límite máximo (2) de ese rol.")
    else:
        st.success("🎉 ¡Draft Completado con éxito!")
