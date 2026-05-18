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
@st.cache_data
def load_champ_data():
    # Asegúrate de que el nombre del archivo coincida con tu archivo real
    df = pd.read_csv("datos_campeones.csv")
    return df

df_champs = load_champ_data()

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
    # Obtenemos el main_role del campeón propuesto
    proposed_role = df_champs[df_champs['name'] == proposed_champ]['main_role'].values[0]
    
    # Filtramos los roles de los campeones que ya están en el equipo
    roles_in_team = df_champs[df_champs['name'].isin(team_champs)]['main_role'].tolist()
    
    # Validamos que no haya ya 2 con ese mismo rol
    return roles_in_team.count(proposed_role) < 2

# 5. SIMULACIÓN DE PROBABILIDADES (Aquí conectas tu modelo)
def calcular_metricas(blue_team, red_team, modelo):
    # 1. Si el draft está vacío, empezamos en 50%-50% y 0 de impacto
    if not blue_team and not red_team:
        return 50.0, 50.0, 0.0
    
    # 2. CONSTRUIR EL VECTOR DE ENTRADA PARA EL MODELO
    # Extraemos las filas de los campeones seleccionados de nuestra base de datos
    df_blue = df_champs[df_champs['name'].isin(blue_team)]
    df_red = df_champs[df_champs['name'].isin(red_team)]
    
    # EJEMPLO A: Si tu modelo fue entrenado con la SUMA o PROMEDIO de las estadísticas de cada equipo:
    # Columnas numéricas a promediar/sumar (Gold_Phys_Dmg, Gold_Mag_Dmg, etc.)
    features_numericas = [col for col in df_champs.columns if col.startswith('Gold_')]
    
    # Calculamos el vector del equipo azul y rojo (llenamos con 0 si están vacíos al inicio del draft)
    stats_blue = df_blue[features_numericas].sum().to_dict() if not df_blue.empty else {c: 0 for c in features_numericas}
    stats_red = df_red[features_numericas].sum().to_dict() if not df_red.empty else {c: 0 for c in features_numericas}
    
    # Creamos el diccionario final combinando ambos bandos con el prefijo correspondiente
    input_data = {}
    for col in features_numericas:
        input_data[f"Blue_{col}"] = stats_blue[col]
        input_data[f"Red_{col}"] = stats_red[col]
        
    # Convertimos a DataFrame de una sola fila (lo que espera sklearn)
    X_input = pd.DataFrame([input_data])
    
    # Asegúrate de que X_input tenga el mismo orden de columnas con el que entrenaste el modelo.
    # Si usaste columnas categóricas (como main_role con One-Hot Encoding), deberás agregarlas aquí también.
    
    # 3. PREDICCIÓN DEL WINRATE
    try:
        # predict_proba devuelve [[prob_perder, prob_ganar]] del equipo azul
        probabilidades = modelo.predict_proba(X_input)[0]
        winrate_blue = round(probabilidades[1] * 100, 1) # Probabilidad de victoria de Blue
        winrate_red = round(100 - winrate_blue, 1)
    except Exception as e:
        # En caso de que falten columnas o el modelo falle mientras el draft esté incompleto
        winrate_blue, winrate_red = 50.0, 50.0

    # 4. CALCULAR EL VALOR AGREGADO DEL ÚLTIMO PICK
    # Evaluamos cuánto cambió el winrate con respecto al turno anterior
    if "last_winrate" not in st.session_state:
        st.session_state.last_winrate = 50.0
        
    # El valor agregado es la diferencia del winrate actual con el del paso anterior
    # Si el último turno fue del equipo Azul, nos interesa cuánto subió para Blue. Si fue Red, cuánto subió para Red.
    valor_agregado = round(winrate_blue - st.session_state.last_winrate, 2)
    st.session_state.last_winrate = winrate_blue
    
    return winrate_blue, winrate_red, valor_agregado

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
        
        # Filtrar campeones ya seleccionados usando la columna 'name'
        disponibles = df_champs[~df_champs['name'].isin(st.session_state.picked_champions)]['name'].tolist()
        
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
