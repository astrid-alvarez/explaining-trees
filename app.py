# =============================================================================
# APLICACIÓN WEB PARA EVALUACIÓN CUALITATIVA (OE3)
# Autor: Astrid Yinnet Álvarez Castro
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import graphviz
import os
import math
import matplotlib.pyplot as plt
from PIL import Image
import re
import uuid
import streamlit.components.v1 as components
import streamlit as st
from collections import Counter
#import base64


# -----------------------------------------------------------------------------
# FUNCIÓN AUXILIAR: MOSTRAR DOT COMO PNG (para ver el árbol completo)
# -----------------------------------------------------------------------------
def mostrar_dot_en_streamlit(dot, vh=85, height_px=900):
    """
    Renderiza el árbol SIN scroll y ajustado al viewport.
    Usa SVG inline y recalcula viewBox con el bbox real.
    """
    try:
        svg_bytes = dot.pipe(format="svg")
        svg = svg_bytes.decode("utf-8", errors="ignore")

        # Quitar width/height fijos para que CSS mande
        svg = re.sub(r'\swidth="[^"]*"', "", svg, count=1)
        svg = re.sub(r'\sheight="[^"]*"', "", svg, count=1)

        wrap_id = f"wrap_{uuid.uuid4().hex}"

        html = f"""
        <div id="{wrap_id}" style="width:100%; min-height:420px; height:auto; margin:0; padding:0; overflow:hidden;">
          {svg}
        </div>
        
        <script>
        (function() {{
          const wrap = document.getElementById("{wrap_id}");
          if (!wrap) return;
          const svg = wrap.querySelector("svg");
          if (!svg) return;
        
          svg.style.width = "100%";
          svg.style.height = "100%";
          svg.style.display = "block";
          svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
        
          function fit() {{
            try {{
              const bbox = svg.getBBox();
              if (bbox && bbox.width > 0 && bbox.height > 0) {{
                svg.setAttribute("viewBox", `${{bbox.x}} ${{bbox.y}} ${{bbox.width}} ${{bbox.height}}`);
        
                const wrapWidth = wrap.clientWidth || 1000;
                const maxH = Math.floor(window.innerHeight * 0.85);
                const h = Math.max(420, Math.min(maxH, (bbox.height / bbox.width) * wrapWidth));
                wrap.style.height = h + "px";
              }}
            }} catch(e) {{}}
          }}
        
          requestAnimationFrame(() => {{
            fit();
            setTimeout(fit, 50);
            setTimeout(fit, 250);
          }});
        
          try {{
            const ro = new ResizeObserver(() => fit());
            ro.observe(wrap);
          }} catch(e) {{}}
        }})();
        </script>
        """
        components.html(html, height=950, scrolling=False)


    except Exception as e:
        st.error(f"Error al renderizar el árbol en SVG: {e}")




# -----------------------------------------------------------------------------
# CONFIGURACIÓN Y ESTILOS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Evaluación XAI", layout="wide")

PALETTE = [
    "#FF8C00", "#32CD32", "#8A2BE2", "#00BFFF",
    "#FFD700", "#DA70D6", "#40E0D0", "#FFB6C1",
    "#B0E2FF", "#7FFFD4", "#FF69B4", "#98FB98",
    "#DDA0DD", "#87CEEB", "#F0E68C", "#FFA07A"
]

INFO_BDS = {
    "BD1_Educacion": {
        "desc": "Datos de estudiantes universitarios para analizar su trayectoria académica y permanencia.",
        "target": "Estado del estudiante (Dropout, Graduate, Enrolled)."
    },
    "BD2_Diabetes": {
        "desc": "Registros clínicos de hospitales sobre hospitalización de pacientes diabéticos.",
        "target": "Tiempo de readmisión (<30, >30, No)."
    },
    "BD3_Forestal": {
        "desc": "Datos espectrales para clasificar tipos de cobertura forestal.",
        "target": "Tipo de cubierta forestal."
    },
    "BD4_EduDane": {
        "desc": "Apropiación tecnológica en establecimientos educativos del Cauca.",
        "target": "Frecuencia de uso de bienes TIC."
    },
    "BD5_Heart": {
        "desc": "Datos clínicos cardíacos.",
        "target": "Nivel de severidad de enfermedad."
    },
    "BD6_Cancer": {
        "desc": "Características celulares sobre cáncer de mama.",
        "target": "Diagnóstico (Benigno vs Maligno)."
    },
    "BD7_Iris": {
        "desc": "Medidas de flores Iris.",
        "target": "Especie de la flor."
    }
}

# -----------------------------------------------------------------------------
# Información contextual adicional (Sección 1)
# - Tipos de atributos
# - Significado de clases de la variable objetivo
# -----------------------------------------------------------------------------
INFO_EXTRA = {

    "BD1_Educacion": {
        "tipos_atributos": [
            "Académicas",
            "Socioeconómicas",
            "Demográficas",
        ],
        "clases_significado": {
            "Dropout": "Abandona el programa antes de completarlo.",
            "Graduate": "Completa el programa y se gradúa.",
            "Enrolled": "Continúa matriculado (sin abandonar ni graduarse).",
        },
    },

    "BD2_Diabetes": {
        "tipos_atributos": [
            "Clínicas",
            "Administrativas",
            "Demográficas",
        ],
        "clases_significado": {
            "<30": "Readmisión hospitalaria en menos de 30 días.",
            ">30": "Readmisión hospitalaria en más de 30 días.",
            "No": "Sin readmisión hospitalaria registrada.",
        },
    },

    "BD3_Forestal": {
        "tipos_atributos": [
            "Espectrales",
        ],
        "clases_significado": {
            "s": "Tipo de cubierta forestal clase s.",
            "d": "Tipo de cubierta forestal clase d.",
            "h": "Tipo de cubierta forestal clase h.",
            "o": "Tipo de cubierta forestal clase o.",
        },
    },

    "BD4_EduDane": {
        "tipos_atributos": [
            "Institucionales",
            "Demográficas",
            "Infraestructura",
            "Inventario",
            "Gestión y prácticas",
        ],
        "clases_significado": {
            "Ningún día de la semana": "No se usan bienes TIC durante la semana.",
            "Una vez al mes pero no todos los meses del año": "Uso esporádico durante el año.",
            "Al menos una vez al mes": "Uso mensual esporádico de bienes TIC.",
            "Al menos una vez a la semana": "Uso semanal esporádico de bienes TIC.",
            "Todos los días de la semana": "Uso diario de bienes TIC.",
        },
    },

    "BD5_Heart": {
        "tipos_atributos": [
            "Demográficas",
            "Clínicas",
            "Pruebas diagnósticas",
            "Ejercicio y esfuerzo físico.",
        ],
        "clases_significado": {
            0: "Nivel 0: menor severidad de la enfermedad.",
            1: "Nivel 1 de severidad de la enfermedad.",
            2: "Nivel 2 de severidad de la enfermedad.",
            3: "Nivel 3 de severidad de la enfermedad.",
            4: "Nivel 4: mayor severidad de la enfermedad.",
        },
    },

    "BD6_Cancer": {
        "tipos_atributos": [
            "Citológicas y morfológicas.",
        ],
        "clases_significado": {
            "Benigno": "Tumor no cancerígeno.",
            "Maligno": "Tumor cancerígeno.",
        },
    },

    "BD7_Iris": {
        "tipos_atributos": [
            "Morfológicas.",
        ],
        "clases_significado": {
            "setosa": "Especie Iris setosa.",
            "versicolor": "Especie Iris versicolor.",
            "virginica": "Especie Iris virginica.",
        },
    },

}


st.title("PLATAFORMA DE EVALUACIÓN XAI: ÁRBOLES ESPECIALISTAS")
st.markdown(
    "<p style='font-size:0.95rem'><b>Objetivo:</b> "
    "Evaluar la explicabilidad de las explicaciones por clase.</p>",
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# CARGA DE DATOS
# -----------------------------------------------------------------------------
@st.cache_resource
def cargar_archivos_pkl():
    archivos = [f for f in os.listdir('.') if f.endswith('_app_data.pkl')]
    diccionario_datos = {}
    for nombre_archivo in archivos:
        try:
            with open(nombre_archivo, 'rb') as f:
                datos = pickle.load(f)
                diccionario_datos[datos['nombre_bd']] = datos
        except Exception as e:
            st.error(f"Error leyendo {nombre_archivo}: {e}")
    return diccionario_datos


bds_disponibles = cargar_archivos_pkl()

if not bds_disponibles:
    st.error("No se encontraron archivos .pkl.")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR: 1. SELECCIÓN DE CASO + DISTRIBUCIÓN
# -----------------------------------------------------------------------------
st.sidebar.header("1. Selección de Caso")

lista_bds = sorted(list(bds_disponibles.keys()))
nombre_bd = st.sidebar.selectbox("Base de Datos", lista_bds)

paquete = bds_disponibles[nombre_bd]
modelo = paquete['modelo']
feat_names = paquete['feature_names']
raw_classes = paquete['class_names']
X_test = paquete['X_test']
y_test = paquete.get('y_test', None)
mapa = paquete['mapa_nombres']

class_names = list(raw_classes)
CLASS_COLORS = {cls: PALETTE[i % len(PALETTE)] for i, cls in enumerate(class_names)}

#total_registros = len(X_test)
# NUEVO: Calculamos el número de columnas (features)
num_features = X_test.shape[1] if hasattr(X_test, 'shape') else 0 

num_clases = len(raw_classes)
info_texto = INFO_BDS.get(nombre_bd, {"desc": "Sin descripción", "target": "N/A"})

# -----------------------------------------------------------------------------
# CORRECCIÓN DE VARIABLES
# -----------------------------------------------------------------------------
# 1. Recuperar tamaño de entrenamiento desde el árbol (dato oculto en sklearn)
n_train = int(modelo.tree_.n_node_samples[0])

# 2. Definir la variable vital para el resto del código
total_registros = len(X_test)  
# 3. Calcular el total real SOLO para la visualización en la tarjeta
total_registros_reales = n_train + total_registros

# 4. Contar variables (columnas)
total_variables = X_test.shape[1] if hasattr(X_test, 'shape') else 0

extra = INFO_EXTRA.get(nombre_bd, None)
tipos = extra.get("tipos_atributos", []) if extra else []
nombres_tipos = [t.split(":")[0].strip() for t in tipos]  # solo categorías

tipos_texto = " · ".join(nombres_tipos) if nombres_tipos else "—"

# 5. Generar la tarjeta con los datos solicitados (con categorías de variables)
st.sidebar.markdown(
    f"""
    <div style="background-color:#FFFFFF;
                padding:10px 14px;
                border-radius:8px;
                color:black;
                font-size:0.8rem;
                border: 1px solid #e0e0e0;">
        <b>Descripción:</b> {info_texto['desc']}<br/>
        <b>Registros:</b> {total_registros_reales}<br/>
        <b>Variables:</b> {total_variables}<br/>
        <b>Categorías de las variables:</b> {tipos_texto}<br/>
        <b>Variable objetivo:</b> {info_texto['target']}
    </div>
    """,
    unsafe_allow_html=True
)



# --- Distribución de clases ---
st.sidebar.subheader("Distribución de la Variable Objetivo")

# -----------------------------------------------------------------------------
# NUEVO: Significado de clases (ubicado junto a la distribución)
# -----------------------------------------------------------------------------
extra = INFO_EXTRA.get(nombre_bd, None)

if extra is not None:
    clases_sig = extra.get("clases_significado", {})

    if len(clases_sig) > 0:
        with st.sidebar.expander("¿Qué representa cada clase?", expanded=False):
            st.markdown(
                """
                <div style="font-size:0.85rem; line-height:1.4;">
                """ +
                "".join(
                    f"<b>{cls}:</b> {clases_sig[cls]}<br/>"
                    for cls in class_names if cls in clases_sig
                ) +
                "".join(
                    f"<b>{cls}:</b> {desc}<br/>"
                    for cls, desc in clases_sig.items() if cls not in class_names
                ) +
                """
                </div>
                """,
                unsafe_allow_html=True
            )


    # ================================
    # 1) Conteos globales (train + test)
    # ================================
    def get_conteos_globales(modelo, y_test, n_clases):
        """
        Usa:
          - value[0] del árbol (conteos en entrenamiento),
          - y_test (conteos en prueba),
        para aproximar la distribución global por clase.
        """
        # Conteos en entrenamiento desde el nodo raíz
        root_counts = np.asarray(modelo.tree_.value[0], dtype=float).ravel()
        root_counts = root_counts.astype(int)

        # Ajustar tamaño por seguridad
        if root_counts.shape[0] < n_clases:
            root_counts = np.pad(root_counts, (0, n_clases - root_counts.shape[0]))
        elif root_counts.shape[0] > n_clases:
            root_counts = root_counts[:n_clases]

        # Conteos en test
        if isinstance(y_test, pd.Series):
            y_arr = y_test.values
        else:
            y_arr = np.array(y_test)

        test_counts = np.zeros(n_clases, dtype=int)
        for i in range(n_clases):
            test_counts[i] = np.sum(y_arr == i)

        # Total = train + test
        return root_counts + test_counts

    n_clases = len(class_names)
    conteos_totales = get_conteos_globales(modelo, y_test, n_clases)

    # Total de la BD usado SOLO para normalizar a porcentaje
    total_bd = float(conteos_totales.sum()) if conteos_totales.sum() > 0 else 1.0

    # ================================
    # 2) Conversión a porcentajes
    # ================================
    porcentajes = (conteos_totales / total_bd) * 100.0

    # ================================
    # 3) Gráfico en porcentaje
    # ================================
    def plot_distribucion_porcentual(pcts, nombres_clases):
        etiquetas = nombres_clases
        colores = [PALETTE[i % len(PALETTE)] for i in range(len(etiquetas))]

        fig, ax = plt.subplots(figsize=(4, 4.5))
        bars = ax.bar(range(len(etiquetas)), pcts, color=colores, edgecolor="black")

        ax.set_xticks(range(len(etiquetas)))
        ax.set_xticklabels(etiquetas, rotation=45, ha='right', fontsize=10)

        ax.set_ylabel("Porcentaje de registros (%)", fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        max_y = max(pcts) if len(pcts) > 0 else 1.0
        for bar, pct in zip(bars, pcts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + (0.02 * max_y),
                f"{pct:.1f}%",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

        # Opcional: limitar el eje superior
        ax.set_ylim(0, min(100, max_y * 1.15))

        plt.tight_layout()
        return fig

    st.sidebar.pyplot(plot_distribucion_porcentual(porcentajes, class_names))
   

# -----------------------------------------------------------------------------
# SIDEBAR: 2. SELECCIÓN DE CLASE + DIAGNÓSTICO DE REGLAS
# -----------------------------------------------------------------------------
# Nombres "bonitos" de clase
nombres_bonitos = []
mapa_idx = {}
for i, val in enumerate(raw_classes):
    nombre = str(val)
    if mapa:
        if val in mapa:
            nombre = mapa[val]
        elif i in mapa:
            nombre = mapa[i]
    nombres_bonitos.append(nombre)
    mapa_idx[nombre] = i

st.sidebar.divider()
st.sidebar.header("2. Selección de Clase")

clase_elegida = st.sidebar.selectbox("Clase a Explicar", nombres_bonitos)
idx_objetivo = mapa_idx[clase_elegida]
color_clase_hex = PALETTE[idx_objetivo % len(PALETTE)]

# --- Diagnóstico de reglas (máximos teóricos para la BD) ---
def diagnostico_reglas_bd(modelo, class_idx, total_muestras, tau, soporte_min_abs):
    """
    Diagnóstico en hojas que predicen class_idx.

    Devuelve:
      - reglas_pot: # hojas que predicen la clase (sin filtros)
      - conf_max: máxima p(clase) (sin filtro de soporte)
      - sup_en_conf_max: soporte de la hoja donde se logra conf_max
      - soporte_max: máximo soporte (sin filtro de confianza)
      - conf_en_sup_max: p(clase) en la hoja con soporte_max
      - conf_max_con_soporte: máxima p(clase) entre hojas con soporte >= soporte_min_abs
      - soporte_max_con_tau: máximo soporte entre hojas con p(clase) >= tau
    """
    tree_ = modelo.tree_
    reglas_pot = 0

    conf_max = -1.0
    sup_en_conf_max = 0

    soporte_max = 0
    conf_en_sup_max = 0.0

    conf_max_con_soporte = -1.0
    soporte_max_con_tau = 0

    for u in range(tree_.node_count):
        if tree_.children_left[u] != -1:
            continue  # no hoja

        v = np.asarray(tree_.value[u], dtype=float).reshape(-1)
        pred = int(np.argmax(v))
        if pred != class_idx:
            continue

        reglas_pot += 1
        sup = int(tree_.n_node_samples[u])
        s = float(v.sum())
        p = float(v[class_idx] / s) if s > 0 else 0.0

        # Máxima confianza (sin filtro de soporte)
        if p > conf_max:
            conf_max = p
            sup_en_conf_max = sup

        # Máximo soporte (sin filtro de confianza)
        if sup > soporte_max:
            soporte_max = sup
            conf_en_sup_max = p

        # Máxima confianza con soporte mínimo
        if sup >= soporte_min_abs:
            conf_max_con_soporte = max(conf_max_con_soporte, p)

        # Máximo soporte con tau
        if p >= tau:
            soporte_max_con_tau = max(soporte_max_con_tau, sup)

    if conf_max < 0:
        conf_max = 0.0
    if conf_max_con_soporte < 0:
        conf_max_con_soporte = 0.0

    return (reglas_pot, conf_max, sup_en_conf_max,
            soporte_max, conf_en_sup_max,
            conf_max_con_soporte, soporte_max_con_tau)



tau_diag = float(st.session_state.get("confianza_pct", 90)) / 100.0
soporte_pct_diag = float(st.session_state.get("soporte_pct", 1.5))
soporte_min_abs = max(1, int(total_registros * (soporte_pct_diag / 100.0)))


(
    reglas_pot,
    conf_max,
    sup_confmax,
    sup_max,
    conf_supmax,
    conf_max_con_soporte,
    sup_max_con_tau
) = diagnostico_reglas_bd(
    modelo,
    idx_objetivo,
    total_registros,
    tau_diag,
    soporte_min_abs
)


sup_max_pct = (sup_max / total_registros * 100.0) if total_registros > 0 else 0.0

def _leaf_stats_for_class(modelo, class_idx):
    """
    Devuelve lista de (p_clase_en_hoja, soporte_en_hoja) SOLO para hojas cuya clase predicha = class_idx.
    """
    tree_ = modelo.tree_
    stats = []

    for u in range(tree_.node_count):
        if tree_.children_left[u] == -1:  # hoja
            v = np.asarray(tree_.value[u], dtype=float).reshape(-1)
            pred = int(np.argmax(v))
            if pred != class_idx:
                continue

            s = float(v.sum())
            p = float(v[class_idx] / s) if s > 0 else 0.0
            sup = int(tree_.n_node_samples[u])
            stats.append((p, sup))

    return stats


def sugerir_filtros_iniciales(modelo, class_idx, total_registros, tau_pref=0.90, soporte_pct_pref=1.5):
    """
    Intenta mantener tau y soporte preferidos.
    Si no existen reglas, ajusta automáticamente (primero soporte, luego tau) para garantizar al menos 1 regla.
    Retorna (confianza_pct_sugerida, soporte_pct_sugerido, motivo).
    """
    stats = _leaf_stats_for_class(modelo, class_idx)

    if not stats:
        # No hay hojas que predigan esa clase (raro, pero posible)
        return 50, 0.1, "Sin hojas que predigan esta clase; se usan valores mínimos para explorar."

    tau0 = float(tau_pref)
    sup_abs0 = max(1, int(total_registros * (soporte_pct_pref / 100.0)))

    # Helpers
    def existe_regla(tau, sup_abs):
        return any((p >= tau and sup >= sup_abs) for (p, sup) in stats)

    # 1) Si con lo preferido ya hay reglas, perfecto
    if existe_regla(tau0, sup_abs0):
        return int(round(tau0 * 100)), float(soporte_pct_pref), "Valores preferidos: existen reglas."

    # 2) Mantener tau0 y bajar soporte al máximo disponible bajo tau0 (si existe)
    soportes_con_tau0 = [sup for (p, sup) in stats if p >= tau0]
    if soportes_con_tau0:
        sup_abs_sug = max(1, max(soportes_con_tau0))  # el mayor soporte que cumple tau0
        soporte_pct_sug = (sup_abs_sug / total_registros) * 100.0
        return int(round(tau0 * 100)), float(soporte_pct_sug), "Se mantuvo confianza; se ajustó soporte al máximo posible para esa confianza."

    # 3) Si ni siquiera hay hojas con p>=tau0, bajar tau gradualmente hasta encontrar algo
    # probamos una rejilla razonable
    taus = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
    for tau in taus:
        soportes = [sup for (p, sup) in stats if p >= tau]
        if soportes:
            # elegimos soporte para que NO sea ultra pequeño: tomamos el percentil 75 de soportes disponibles
            soportes_sorted = sorted(soportes)
            k = int(0.75 * (len(soportes_sorted) - 1)) if len(soportes_sorted) > 1 else 0
            sup_abs_sug = max(1, soportes_sorted[k])
            soporte_pct_sug = (sup_abs_sug / total_registros) * 100.0
            return int(round(tau * 100)), float(soporte_pct_sug), "Se ajustó confianza y soporte para garantizar reglas."

    # 4) Último recurso
    return 50, 0.1, "No se encontró combinación razonable; se usan mínimos para explorar."


# -------------------------------------------------------------------------
# 2) Diagnóstico de reglas (UI simple + advertencia automática)
# -------------------------------------------------------------------------
def diagnostico_reglas_bd(modelo, class_idx):
    """
    Calcula:
      - número de hojas que predicen la clase objetivo (reglas potenciales),
      - confianza máxima alcanzable en una hoja de esa clase,
      - soporte máximo (absoluto y porcentual).
    Nota: el porcentaje de soporte se calcula sobre el TOTAL usado por el árbol (nodo raíz),
    para evitar valores > 100%.
    """
    tree_ = modelo.tree_
    total_base = int(tree_.n_node_samples[0])  # total de muestras que vio el árbol (entrenamiento)

    reglas = 0
    conf_max = 0.0
    soporte_max = 0

    for u in range(tree_.node_count):
        if tree_.children_left[u] == -1:  # hoja
            v = np.asarray(tree_.value[u], dtype=float).reshape(-1)
            pred = int(np.argmax(v))
            if pred == class_idx:
                reglas += 1
                sup = int(tree_.n_node_samples[u])
                soporte_max = max(soporte_max, sup)

                s = v.sum()
                p = (v[class_idx] / s) if s > 0 else 0.0
                conf_max = max(conf_max, float(p))

    soporte_pct = (soporte_max / total_base * 100.0) if total_base > 0 else 0.0
    return reglas, conf_max, soporte_max, soporte_pct

def diagnostico_con_filtros(modelo, class_idx, tau, soporte_abs):
    """
    Verifica si existen hojas (reglas) que cumplan SIMULTÁNEAMENTE:
      - predicen la clase
      - p(clase) >= tau
      - soporte >= soporte_abs
    Devuelve:
      - n_ok: cuántas reglas cumplen
      - conf_max_ok: mayor p(clase) entre las que cumplen
      - soporte_max_ok: mayor soporte entre las que cumplen
    """
    tree_ = modelo.tree_
    n_ok = 0
    conf_max_ok = 0.0
    soporte_max_ok = 0

    for u in range(tree_.node_count):
        if tree_.children_left[u] == -1:
            v = np.asarray(tree_.value[u], dtype=float).reshape(-1)
            pred = int(np.argmax(v))
            if pred != class_idx:
                continue

            sup = int(tree_.n_node_samples[u])
            s = v.sum()
            p = (v[class_idx] / s) if s > 0 else 0.0

            if (p >= tau) and (sup >= soporte_abs):
                n_ok += 1
                conf_max_ok = max(conf_max_ok, float(p))
                soporte_max_ok = max(soporte_max_ok, sup)

    return n_ok, conf_max_ok, soporte_max_ok


# --- Diagnóstico base (sin filtros) ---
reglas_pot, conf_max, sup_max, sup_max_pct = diagnostico_reglas_bd(
    modelo, idx_objetivo
)


st.sidebar.subheader("Diagnóstico de Reglas")
st.sidebar.markdown(
    f"""
    <div style="background-color:#FFFFFF;
                padding:10px 14px;
                border-radius:8px;
                color:black;
                font-size:0.9rem;
                border: 1px solid #e0e0e0;">
        <b>Reglas Potenciales:</b> {reglas_pot}<br/>
        ▪ <b>Confianza Máx:</b> {conf_max*100:.1f}%<br/>
        ▪ <b>Soporte Máx:</b> {sup_max} ({sup_max_pct:.1f}%)
    </div>
    """,
    unsafe_allow_html=True
)

# --- (A) Advertencia automática por bajo soporte (riesgo de sobreajuste) ---
# Umbral simple: si la mejor regla cubre muy poco del total, avisar.
umbral_pct_bajo = 1.0  # 1% del total (se puede cambiar a 0.5 o 2.0 )
if sup_max_pct > 0 and sup_max_pct < umbral_pct_bajo:
    st.sidebar.warning(
        f"Esta clase tiene reglas muy específicas: el soporte máximo es {sup_max_pct:.1f}% "
        f"del total. Podría haber mayor riesgo de sobreajuste."
    )

# --- Expander de interpretación (mismo tamaño de letra y espaciado consistente) ---
st.sidebar.markdown("")  # espacio pequeño
with st.sidebar.expander("¿Cómo interpretar este diagnóstico?", expanded=False):
    st.markdown(
        """
        <div style="font-size:0.80rem; line-height:1.25;">
          <p style="margin:0 0 0.65rem 0;">
            <b>Reglas potenciales:</b> número de hojas del árbol que predicen esta clase.
            Un valor alto sugiere más “caminos” posibles para explicarla.
          </p>
          <p style="margin:0 0 0.65rem 0;">
            <b>Confianza máx:</b> qué tan “segura” puede ser una regla para esta clase.
            Muy alta puede indicar una regla muy específica.
          </p>
          <p style="margin:0;">
            <b>Soporte máx:</b> cuántos registros alcanza la regla más representativa.
            Soportes bajos pueden implicar mayor riesgo de sobreajuste.
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# FUNCIONES AUXILIARES (BASADAS EN PARTE 13)  -- LÓGICA DE LOS ÁRBOLES
# -----------------------------------------------------------------------------
def _arr_str_int(a):
    a = list(map(int, a))
    return "[" + ", ".join(str(x) for x in a) + "]"


def _is_leaf(tree_, u: int) -> bool:
    return tree_.children_left[u] == -1


def _predicted_class_idx(tree_, u: int) -> int:
    v = np.asarray(tree_.value[u], dtype=float)
    v = v.reshape(-1)
    return int(np.argmax(v))


def _node_prob_for_class(tree_, u: int, class_idx: int) -> float:
    v = np.asarray(tree_.value[u], dtype=float)
    v = v.reshape(-1)
    s = v.sum()
    if s <= 0:
        return 0.0
    return float(v[class_idx] / s)


def _node_counts_and_probvec(tree_, u: int):
    v = np.asarray(tree_.value[u], dtype=float)
    w = v.reshape(-1)
    s = w.sum()
    if s > 0:
        probs = w / s
    else:
        probs = np.zeros_like(w)
    samples = int(tree_.n_node_samples[u])
    return w, probs, samples


def _keep_mask_strict_monotonic(tree_, class_idx: int, tau: float,
                                min_samples_to_keep: int = 0):
    """
    MODO ESTRICTO:
    - p_hijo >= p_padre en cada paso.
    - La hoja predice la clase objetivo, p >= tau, soporte >= min_samples_to_keep.
    """
    n = tree_.node_count
    keep = np.zeros(n, dtype=bool)

    def dfs(u: int, parent_p: float = 0.0) -> bool:
        samples_u = int(tree_.n_node_samples[u])
        p_current = _node_prob_for_class(tree_, u, class_idx)

        if p_current < parent_p:
            return False

        if _is_leaf(tree_, u):
            ok = (
                _predicted_class_idx(tree_, u) == class_idx and
                p_current >= tau and
                samples_u >= min_samples_to_keep
            )
            if ok:
                keep[u] = True
            return ok

        L, R = int(tree_.children_left[u]), int(tree_.children_right[u])
        left_ok = dfs(L, p_current)
        right_ok = dfs(R, p_current)

        if left_ok or right_ok:
            keep[u] = True
            return True

        return False

    dfs(0, 0.0)
    return keep


def _keep_mask_non_strict(tree_, class_idx: int, tau: float,
                          min_samples_to_keep: int = 0):
    """
    MODO NO ESTRICTO:
      - SIN restricción de monotonía.
      - Marca nodos que pertenecen a al menos un camino raíz→hoja donde la hoja:
          * predice la clase objetivo,
          * tiene p >= tau,
          * tiene soporte >= min_samples_to_keep.
    """
    n = tree_.node_count
    keep = np.zeros(n, dtype=bool)

    def dfs(u: int, path):
        if _is_leaf(tree_, u):
            samples_u = int(tree_.n_node_samples[u])
            p_current = _node_prob_for_class(tree_, u, class_idx)
            ok = (
                _predicted_class_idx(tree_, u) == class_idx and
                p_current >= tau and
                samples_u >= min_samples_to_keep
            )
            if ok:
                for node_id in path + [u]:
                    keep[node_id] = True
            return ok

        L, R = int(tree_.children_left[u]), int(tree_.children_right[u])

        path.append(u)
        left_ok = dfs(L, path)
        right_ok = dfs(R, path)
        path.pop()

        if left_ok or right_ok:
            keep[u] = True
            return True

        return False

    dfs(0, [])
    return keep


def _get_paths_for_class(tree_, keep):
    paths = {}

    def dfs(u, path):
        if not keep[u]:
            return

        if _is_leaf(tree_, u):
            paths[u] = list(path)
            return

        feat_idx = int(tree_.feature[u])
        thr = float(tree_.threshold[u])
        L = int(tree_.children_left[u])
        R = int(tree_.children_right[u])

        if keep[L]:
            dfs(L, path + [(feat_idx, thr, "<=", u)])
        if keep[R]:
            dfs(R, path + [(feat_idx, thr, ">", u)])

    dfs(0, [])
    return paths

from collections import Counter

def top_variables_influyentes(tree_, keep_mask, feature_names, topk=5):
    """
    Calcula las variables más influyentes DEL ÁRBOL ESPECIALISTA ACTUAL.
    Cuenta cuántas veces aparece cada variable en nodos keep (no hojas).
    """
    counts = Counter()

    for u in range(tree_.node_count):
        if not keep_mask[u]:
            continue
        if tree_.children_left[u] == -1:
            continue  # ignorar hojas
        feat_idx = int(tree_.feature[u])
        if 0 <= feat_idx < len(feature_names):
            counts[feature_names[feat_idx]] += 1

    return [name for name, _ in counts.most_common(topk)]



def _compact_path_to_intervals(path, feature_names):
    bounds = {}
    pos = 0

    for feat_idx, thr, op, node_id in path:
        if feat_idx < 0:
            pos += 1
            continue

        feat_name = feature_names[feat_idx]

        if feat_name not in bounds:
            bounds[feat_name] = {
                "lower": -math.inf,
                "upper": math.inf,
                "node_idx": node_id,
                "first_pos": pos
            }

        b = bounds[feat_name]

        if op == "<=":
            b["upper"] = min(b["upper"], thr)
        else:
            b["lower"] = max(b["lower"], thr)

        pos += 1

    conds = []
    for feat_name in sorted(bounds.keys(), key=lambda f: bounds[f]["first_pos"]):
        b = bounds[feat_name]
        lo, hi, node_idx = b["lower"], b["upper"], b["node_idx"]

        if lo == -math.inf and hi == math.inf:
            continue
        elif lo == -math.inf:
            text = f"{feat_name} <= {hi:.3f}"
        elif hi == math.inf:
            text = f"{feat_name} > {lo:.3f}"
        else:
            text = f"{lo:.3f} < {feat_name} <= {hi:.3f}"

        conds.append({"text": text, "node_idx": node_idx})

    return conds


def build_compacted_graphviz_non_strict(modelo, clase, tau, soporte,
                                        keep_mask, feature_names,
                                        class_names_list, color_clase):
    """
    MODO NO ESTRICTO:
      - Usa _keep_mask_non_strict (sin monotonía).
      - Compacta desigualdades por feature.
      - Árbol virtual compartido, SIN poda p_hijo >= p_padre.
    """
    tree_ = modelo.tree_
    cidx = class_names_list.index(clase)

    paths = _get_paths_for_class(tree_, keep_mask)
    if not paths:
        return None

    compacted_paths = {}
    for leaf, path in paths.items():
        compacted_paths[leaf] = _compact_path_to_intervals(path, feature_names)

    next_id = 0
    node_children = {}
    node_label = {}
    node_src_idx = {}
    leaf_by_node = {}

    def new_internal(label_text, src_idx=None):
        nonlocal next_id
        node_id = f"N{next_id}"
        next_id += 1
        node_children[node_id] = {}
        node_label[node_id] = label_text
        node_src_idx[node_id] = src_idx
        return node_id

    root_id = new_internal("ROOT", src_idx=None)

    for leaf_idx, cond_list in compacted_paths.items():
        current = root_id
        for cond_info in cond_list:
            text = cond_info["text"]
            src_idx = cond_info["node_idx"]

            children = node_children[current]
            if text not in children:
                child_id = new_internal(text, src_idx=src_idx)
                children[text] = child_id
            current = children[text]
        leaf_by_node[current] = leaf_idx

    def prob_of_src(src_idx: int) -> float:
        if src_idx is None or src_idx < 0:
            return 0.0
        return _node_prob_for_class(tree_, src_idx, cidx)

    dot = graphviz.Digraph(
        graph_attr={
            "rankdir": "TB",
            "splines": "true",
            "fontname": "Helvetica",
            "dpi": "300",
            "label": (
                f"Clase: {clase} | Confianza: τ={tau:.2f} | "
                f"Soporte: ≥{soporte} muestras | ÁRBOL COMPACTO (NO ESTRICTO)"
            ),
            "labelloc": "t",
        },
        node_attr={
            "shape": "box",
            "style": "rounded,filled",
            "fontname": "Helvetica",
            "fontsize": "11",
            "penwidth": "1.6",
            "color": "black",
        },
        edge_attr={
            "fontname": "Helvetica",
            "fontsize": "10",
            "color": "black",
            "arrowsize": "0.8",
        },
    )

    def draw(node_id):
        src_idx = node_src_idx.get(node_id, None)
        p_here = prob_of_src(src_idx) if src_idx is not None else 0.0

        if node_id == root_id:
            dot.node(node_id, label="ROOT", fillcolor=color_clase)
        else:
            base_cond = node_label[node_id]
            if src_idx is not None and src_idx >= 0:
                w, probs, samples = _node_counts_and_probvec(tree_, src_idx)
                gini = float(tree_.impurity[src_idx])
                p_c = float(probs[cidx]) if probs.size > 0 else 0.0
                pred_lbl = class_names_list[int(np.argmax(w))]
                label = (
                    f"{base_cond}\n"
                    f"gini = {gini:.3f} | samples = {samples}\n"
                    f"p({clase}) = {p_c:.3f}\n"
                    f"class = {pred_lbl}"
                )
            else:
                label = base_cond

            dot.node(node_id, label=label, fillcolor=color_clase)

        for _, child_id in node_children[node_id].items():
            dot.edge(node_id, child_id, label="")
            draw(child_id)

        if node_id in leaf_by_node:
            leaf_idx = leaf_by_node[node_id]
            w_leaf, probs_leaf, samples_leaf = _node_counts_and_probvec(tree_, leaf_idx)
            p_c_leaf = float(probs_leaf[cidx]) if probs_leaf.size > 0 else 0.0
            est_counts_leaf = np.rint(probs_leaf * samples_leaf).astype(int)
            pred_lbl_leaf = class_names_list[int(np.argmax(w_leaf))]

            leaf_label = (
                f"samples = {samples_leaf}\n"
                f"value = {_arr_str_int(est_counts_leaf)}\n"
                f"p({clase}) = {p_c_leaf:.3f}\n"
                f"class = {pred_lbl_leaf}"
            )
            leaf_node_id = f"L{leaf_idx}"
            dot.node(leaf_node_id, label=leaf_label,
                     shape="ellipse", fillcolor=color_clase)
            dot.edge(node_id, leaf_node_id, label="")

    draw(root_id)
    return dot


def build_compacted_graphviz_strict(modelo, clase, tau, soporte,
                                    keep_mask, feature_names,
                                    class_names_list, color_clase):
    """
    MODO ESTRICTO (PARTE 13):
      - Usa _keep_mask_strict_monotonic (p_hijo >= p_padre).
      - Compacta desigualdades por feature.
      - Aplica poda p_hijo < p_padre en el árbol virtual.
    """
    tree_ = modelo.tree_
    cidx = class_names_list.index(clase)

    paths = _get_paths_for_class(tree_, keep_mask)
    if not paths:
        return None

    compacted_paths = {}
    for leaf, path in paths.items():
        compacted_paths[leaf] = _compact_path_to_intervals(path, feature_names)

    next_id = 0
    node_children = {}
    node_label = {}
    node_src_idx = {}
    leaf_by_node = {}

    def new_internal(label_text, src_idx=None):
        nonlocal next_id
        node_id = f"N{next_id}"
        next_id += 1
        node_children[node_id] = {}
        node_label[node_id] = label_text
        node_src_idx[node_id] = src_idx
        return node_id

    root_id = new_internal("ROOT", src_idx=None)

    for leaf_idx, cond_list in compacted_paths.items():
        current = root_id
        for cond_info in cond_list:
            text = cond_info["text"]
            src_idx = cond_info["node_idx"]

            children = node_children[current]
            if text not in children:
                child_id = new_internal(text, src_idx=src_idx)
                children[text] = child_id
            current = children[text]
        leaf_by_node[current] = leaf_idx

    def prob_of_src(src_idx: int) -> float:
        if src_idx is None or src_idx < 0:
            return 0.0
        return _node_prob_for_class(tree_, src_idx, cidx)

    dot = graphviz.Digraph(
        graph_attr={
            "rankdir": "TB",
            "splines": "true",
            "fontname": "Helvetica",
            "dpi": "300",
            "label": (
                f"Clase: {clase} | Confianza: τ={tau:.2f} | "
                f"Soporte: ≥{soporte} muestras | "
                f"ÁRBOL COMPACTO (MONOTONÍA ESTRICTA)"
            ),
            "labelloc": "t",
        },
        node_attr={
            "shape": "box",
            "style": "rounded,filled",
            "fontname": "Helvetica",
            "fontsize": "11",
            "penwidth": "1.6",
            "color": "black",
        },
        edge_attr={
            "fontname": "Helvetica",
            "fontsize": "10",
            "color": "black",
            "arrowsize": "0.8",
        },
    )

    def draw(node_id, parent_p: float = 0.0):
        src_idx = node_src_idx.get(node_id, None)
        p_here = prob_of_src(src_idx) if src_idx is not None else parent_p

        if node_id == root_id:
            dot.node(node_id, label="ROOT", fillcolor=color_clase)
        else:
            base_cond = node_label[node_id]
            if src_idx is not None and src_idx >= 0:
                w, probs, samples = _node_counts_and_probvec(tree_, src_idx)
                gini = float(tree_.impurity[src_idx])
                p_c = float(probs[cidx]) if probs.size > 0 else 0.0
                pred_lbl = class_names_list[int(np.argmax(w))]
                label = (
                    f"{base_cond}\n"
                    f"gini = {gini:.3f} | samples = {samples}\n"
                    f"p({clase}) = {p_c:.3f}\n"
                    f"class = {pred_lbl}"
                )
            else:
                label = base_cond

            dot.node(node_id, label=label, fillcolor=color_clase)

        for _, child_id in node_children[node_id].items():
            src_child = node_src_idx.get(child_id, None)
            p_child = prob_of_src(src_child) if src_child is not None else p_here

            if p_child < p_here:
                continue  # poda estricta

            dot.edge(node_id, child_id, label="")
            draw(child_id, p_here)

        if node_id in leaf_by_node:
            leaf_idx = leaf_by_node[node_id]
            w_leaf, probs_leaf, samples_leaf = _node_counts_and_probvec(tree_, leaf_idx)
            p_c_leaf = float(probs_leaf[cidx]) if probs_leaf.size > 0 else 0.0
            est_counts_leaf = np.rint(probs_leaf * samples_leaf).astype(int)
            pred_lbl_leaf = class_names_list[int(np.argmax(w_leaf))]

            leaf_label = (
                f"samples = {samples_leaf}\n"
                f"value = {_arr_str_int(est_counts_leaf)}\n"
                f"p({clase}) = {p_c_leaf:.3f}\n"
                f"class = {pred_lbl_leaf}"
            )
            leaf_node_id = f"L{leaf_idx}"
            dot.node(leaf_node_id, label=leaf_label,
                     shape="ellipse", fillcolor=color_clase)
            dot.edge(node_id, leaf_node_id, label="")

    draw(root_id, parent_p=0.0)
    return dot

def top_variables_generalizado(tree_, feature_names, topk=5):
    """
    Variables más influyentes del ÁRBOL GENERALIZADO.
    Mismo criterio del especialista: cuenta cuántas veces aparece cada variable
    en nodos internos (no hojas).
    """
    counts = Counter()

    for u in range(tree_.node_count):
        if tree_.children_left[u] == -1:
            continue  # ignorar hojas
        feat_idx = int(tree_.feature[u])
        if 0 <= feat_idx < len(feature_names):
            counts[feature_names[feat_idx]] += 1

    return [name for name, _ in counts.most_common(topk)]


# -----------------------------------------------------------------------------
# PANEL PRINCIPAL: COLUMNA IZQUIERDA (CONTROL Y FILTRADO)
# -----------------------------------------------------------------------------
col1, col2 = st.columns([1, 3])

with col1:
    # Base de datos seleccionada
    st.markdown(
        f"""
        <div style="background-color:#FFFFFF;
                    padding:8px 16px;
                    border-radius:8px;
                    color:black;
                    font-weight:bold;
                    margin-bottom:8px;">
            Base de Datos: {nombre_bd}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Indicador visual de clase
    st.markdown(
        f"""
        <div style="background-color:{color_clase_hex};
                    padding:3px !important;
                    border-radius:8px;
                    text-align:center;
                    color:black;
                    border:1px solid black;
                    margin-bottom:18px;">
            <h4 style="margin:0; color:black;">Clase Objetivo:</h4>
            <h3 style="margin:0; color:black;">{clase_elegida}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 3. Filtros de Simplificación")

    modo_arbol = st.radio(
        "Tipo de Árbol",
        (
            "Modo NO ESTRICTO (Original/Exploratorio)",
            "Modo ESTRICTO (+ Monotonía probabilística)"
        ),
        index=0,
        key="modo_arbol"
    )

    with st.expander("Guía rápida de filtros", expanded=False):
        st.markdown(
            """
            <div style="font-size:0.80rem; line-height:1.25;">
            <b>Modo NO ESTRICTO:</b> muestra reglas que cumplen los umbrales.<br/>
            <b>Modo ESTRICTO:</b> además exige que la probabilidad de la clase no baje en el camino.<br/>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- sugerir defaults "inteligentes" para esta BD y esta clase ---
    conf_sug, sop_sug, motivo = sugerir_filtros_iniciales(
        modelo=modelo,
        class_idx=idx_objetivo,
        total_registros=total_registros,
        tau_pref=0.90,
        soporte_pct_pref=1.5
    )

    # Solo resetear valores cuando cambie BD o clase (si no, molesta al usuario)
    bd_prev = st.session_state.get("bd_prev")
    cls_prev = st.session_state.get("cls_prev")

    if (bd_prev != nombre_bd) or (cls_prev != idx_objetivo):
        st.session_state["confianza_pct"] = conf_sug
        st.session_state["soporte_pct"] = sop_sug
        st.session_state["bd_prev"] = nombre_bd
        st.session_state["cls_prev"] = idx_objetivo

    # Inputs SIEMPRE (sin value= para evitar warning por session_state)
    st.number_input(
        "Confianza Mínima (%)",
        min_value=0,
        max_value=100,
        step=5,
        key="confianza_pct"
    )

    st.number_input(
        "Soporte Mínimo (% Total)",
        min_value=0.1,
        max_value=50.0,
        step=0.5,
        format="%.2f",
        key="soporte_pct"
    )

    #  Leer valores finales desde session_state
    confianza_pct = float(st.session_state["confianza_pct"])
    soporte_pct = float(st.session_state["soporte_pct"])

    #  Cálculos DENTRO de col1 (siempre definidos)
    soporte_absoluto = int(total_registros * (soporte_pct / 100.0))
    if soporte_absoluto < 1:
        soporte_absoluto = 1

    st.caption(f"Soporte absoluto: {soporte_absoluto} muestras.")

    tau = confianza_pct / 100.0

    n_ok, conf_ok, sup_ok = diagnostico_con_filtros(
        modelo, idx_objetivo, tau, soporte_absoluto
    )
    
    if n_ok == 0:
        st.info(
            "Con los filtros actuales no existen reglas que sean simultáneamente "
            "muy confiables y representativas.\n\n"
            "Sugerencia: reduce la **Confianza mínima** o el **Soporte mínimo** para explorar más explicaciones."
        )
        # -----------------------------
    # Calcular keep_mask del árbol especialista (para variables influyentes)
    # -----------------------------
    tree_ = modelo.tree_
    cidx = idx_objetivo

    modo_no_estricto = ("no estricto" in modo_arbol.lower()) or (
        "no" in modo_arbol.lower() and "estrict" in modo_arbol.lower()
    )

    if modo_no_estricto:
        keep_mask = _keep_mask_non_strict(
            tree_,
            cidx,
            tau,
            min_samples_to_keep=soporte_absoluto
        )
    else:
        keep_mask = _keep_mask_strict_monotonic(
            tree_,
            cidx,
            tau,
            min_samples_to_keep=soporte_absoluto
        )

     # -----------------------------
    # Variables más influyentes (árbol especialista)
    # -----------------------------
    st.markdown("**Variables más influyentes (árbol especialista):**")

    if keep_mask is not None and keep_mask.any():
        top_vars = top_variables_influyentes(
            tree_=tree_,
            keep_mask=keep_mask,
            feature_names=feat_names,
            topk=5
        )

        if len(top_vars) == 0:
            st.caption("—")
        else:
            for v in top_vars:
                st.markdown(f"- `{v}`")
    else:
        st.caption("—")

# -----------------------------------------------------------------------------
# PANEL PRINCIPAL: COLUMNA DERECHA (COMPARACIÓN + ÁRBOL)
# -----------------------------------------------------------------------------
with col2:
    # -----------------------------
    # Resumen del árbol (profundidad/nodos/hojas)
    # -----------------------------
    def resumen_arbol(modelo):
        t = modelo.tree_
        n_nodes = int(t.node_count)
        max_depth = int(t.max_depth)
        n_leaves = int(np.sum(t.children_left == -1))
        return max_depth, n_nodes, n_leaves
    
    max_depth, n_nodes, n_leaves = resumen_arbol(modelo)
    
    # -----------------------------
    # TARJETA: ÁRBOL GENERALIZADO
    st.subheader("🌳ÁRBOL GENERALIZADO")

    st.markdown(
    f"""<div style="background-color:#FFFFFF;
                padding:14px 16px;
                border-radius:10px;
                color:black;
                font-size:0.92rem;
                border: 1px solid #e0e0e0;
                line-height:1.45;">
    
      <div style="margin-bottom:8px;">
        <b>Resumen del árbol:</b> profundidad máx = {max_depth} | nodos = {n_nodes} | hojas = {n_leaves}.
      </div>
    
      <div style="font-size:0.85rem; color:#555;">
        Este árbol corresponde al modelo completo entrenado, sin aplicar filtros ni umbrales.
        Las explicaciones por clase se presentan mediante el Árbol Especialista.
      </div>
    
    </div>""",
    unsafe_allow_html=True
    )

 
    if max_depth >= 12:
        st.info("Árbol profundo: puede ser difícil de leer ampliando, descarga el Árbol Generalizado (PNG).")


    # -----------------------------
    # Árbol generalizado (comparación) + descarga PNG
    # -----------------------------
    with st.expander("🆚 Comparar con Árbol Generalizado (Clic para desplegar)", expanded=False):

        st.markdown(f"### {nombre_bd}")
    
 
        # -----------------------------
        # Variables más influyentes (Árbol Generalizado)
        # -----------------------------
        st.markdown("**Variables más influyentes (árbol generalizado):**")
    
        try:
            top_vars_global = top_variables_generalizado(
                tree_=modelo.tree_,
                feature_names=feat_names,
                topk=5
            )
    
            if len(top_vars_global) == 0:
                st.caption("—")
            else:
                for v in top_vars_global:
                    st.markdown(f"- `{v}`")
    
        except Exception as e:
            st.caption("—")
            st.warning(
                f"No fue posible calcular variables influyentes del árbol generalizado: {e}"
            )
    
        prefijo_bd = nombre_bd.split('_')[0]
        nombre_imagen_global = f"ARBOL_GENERALIZADO_{prefijo_bd}.png"
    
        if os.path.exists(nombre_imagen_global):
    
            st.image(
                nombre_imagen_global,
                caption=f"Modelo Generalizado - {prefijo_bd}",
                use_container_width=True
            )
    
            with open(nombre_imagen_global, "rb") as f:
                st.download_button(
                    "⬇️ Descargar Árbol Generalizado (PNG)",
                    data=f.read(),
                    file_name=nombre_imagen_global,
                    mime="image/png"
                )
    
        else:
            st.warning(
                f"No se encontró la imagen '{nombre_imagen_global}'. "
                f"Asegúrate de tenerla en la carpeta."
            )


    # -----------------------------
    # Árbol especialista
    # -----------------------------
    st.subheader("🌳 ÁRBOL ESPECIALISTA")

    tree_ = modelo.tree_
    cidx = idx_objetivo

    # Condición robusta (no depende de mayúsculas/minúsculas exactas)
    modo_no_estricto = ("no estricto" in modo_arbol.lower()) or ("no" in modo_arbol.lower() and "estrict" in modo_arbol.lower())

    if modo_no_estricto:
        keep_mask = _keep_mask_non_strict(
            tree_,
            cidx,
            tau,
            min_samples_to_keep=soporte_absoluto
        )
        st.caption(f"Modo NO estricto: nodos_keep={int(keep_mask.sum())}")

        if keep_mask.any():
            g = build_compacted_graphviz_non_strict(
                modelo=modelo,
                clase=class_names[cidx],
                tau=tau,
                soporte=soporte_absoluto,
                keep_mask=keep_mask,
                feature_names=feat_names,
                class_names_list=class_names,
                color_clase=color_clase_hex
            )
        else:
            g = None
    else:
        keep_mask = _keep_mask_strict_monotonic(
            tree_,
            cidx,
            tau,
            min_samples_to_keep=soporte_absoluto
        )
        st.caption(f"Modo ESTRICTO: nodos_keep={int(keep_mask.sum())}")

        if keep_mask.any():
            g = build_compacted_graphviz_strict(
                modelo=modelo,
                clase=class_names[cidx],
                tau=tau,
                soporte=soporte_absoluto,
                keep_mask=keep_mask,
                feature_names=feat_names,
                class_names_list=class_names,
                color_clase=color_clase_hex
            )
        else:
            g = None

   
    if g is not None:
        mostrar_dot_en_streamlit(g)
    
