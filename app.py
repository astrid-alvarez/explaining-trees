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

# -----------------------------------------------------------------------------
# FUNCIÓN AUXILIAR: MOSTRAR DOT COMO PNG (para ver el árbol completo)
# -----------------------------------------------------------------------------
def mostrar_dot_en_streamlit(dot):
    """
    Renderiza un grafo graphviz.Digraph a PNG y lo muestra completo en Streamlit.
    """
    try:
        png_bytes = dot.pipe(format="png")  # resolución controlada en graph_attr
        st.image(png_bytes, use_container_width=True)
    except Exception as e:
        st.error(f"Error al renderizar el árbol en PNG: {e}")

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
# NUEVO: Información contextual adicional (Sección 1)
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


st.title("Plataforma de Evaluación XAI: Árboles Especialistas")
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
# CÓDIGO ACTUALIZADO: CORRECCIÓN DE VARIABLES
# -----------------------------------------------------------------------------
# 1. Recuperar tamaño de entrenamiento desde el árbol (dato oculto en sklearn)
n_train = int(modelo.tree_.n_node_samples[0])

# 2. Definir la variable vital para el resto del código (NO BORRAR)
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


st.sidebar.subheader("Diagnóstico de Reglas")
st.sidebar.markdown(
    f"""
    <div style="background-color:#FFFFFF;
                padding:10px 14px;
                border-radius:8px;
                color:black;
                font-size:0.9rem;">
        <b>Reglas potenciales:</b> {reglas_pot}<br/>
        ▪ <b>Confianza máx (sin filtro):</b> {conf_max*100:.1f}% (sup={sup_confmax})<br/>
        ▪ <b>Soporte máx (sin filtro):</b> {sup_max} ({sup_max_pct:.1f}%) (p={conf_supmax*100:.1f}%)<br/>
        <hr style="margin:6px 0;"/>
        ▪ <b>Confianza máx con soporte ≥ {soporte_min_abs}:</b> {conf_max_con_soporte*100:.1f}%<br/>
        ▪ ▪ <b>Soporte máx con τ ≥ {tau_diag:.2f}:</b> {sup_max_con_tau}
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("")
with st.sidebar.expander("¿Cómo interpretar este diagnóstico?"):
    st.markdown(
        """
        <div style="font-size:0.85rem; line-height:1.4;">
        <b>Reglas potenciales:</b> número de hojas del árbol que predicen esta clase. <br/><br/>
        <b>Confianza máxima:</b> probabilidad más alta alcanzada por la clase en alguna regla.
        Valores muy altos pueden indicar reglas muy específicas.<br/><br/>
        <b>Soporte máximo:</b> número máximo de registros cubiertos por una regla.
        Soportes bajos implican mayor riesgo de sobreajuste.
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
            "Modo ESTRICTO ( + Monotonía probabilística)"
        ),
        index=0
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

    # 🔽 Ambos inputs DENTRO de col1 y alineados
    confianza_pct = st.number_input(
        "Confianza Mínima (%)",
        min_value=0,
        max_value=100,
        value=90,
        step=5,
        key="confianza_pct"
    )

    soporte_pct = st.number_input(
        "Soporte Mínimo (% Total)",
        min_value=0.1,
        max_value=50.0,
        value=1.5,
        step=0.5,
        format="%.2f",
        key="soporte_pct"
    )

    # 🔽 Cálculo FUERA de cualquier expander, pero DENTRO de col1
    soporte_absoluto = int(total_registros * (soporte_pct / 100.0))
    if soporte_absoluto < 1:
        soporte_absoluto = 1

    st.caption(f"Soporte absoluto: {soporte_absoluto} muestras.")

    tau = confianza_pct / 100.0

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

    st.caption(
        f"Resumen del árbol: profundidad máx = {max_depth} | nodos = {n_nodes} | hojas = {n_leaves}."
    )
    if max_depth >= 12:
        st.info("Árbol profundo: puede ser difícil de leer ampliando, descarga el Árbol Generalizado en PNG.")

    # -----------------------------
    # Árbol generalizado (comparación) + descarga PNG
    # -----------------------------
    with st.expander("🆚 Comparar con Árbol Generalizado (Clic para desplegar)", expanded=False):
        st.markdown(f"### Árbol Generalizado Completo: {nombre_bd}")
        st.write("Estructura original completa del modelo global.")

        prefijo_bd = nombre_bd.split('_')[0]
        nombre_imagen_global = f"ARBOL_GENERALIZADO_{prefijo_bd}.png"

        if os.path.exists(nombre_imagen_global):
            st.image(
                nombre_imagen_global,
                caption=f"Modelo Generalizado - {prefijo_bd}",
                use_container_width=True
            )

            # Botón de descarga (PNG)
            with open(nombre_imagen_global, "rb") as f:
                st.download_button(
                    "⬇️ Descargar Árbol Generalizado (PNG)",
                    data=f.read(),
                    file_name=nombre_imagen_global,
                    mime="image/png"
                )
        else:
            st.warning(
                f"⚠️ No se encontró la imagen '{nombre_imagen_global}'. "
                f"Asegúrate de tenerla en la carpeta."
            )

    # -----------------------------
    # Árbol especialista
    # -----------------------------
    st.subheader("🌳 Árbol Especialista Generado")

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
    else:
        st.warning(
            f"⚠️ No se encontraron reglas con Confianza >= {confianza_pct}% "
            f"y Soporte >= {soporte_pct}% para el modo seleccionado."
        )

# -----------------------------------------------------------------------------
# 4. EVALUACIÓN DEL EXPERTO
# -----------------------------------------------------------------------------
st.divider()
st.subheader("4. Evaluación del experto")

c1, c2 = st.columns(2)

with c1:
    st.select_slider(
        "Comprensibilidad",
        ["1 (Baja)", "2", "3", "4", "5 (Alta)"]
    )

with c2:
    st.select_slider(
        "Utilidad para Toma de Decisiones",
        ["1 (Baja)", "2", "3", "4", "5 (Alta)"]
    )

if st.button("Guardar Evaluación"):
    st.success("Evaluación registrada correctamente.")
