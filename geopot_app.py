"""
GeoPot — Simulador de Potencial Gravitacional Zonal
=====================================================

Software de apoyo para el cálculo del potencial gravitacional de un cuerpo
elipsoidal en equilibrio hidrostático, mediante el desarrollo en armónicos
zonales (polinomios de Legendre) según la teoría de Clairaut.

Referencia teórica: Avila, M. "Geodesia Física" (ecuación 2.76 y Tabla 5).

Estructura del programa
------------------------
1. constants.py (embebido)  -> Parámetros físicos de referencia (WGS84).
2. GeodesyEngine             -> Núcleo matemático: toda la física vive aquí,
                                 desacoplada de la interfaz.
3. render_*()                -> Funciones de presentación (Streamlit).
4. main()                    -> Orquesta la aplicación.

Ejecutar con:
    streamlit run geopot_app.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.special import legendre


# =====================================================================
# 1. CONSTANTES DE REFERENCIA — WGS84, Tabla 3 (Avila, "Geodesia Física")
# =====================================================================

@dataclass(frozen=True)
class WGS84:
    """Parámetros físicos y geométricos del elipsoide de referencia WGS84."""

    GM: float = 3.986004418e14        # Constante kepleriana (m^3/s^2)
    a: float = 6_378_137.0            # Semieje mayor (m)
    b: float = 6_356_752.3142         # Semieje menor (m)
    C: float = 8.0354872e37           # Momento de inercia polar (kg*m^2)
    A: float = 8.0091029e37           # Momento de inercia ecuatorial (kg*m^2)
    M: float = 5.9733328e24           # Masa terrestre (kg)


# Coeficientes zonales oficiales (Tabla 5) para validar el cálculo
TABLA_5_OFICIAL: Dict[int, float] = {
    1: 1.08262982131e-3,
    2: -2.37091120053e-6,
    3: 6.08346498882e-9,
    4: -1.42681087920e-10,
    5: 1.21439275882e-13,
}


# =====================================================================
# 2. NÚCLEO MATEMÁTICO — independiente de la interfaz
# =====================================================================

@dataclass
class ResultadoCalculo:
    """Contenedor con cada paso intermedio del cálculo, para trazabilidad."""

    a2: float
    b2: float
    E2: float
    f: float
    e2: float
    r: float
    theta_deg: float
    cos_theta: float
    P2n: float
    J2n: float
    V: float
    n: int
    grado: int


class GeodesyEngine:
    """Encapsula la física del desarrollo del potencial en armónicos zonales.

    Todos los métodos son puros (sin efectos secundarios), lo que permite
    probarlos de forma aislada y reutilizarlos tanto en la interfaz gráfica
    como en un futuro script de línea de comandos o notebook.
    """

    def __init__(self, params: WGS84 = WGS84()):
        self.params = params

    # -- geometría del elipsoide -------------------------------------
    def flattening(self, a: float, b: float) -> float:
        """Achatamiento geométrico f = (a - b) / a."""
        return (a - b) / a

    def eccentricity_sq(self, a: float, b: float) -> float:
        """Excentricidad al cuadrado e^2 = 1 - (b/a)^2."""
        return 1.0 - (b / a) ** 2

    def radius_at_latitude(self, a: float, f: float, phi_rad: float) -> float:
        """Radio r(phi) = a(1 - f sen^2 phi)."""
        return a * (1.0 - f * np.sin(phi_rad) ** 2)

    # -- coeficientes armónicos ---------------------------------------
    def zonal_harmonic(self, n: int, e2: float) -> float:
        """Coeficiente J_2n según la teoría de Clairaut.

        J_2n = (-1)^(n+1) * [3 e^(2n) / ((2n+1)(2n+3))] * [1 - n + 5n * ratio]
        donde ratio = (C - A) / (M a^2 e^2).
        """
        p = self.params
        ratio = (p.C - p.A) / (p.M * p.a ** 2 * e2)
        signo = (-1) ** (n + 1)
        termino = 3.0 * e2 ** n / ((2 * n + 1) * (2 * n + 3))
        factor = 1 - n + 5 * n * ratio
        return signo * termino * factor

    # -- potencial ------------------------------------------------------
    def potential(self, n: int, phi_deg: float, a: float = None, b: float = None) -> ResultadoCalculo:
        """Calcula V(phi) para un orden armónico n, guardando cada paso."""
        p = self.params
        a = a if a is not None else p.a
        b = b if b is not None else p.b

        a2, b2 = a ** 2, b ** 2
        E2 = a2 - b2
        f = self.flattening(a, b)
        e2 = self.eccentricity_sq(a, b)

        phi_rad = np.radians(phi_deg)
        r = self.radius_at_latitude(a, f, phi_rad)
        theta_deg = 90.0 - phi_deg
        cos_theta = np.sin(phi_rad)  # cos(theta) = cos(90-phi) = sin(phi)

        grado = 2 * n
        P2n = float(legendre(grado)(cos_theta))
        J2n = self.zonal_harmonic(n, e2)

        termino_armonico = (a / r) ** grado * J2n * P2n
        V = (p.GM / r) * (1.0 + termino_armonico)

        return ResultadoCalculo(a2, b2, E2, f, e2, r, theta_deg, cos_theta, P2n, J2n, V, n, grado)

    def potential_field(self, n: int, J2n: float, a: float, b: float,
                         lats_deg: np.ndarray, lons_deg: np.ndarray):
        """Evalúa V sobre una malla de latitudes/longitudes (para el 3D)."""
        f = self.flattening(a, b)
        Lats, Lons = np.meshgrid(lats_deg, lons_deg, indexing="ij")
        lat_rad, lon_rad = np.radians(Lats), np.radians(Lons)

        r = self.radius_at_latitude(a, f, lat_rad)
        X = r * np.cos(lat_rad) * np.cos(lon_rad)
        Y = r * np.cos(lat_rad) * np.sin(lon_rad)
        Z = (b / a) * r * np.sin(lat_rad)

        grado = 2 * n
        P = legendre(grado)(np.sin(lat_rad))
        termino = (a / r) ** grado * J2n * P
        V = (self.params.GM / r) * (1.0 + termino)
        return X, Y, Z, V

    def validar_contra_tabla(self, n: int, J2n: float) -> tuple[float, float] | None:
        """Compara J_2n calculado contra el valor oficial de la Tabla 5."""
        if n not in TABLA_5_OFICIAL:
            return None
        oficial = TABLA_5_OFICIAL[n]
        diferencia_pct = abs((J2n - oficial) / oficial) * 100
        return oficial, diferencia_pct


# =====================================================================
# 3. CAPA DE PRESENTACIÓN (Streamlit)
# =====================================================================

def render_sidebar(params: WGS84) -> tuple[int, float]:
    """Dibuja el panel lateral y devuelve (n, phi) ingresados por el usuario."""
    with st.sidebar:
        st.markdown("## ⚙️ Parámetros del sistema")
        st.caption("Fijos — WGS84, Tabla 3 (Avila, *Geodesia Física*, pág. 19)")
        st.code(
            f"GM = {params.GM:.6e} m³/s²\n"
            f"a  = {params.a:,.4f} m\n"
            f"b  = {params.b:,.4f} m\n"
            f"C  = {params.C:.6e} kg·m²\n"
            f"A  = {params.A:.6e} kg·m²\n"
            f"M  = {params.M:.6e} kg",
            language="text",
        )

        st.markdown("## 🎛️ Entradas")
        n = st.number_input(
            "Orden armónico n  (n=1 → J₂, n=2 → J₄, ...)",
            min_value=1, max_value=10, value=1, step=1,
        )
        phi = st.number_input(
            "Latitud φ (grados)",
            min_value=-90.0, max_value=90.0, value=45.0, format="%g",
        )
        st.divider()
        st.caption("GeoPot v1.0 · Motor: teoría de Clairaut para el elipsoide de nivel")
    return int(n), float(phi)


def render_header():
    st.set_page_config(page_title="GeoPot — Potencial Gravitacional Zonal", layout="wide")
    st.title("🌍 GeoPot — Simulador de Potencial Gravitacional Zonal")
    st.caption(
        "Cálculo del potencial gravitacional mediante desarrollo en armónicos "
        "zonales (Legendre), según la teoría de Clairaut para un elipsoide en "
        "equilibrio hidrostático."
    )


def render_steps(res: ResultadoCalculo):
    st.subheader(f"Desarrollo para n = {res.n}  (grado 2n = {res.grado})")
    c1, c2 = st.columns(2)
    with c1:
        st.latex(rf"a^2 = {res.a2:,.2f}\ \ ,\ \ b^2 = {res.b2:,.2f}")
        st.latex(rf"E^2 = a^2-b^2 = {res.E2:,.2f}")
        st.latex(rf"f = \dfrac{{a-b}}{{a}} = {res.f:.8f}")
        st.latex(rf"r(\varphi) = a(1-f\,\mathrm{{sen}}^2\varphi) = {res.r:,.3f}\ \text{{m}}")
    with c2:
        st.latex(rf"\theta = 90^\circ-\varphi = {res.theta_deg:.4f}^\circ")
        st.latex(rf"P_{{{res.grado}}}(\cos\theta) = {res.P2n:.6f}")
        st.latex(rf"J_{{{res.grado}}} = {res.J2n:.6e}")
        st.latex(rf"V = {res.V:.6e}\ \text{{m}}^2/\text{{s}}^2")


def render_validation(engine: GeodesyEngine, res: ResultadoCalculo):
    st.subheader("Validación contra la Tabla 5 del libro")
    check = engine.validar_contra_tabla(res.n, res.J2n)
    if check is None:
        st.info("La Tabla 5 del libro no cubre este orden — no hay valor de referencia.")
        return
    oficial, diff = check
    col1, col2, col3 = st.columns(3)
    col1.metric(f"J{res.grado} calculado", f"{res.J2n:.4e}")
    col2.metric(f"J{res.grado} oficial (libro)", f"{oficial:.4e}")
    col3.metric("Diferencia", f"{diff:.2f} %",
                delta="Coherente" if diff < 5 else "Revisar", delta_color="off")


def render_3d(engine: GeodesyEngine, res: ResultadoCalculo, params: WGS84):
    st.subheader(f"Efecto del armónico de orden {res.grado} sobre el elipsoide")
    lats = np.linspace(-90, 90, 45)
    lons = np.linspace(-180, 180, 45)
    X, Y, Z, V = engine.potential_field(res.n, res.J2n, params.a, params.b, lats, lons)

    phi_rad = np.radians(90.0 - res.theta_deg)
    x_pt = res.r * np.cos(phi_rad)
    z_pt = (params.b / params.a) * res.r * np.sin(phi_rad)

    fig = go.Figure(go.Surface(
        x=X / 1e3, y=Y / 1e3, z=Z / 1e3,
        surfacecolor=V / 1e6, colorscale="Viridis",
        colorbar=dict(title="MJ/kg"),
    ))
    fig.add_trace(go.Scatter3d(
        x=[x_pt / 1e3], y=[0], z=[z_pt / 1e3],
        mode="markers", marker=dict(size=8, color="red"),
        name="Punto evaluado",
    ))
    fig.update_layout(
        scene=dict(xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)"),
        margin=dict(l=0, r=0, b=0, t=10), height=560,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_explanation(res: ResultadoCalculo, phi: float):
    lat_abs = abs(phi)
    tipo = "ecuatorial/bajo" if lat_abs < 25 else ("latitud media" if lat_abs <= 65 else "polar/alto")
    st.subheader("Lectura física del resultado")
    st.markdown(
        f"- El orden **n={res.n}** define el grado par **2n={res.grado}** del coeficiente "
        f"**J₍{res.grado}₎ = {res.J2n:.4e}**.\n"
        f"- En **φ = {phi:g}°** (zona {tipo}), el polinomio de Legendre vale "
        f"**P₍{res.grado}₎(cos θ) = {res.P2n:.4f}**.\n"
        f"- El potencial resultante es **V = {res.V:.4e} m²/s²** "
        f"({res.V/1e6:.4f} MJ/kg)."
    )


# =====================================================================
# 4. ORQUESTACIÓN
# =====================================================================

def main():
    render_header()
    params = WGS84()
    engine = GeodesyEngine(params)

    n, phi = render_sidebar(params)
    resultado = engine.potential(n, phi)

    tab_calculo, tab_3d, tab_validacion = st.tabs(
        ["📐 Desarrollo paso a paso", "🌐 Visualización 3D", "✅ Validación"]
    )
    with tab_calculo:
        render_steps(resultado)
        render_explanation(resultado, phi)
    with tab_3d:
        render_3d(engine, resultado, params)
    with tab_validacion:
        render_validation(engine, resultado)


if __name__ == "__main__":
    main()
