"""
Potencial Gravitacional Zonal
=============================

Software de apoyo para el cálculo del potencial gravitacional de un cuerpo
elipsoidal en equilibrio hidrostático, mediante el desarrollo en armónicos
zonales (polinomios de Legendre) según la teoría de Clairaut.

Referencia teórica: Avila, M. "Geodesia Física" (ecuación 2.76 y Tabla 5).

Estructura del programa
------------------------
1. WGS84                 -> Parámetros físicos de referencia.
2. GeodesyEngine         -> Núcleo matemático: toda la física vive aquí,
                            desacoplada de la interfaz.
3. render_*()            -> Funciones de presentación (Streamlit).
4. main()                -> Orquesta la aplicación.

Ejecutar con:
    streamlit run geopot_app.py
"""

from __future__ import annotations

from dataclasses import dataclass

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


# =====================================================================
# 2. NÚCLEO MATEMÁTICO — independiente de la interfaz
# =====================================================================

@dataclass
class ResultadoCalculo:
    """Contenedor con cada paso intermedio del cálculo, para trazabilidad."""

    n: int
    grado: int
    phi_deg: float
    # geometría
    a: float
    b: float
    a2: float
    b2: float
    E2: float
    f: float
    e2: float
    sin2_phi: float
    r: float
    theta_deg: float
    cos_theta: float
    # polinomio de Legendre
    P2n: float
    # coeficiente zonal
    ratio: float      # (C - A) / (M a^2 e^2)
    signo: int        # (-1)^(n+1)
    coef: float       # 3 e^(2n) / ((2n+1)(2n+3))
    factor: float     # 1 - n + 5 n ratio
    J2n: float
    # potencial
    a_sobre_r: float
    a_r_pot: float    # (a/r)^(2n)
    term_arm: float   # (a/r)^(2n) * J2n * P2n
    V_kepler: float   # GM / r
    V: float


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
    def zonal_harmonic_parts(self, n: int, e2: float, a: float = None):
        """Devuelve (ratio, signo, coef, factor, J_2n) según Clairaut.

        J_2n = (-1)^(n+1) * [3 e^(2n) / ((2n+1)(2n+3))] * [1 - n + 5n * ratio]
        donde ratio = (C - A) / (M a^2 e^2).
        """
        p = self.params
        a = a if a is not None else p.a
        ratio = (p.C - p.A) / (p.M * a ** 2 * e2)
        signo = (-1) ** (n + 1)
        coef = 3.0 * e2 ** n / ((2 * n + 1) * (2 * n + 3))
        factor = 1 - n + 5 * n * ratio
        return ratio, signo, coef, factor, signo * coef * factor

    def zonal_harmonic(self, n: int, e2: float) -> float:
        """Coeficiente J_2n (solo el valor final)."""
        return self.zonal_harmonic_parts(n, e2)[-1]

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
        sin2_phi = float(np.sin(phi_rad) ** 2)
        r = float(self.radius_at_latitude(a, f, phi_rad))
        theta_deg = 90.0 - phi_deg
        cos_theta = float(np.sin(phi_rad))  # cos(theta) = cos(90-phi) = sin(phi)

        grado = 2 * n
        P2n = float(legendre(grado)(cos_theta))
        ratio, signo, coef, factor, J2n = self.zonal_harmonic_parts(n, e2, a)

        a_sobre_r = a / r
        a_r_pot = a_sobre_r ** grado
        term_arm = a_r_pot * J2n * P2n
        V_kepler = p.GM / r
        V = V_kepler * (1.0 + term_arm)

        return ResultadoCalculo(
            n=n, grado=grado, phi_deg=phi_deg,
            a=a, b=b, a2=a2, b2=b2, E2=E2, f=f, e2=e2,
            sin2_phi=sin2_phi, r=r, theta_deg=theta_deg, cos_theta=cos_theta,
            P2n=P2n,
            ratio=ratio, signo=signo, coef=coef, factor=factor, J2n=J2n,
            a_sobre_r=a_sobre_r, a_r_pot=a_r_pot, term_arm=term_arm,
            V_kepler=V_kepler, V=V,
        )

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


# =====================================================================
# 3. CAPA DE PRESENTACIÓN (Streamlit)
# =====================================================================

# -- utilidades de formato LaTeX --------------------------------------

def num(x: float, dec: int = 2) -> str:
    """Número con espacios finos como separador de miles (para LaTeX)."""
    return f"{x:,.{dec}f}".replace(",", r"\,")


def sci(x: float, digits: int = 6) -> str:
    """Notación científica en LaTeX: 1.5e-5 -> 1.500000 \\times 10^{-5}."""
    if x == 0:
        return "0"
    mant, exp = f"{x:.{digits}e}".split("e")
    return rf"{mant}\times 10^{{{int(exp)}}}"


def bloque(*lineas: str):
    """Muestra varias líneas alineadas en un solo bloque LaTeX."""
    st.latex(r"\begin{aligned}" + r" \\ ".join(lineas) + r"\end{aligned}")


def paso(k: int, titulo: str):
    st.markdown(f"**Paso {k} — {titulo}**")


# -- piezas de la interfaz -------------------------------------------

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
        st.caption("Motor: teoría de Clairaut para el elipsoide de nivel")
    return int(n), float(phi)


def render_header():
    st.set_page_config(page_title="Potencial Gravitacional Zonal", layout="wide")
    # La columna derecha (visualización) se queda fija al hacer scroll
    st.markdown(
        """
        <style>
        div[data-testid="stColumn"]:nth-of-type(2),
        div[data-testid="column"]:nth-of-type(2) {
            position: sticky; top: 4rem; align-self: flex-start;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("🌍 Potencial Gravitacional Zonal")
    st.caption(
        "Cálculo del potencial gravitacional mediante desarrollo en armónicos "
        "zonales (Legendre), según la teoría de Clairaut para un elipsoide en "
        "equilibrio hidrostático."
    )


def render_procedure(res: ResultadoCalculo, params: WGS84):
    """Desarrollo completo: fórmula → reemplazo → resultado, en cada paso."""
    n, g = res.n, res.grado
    st.subheader(f"Desarrollo para n = {n}  (grado 2n = {g})")

    # 1. Cuadrados de los semiejes
    paso(1, "Cuadrados de los semiejes")
    bloque(
        r"a^2 &= a \cdot a",
        rf"&= ({num(res.a, 4)})^2",
        rf"&= {num(res.a2)}\ \mathrm{{m^2}}",
    )
    bloque(
        r"b^2 &= b \cdot b",
        rf"&= ({num(res.b, 4)})^2",
        rf"&= {num(res.b2)}\ \mathrm{{m^2}}",
    )

    # 2. E^2
    paso(2, "Diferencia de cuadrados")
    bloque(
        r"E^2 &= a^2 - b^2",
        rf"&= {num(res.a2)} - {num(res.b2)}",
        rf"&= {num(res.E2)}\ \mathrm{{m^2}}",
    )

    # 3. Achatamiento
    paso(3, "Achatamiento geométrico")
    bloque(
        r"f &= \dfrac{a-b}{a}",
        rf"&= \dfrac{{{num(res.a, 4)} - {num(res.b, 4)}}}{{{num(res.a, 4)}}}",
        rf"&= {res.f:.10f}",
    )

    # 4. Excentricidad
    paso(4, "Excentricidad al cuadrado")
    bloque(
        r"e^2 &= 1 - \dfrac{b^2}{a^2}",
        rf"&= 1 - \dfrac{{{num(res.b2)}}}{{{num(res.a2)}}}",
        rf"&= {res.e2:.10f}",
    )

    # 5. Radio
    paso(5, "Radio del elipsoide a la latitud φ")
    bloque(
        r"r(\varphi) &= a\,(1 - f\,\mathrm{sen}^2\varphi)",
        rf"&= {num(res.a, 4)}\,\left(1 - {res.f:.8f}\cdot \mathrm{{sen}}^2({res.phi_deg:g}^\circ)\right)",
        rf"&= {num(res.a, 4)}\,\left(1 - {res.f:.8f}\cdot {res.sin2_phi:.8f}\right)",
        rf"&= {num(res.r, 3)}\ \mathrm{{m}}",
    )

    # 6. Colatitud
    paso(6, "Colatitud y coseno de la colatitud")
    bloque(
        r"\theta &= 90^\circ - \varphi",
        rf"&= 90^\circ - ({res.phi_deg:g}^\circ)",
        rf"&= {res.theta_deg:.4f}^\circ",
    )
    bloque(
        r"\cos\theta &= \mathrm{sen}\,\varphi",
        rf"&= \mathrm{{sen}}({res.phi_deg:g}^\circ)",
        rf"&= {res.cos_theta:.8f}",
    )

    # 7. Legendre
    paso(7, f"Polinomio de Legendre de grado {g}")
    bloque(
        r"P_{m}(x) &= \dfrac{1}{2^{m}\, m!}\,\dfrac{d^{m}}{dx^{m}}\,(x^2-1)^{m}"
        rf"\quad (m = 2n = {g})",
        rf"P_{{{g}}}(\cos\theta) &= P_{{{g}}}({res.cos_theta:.8f})",
        rf"&= {res.P2n:.8f}",
    )

    # 8. Cociente inercial
    paso(8, "Cociente de momentos de inercia")
    p = params
    bloque(
        r"\rho &= \dfrac{C-A}{M\,a^2\,e^2}",
        rf"&= \dfrac{{{sci(p.C, 7)} - {sci(p.A, 7)}}}"
        rf"{{({sci(p.M, 7)})\,({sci(res.a2, 7)})\,({res.e2:.10f})}}",
        rf"&= \dfrac{{{sci(p.C - p.A, 7)}}}{{{sci(p.M * res.a2 * res.e2, 7)}}}",
        rf"&= {res.ratio:.8f}",
    )

    # 9. Coeficiente zonal
    paso(9, f"Coeficiente zonal J_{{{g}}} (Clairaut)")
    bloque(
        rf"J_{{{g}}} &= (-1)^{{n+1}}\,\dfrac{{3\,(e^2)^{{n}}}}{{(2n+1)(2n+3)}}"
        r"\,\left[\,1 - n + 5n\,\rho\,\right]",
        rf"&= (-1)^{{{n}+1}}\,\dfrac{{3\,({res.e2:.10f})^{{{n}}}}}{{({2*n+1})({2*n+3})}}"
        rf"\,\left[\,1 - {n} + 5\cdot{n}\,({res.ratio:.8f})\,\right]",
        rf"&= ({res.signo})\,\left({sci(res.coef)}\right)\,\left({res.factor:.8f}\right)",
        rf"&= {sci(res.J2n)}",
    )

    # 10. Potencial
    paso(10, "Potencial gravitacional V(φ)")
    bloque(
        rf"\dfrac{{a}}{{r}} &= \dfrac{{{num(res.a, 4)}}}{{{num(res.r, 3)}}}"
        rf" = {res.a_sobre_r:.10f}",
        rf"\left(\dfrac{{a}}{{r}}\right)^{{{g}}} &= ({res.a_sobre_r:.10f})^{{{g}}}"
        rf" = {res.a_r_pot:.10f}",
    )
    bloque(
        rf"V &= \dfrac{{GM}}{{r}}\left[\,1 + \left(\dfrac{{a}}{{r}}\right)^{{2n}}"
        rf"J_{{{g}}}\,P_{{{g}}}(\cos\theta)\,\right]",
        rf"&= \dfrac{{{sci(p.GM, 9)}}}{{{num(res.r, 3)}}}"
        rf"\left[\,1 + ({res.a_r_pot:.10f})\,({sci(res.J2n)})\,({res.P2n:.8f})\,\right]",
        rf"&= \left({sci(res.V_kepler, 9)}\right)\left[\,1 + \left({sci(res.term_arm)}\right)\right]",
        rf"&= {sci(res.V, 9)}\ \mathrm{{m^2/s^2}}",
    )


def render_reading(res: ResultadoCalculo):
    """Interpretación física del resultado."""
    lat_abs = abs(res.phi_deg)
    zona = "ecuatorial / baja" if lat_abs < 25 else ("latitud media" if lat_abs <= 65 else "polar / alta")
    g = res.grado
    dV = res.V_kepler * res.term_arm
    efecto = "aumenta" if dV > 0 else "disminuye"
    signo_P = "positivo" if res.P2n > 0 else "negativo"

    st.subheader("Lectura física del resultado")
    st.markdown(
        f"""
**Coeficiente.** El orden $n={res.n}$ corresponde al armónico zonal de grado $2n={g}$,
con $J_{{{g}}} = {sci(res.J2n)}$.

**Geometría.** A $\\varphi = {res.phi_deg:g}^\\circ$ (zona {zona}) el radio del elipsoide es
$r = {num(res.r, 3)}\\ \\mathrm{{m}}$ y el polinomio de Legendre es {signo_P}:
$P_{{{g}}}(\\cos\\theta) = {res.P2n:.6f}$.

**Potencial.** El término central (masa puntual) aporta
$GM/r = {sci(res.V_kepler, 6)}\\ \\mathrm{{m^2/s^2}}$.
El armónico {efecto} ese valor en
$\\Delta V = {sci(dV, 4)}\\ \\mathrm{{m^2/s^2}}$
(corrección relativa de ${sci(res.term_arm, 4)}$), por lo que el potencial total es
$V = {sci(res.V, 6)}\\ \\mathrm{{m^2/s^2}}$ ({res.V / 1e6:.4f} MJ/kg).
"""
    )


def render_3d(engine: GeodesyEngine, res: ResultadoCalculo, params: WGS84):
    st.subheader(f"Visualización 3D — armónico de orden {res.grado}")
    lats = np.linspace(-90, 90, 45)
    lons = np.linspace(-180, 180, 45)
    X, Y, Z, V = engine.potential_field(res.n, res.J2n, params.a, params.b, lats, lons)

    phi_rad = np.radians(res.phi_deg)
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
        legend=dict(orientation="h", y=-0.05),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("El color muestra el potencial V sobre el elipsoide; el punto rojo es la latitud evaluada.")


# =====================================================================
# 4. ORQUESTACIÓN
# =====================================================================

def main():
    render_header()
    params = WGS84()
    engine = GeodesyEngine(params)

    n, phi = render_sidebar(params)
    resultado = engine.potential(n, phi)

    col_calculo, col_3d = st.columns([1.1, 1], gap="large")
    with col_calculo:
        render_procedure(resultado, params)
        render_reading(resultado)
    with col_3d:
        render_3d(engine, resultado, params)


if __name__ == "__main__":
    main()
