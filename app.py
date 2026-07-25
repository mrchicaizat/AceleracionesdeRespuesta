# =====================================================
#                    Universidad Central del Ecuador 
# Nombre: Chicaiza Tuqueres Michael Ruben
# Codigo : E5 - P2
# Interfaz Grafica
# =====================================================



import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# FUNCION NEC15
# =====================================================

def f_Get_SpectrumNEC15(zona, suelo, region, Tn):

    zona_dict = {
        'I':   (0.15, 1),
        'II':  (0.25, 2),
        'III': (0.30, 3),
        'IV':  (0.35, 4),
        'V':   (0.40, 5),
        'VI':  (0.50, 6)
    }

    z, c = zona_dict[zona]

    suelo_dict = {
        'A': (1, 1),
        'B': (1, 2),
        'C': (1, 3),
        'D': (1, 4),
        'E': (1.5, 5)
    }

    r, f = suelo_dict[suelo]

    region_dict = {
        'Sierra': 2.48,
        'Esmeraldas': 2.48,
        'Galapagos': 2.48,
        'Costa': 1.80,
        'Oriente': 2.60
    }

    n = region_dict[region]

    TablaFd = np.array([
        [0.9,0.9,0.9,0.9,0.9,0.9],
        [1,1,1,1,1,1],
        [1.36,1.28,1.19,1.15,1.11,1.06],
        [1.62,1.45,1.36,1.28,1.19,1.11],
        [2.1,1.75,1.7,1.65,1.6,1.5]
    ])

    TablaFs = np.array([
        [0.75,0.75,0.75,0.75,0.75,0.75],
        [0.75,0.75,0.75,0.75,0.75,0.75],
        [0.85,0.94,1.02,1.06,1.11,1.23],
        [1.02,1.06,1.11,1.19,1.28,1.40],
        [1.5,1.6,1.7,1.8,1.9,2]
    ])

    TablaFa = np.array([
        [0.9,0.9,0.9,0.9,0.9,0.9],
        [1,1,1,1,1,1],
        [1.4,1.3,1.25,1.23,1.2,1.18],
        [1.6,1.4,1.3,1.25,1.2,1.12],
        [1.8,1.4,1.25,1.1,1.0,0.85]
    ])

    Fa = TablaFa[f-1, c-1]
    Fd = TablaFd[f-1, c-1]
    Fs = TablaFs[f-1, c-1]

    To = 0.1 * Fs * Fd / Fa
    Tc = 0.55 * Fs * Fd / Fa

    Sa = np.zeros(len(Tn))

    for i in range(len(Tn)):
        Ti = Tn[i]

        if Ti <= To:
            Sa[i] = z * Fa * (1 + (n - 1) * Ti / To)
        elif Ti <= Tc:
            Sa[i] = n * z * Fa
        else:
            Sa[i] = n * z * Fa * (Tc / Ti)**r

    Sa0 = z * Fa

    Tnf = np.concatenate(([0], Tn))
    Saf = np.concatenate(([Sa0], Sa))

    return Tnf, Saf

#ESPECTRO ASCE 7-16

def espectro_ASCE(suelo, Ss, S1, TL, Fa_manual=None, Fv_manual=None):

    suelo = suelo.upper()

    # =================================================
    # Fa
    # =================================================

    if suelo == "A":
        Fa = 0.8

    elif suelo == "B":
        Fa = 0.9

    elif suelo == "C":
        Ss_tab = np.array([0.25,0.50,0.75,1.00,1.25,1.50])
        Fa_tab = np.array([1.3,1.3,1.2,1.2,1.2,1.2])
        Fa = np.interp(Ss, Ss_tab, Fa_tab)

    elif suelo == "D":
        Ss_tab = np.array([0.25,0.50,0.75,1.00,1.25,1.50])
        Fa_tab = np.array([1.6,1.4,1.2,1.1,1.0,1.0])
        Fa = np.interp(Ss, Ss_tab, Fa_tab)

    elif suelo == "E":
        if Ss <= 0.75:
            Ss_tab = np.array([0.25,0.50,0.75])
            Fa_tab = np.array([2.4,1.7,1.3])
            Fa = np.interp(Ss, Ss_tab, Fa_tab)
        else:
            Fa = Fa_manual

    elif suelo == "F":
        Fa = Fa_manual


    # =================================================
    # Fv
    # =================================================

    if suelo in ["A","B"]:
        Fv = 0.8

    elif suelo == "C":
        S1_tab = np.array([0.10,0.20,0.30,0.40,0.50,0.60])
        Fv_tab = np.array([1.5,1.5,1.5,1.5,1.5,1.4])
        Fv = np.interp(S1, S1_tab, Fv_tab)

    elif suelo == "D":
        S1_tab = np.array([0.10,0.20,0.30,0.40,0.50,0.60])
        Fv_tab = np.array([2.4,2.2,2.0,1.9,1.8,1.7])
        Fv = np.interp(S1, S1_tab, Fv_tab)

    elif suelo == "E":
        if S1 <= 0.10:
            Fv = 4.2
        else:
            Fv = Fv_manual

    elif suelo == "F":
        Fv = Fv_manual

    # =================================================
    # PARAMETROS ESPECTRALES
    # =================================================

    Sms = Fa * Ss
    Sm1 = Fv * S1

    Sds = (2/3) * Sms
    Sd1 = (2/3) * Sm1

    Ts = Sd1 / Sds
    T0 = 0.2 * Ts

    Tmax = 10
    dT = 0.01

    T = np.arange(dT, Tmax + dT, dT)
    Sa = np.zeros(len(T))

    for i in range(len(T)):

        if T[i] <= T0:
            Sa[i] = Sds * (0.4 + 0.6 * T[i] / T0)

        elif T[i] <= Ts:
            Sa[i] = Sds

        elif T[i] <= TL:
            Sa[i] = Sd1 / T[i]

        else:
            Sa[i] = (Sd1 * TL) / (T[i]**2)

    return T, Sa, Fa, Fv, Sds, Sd1, Ts



# ==========================================================
# FUNCIONES NCh2745 (COMPLETO EN UN SOLO BLOQUE)
# ==========================================================

def cargarParametrosSueloABC(sueloABC, g):

    sueloABC = sueloABC.upper()

    if sueloABC == "A":
        P = {"Ta":0.03,"Tb":0.11,"Tc":0.29,"Td":2.51,
             "SA":1085/100,"AV":50/100,"AD":20/100,
             "A":0.40*g}

    elif sueloABC == "B":
        P = {"Ta":0.03,"Tb":0.20,"Tc":0.54,"Td":2.00,
             "SA":1100/100,"AV":94/100,"AD":30/100,
             "A":0.41*g}

    elif sueloABC == "C":
        P = {"Ta":0.03,"Tb":0.375,"Tc":0.68,"Td":1.58,
             "SA":1212/100,"AV":131/100,"AD":33/100,
             "A":0.45*g}

    else:
        raise ValueError("Suelo invalido. Use 'A', 'B' o 'C'.")

    return P


def construirEspectroBase5(T, P):

    Sa = np.zeros_like(T)

    a = (T <= P["Ta"])
    Sa[a] = P["A"]

    b = (T > P["Ta"]) & (T <= P["Tb"])
    Sa[b] = (
        P["A"] +
        (P["SA"] - P["A"]) *
        (T[b] - P["Ta"]) /
        (P["Tb"] - P["Ta"])
    )

    c = (T > P["Tb"]) & (T <= P["Tc"])
    Sa[c] = P["SA"]

    d = (T > P["Tc"]) & (T <= P["Td"])
    Sa[d] = (2*np.pi) * P["AV"] / T[d]

    e = (T > P["Td"])
    Sa[e] = (4*np.pi**2) * P["AD"] / (T[e]**2)

    return Sa


def f_get_BD(Beta):

    Tabla = np.array([
        [2, 0.65],
        [5, 1],
        [10,1.37],
        [15,1.67],
        [20,1.94],
        [25,2.17],
        [30,2.38],
        [50,3.02]
    ])

    allBeta = Tabla[:,0]
    allBd = Tabla[:,1]

    idx = np.argmin(np.abs(allBeta - Beta))
    nearestVal = allBeta[idx]

    if Beta <= 2:
        return 0.65

    elif Beta >= 50:
        return 3.02

    elif Beta == nearestVal:
        return Tabla[idx,1]

    elif Beta < nearestVal:
        x1, y1 = Tabla[idx-1]
        x2, y2 = Tabla[idx]
        return np.interp(Beta, [x1,x2], [y1,y2])

    else:
        x1, y1 = Tabla[idx]
        x2, y2 = Tabla[idx+1]
        return np.interp(Beta, [x1,x2], [y1,y2])



# =====================================================
# CONFIGURACION PAGINA
# =====================================================

st.set_page_config(page_title="Espectros de Respuesta", layout="centered")

if "pantalla" not in st.session_state:
    st.session_state.pantalla = "inicio"


# =====================================================
# PANTALLA INICIO
# =====================================================

if st.session_state.pantalla == "inicio":

    st.title("ESPECTROS DE RESPUESTA")
    st.markdown("### Seleccione el espectro a utilizar")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("NEC 15"):
            st.session_state.pantalla = "NEC"

    with col2:
        if st.button("NCh2745"):
            st.session_state.pantalla = "NCh"

    with col3:
        if st.button("ASCE 7-16"):
            st.session_state.pantalla = "ASCE"

# =====================================================
# PANTALLA NEC (PROFESIONAL)
# =====================================================

elif st.session_state.pantalla == "NEC":

    st.title("Espectro NEC 15")

    if st.button("⬅ Volver al inicio"):
        st.session_state.pantalla = "inicio"

    # -------------------------------------------
    # SIDEBAR PARAMETROS
    # -------------------------------------------

    st.sidebar.header("Parámetros NEC")

    zona = st.sidebar.selectbox("Zona Sísmica", ['I','II','III','IV','V','VI'])
    suelo = st.sidebar.selectbox("Tipo de Suelo", ['A','B','C','D','E'])
    region = st.sidebar.selectbox("Región", ['Sierra','Costa','Oriente','Esmeraldas','Galapagos'])

    hn = st.sidebar.number_input("Altura hn (m)", value=9.0)
    R = st.sidebar.number_input("R", value=8.0)
    I = st.sidebar.number_input("I", value=1.0)
    fip = st.sidebar.number_input("fip", value=1.0)
    fie = st.sidebar.number_input("fie", value=1.0)

    if st.sidebar.button("Calcular Espectro NEC"):

        Tn = np.arange(0.1, 4.01, 0.01)

        TnC, SaC = f_Get_SpectrumNEC15(zona, suelo, region, Tn)

        ct = 0.055
        alfa = 0.9
        Ta = ct * hn**alfa
        f = I / (R * fie * fip)

        # 🔹 Interpolación más estable
        SaTa = np.interp(Ta, TnC, SaC)
        C = SaTa * f


        # =========================
        # GRAFICA
        # =========================

        fig, ax = plt.subplots(figsize=(9,5))

        ax.plot(TnC, SaC, linewidth=2, label="Elástico")
        ax.plot(TnC, SaC * f, linewidth=2, label="Inelástico")

        # SIN líneas punteadas

        ax.set_xlabel("T (s)")
        ax.set_ylabel("Sa (g)")
        ax.set_title("Espectro NEC15")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

        # =========================
        # TABLA
        # =========================

        import pandas as pd
        g = 9.806

        df_nec = pd.DataFrame({
            "T (s)": TnC,
            "Sa_elástico (g)": SaC,
            "Sa_inelástico (g)": SaC * f,
            "Sa_elástico (m/s²)": SaC * g,
            "Sa_inelástico (m/s²)": SaC * g * f
        })

        with st.expander("Ver Tabla Completa"):
            st.dataframe(df_nec)

        csv_nec = df_nec.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Descargar Tabla",
            data=csv_nec,
            file_name="Espectro_NEC15.csv",
            mime="text/csv",
        )

# =====================================================
# PANTALLA ASCE (PROFESIONAL CORREGIDA)
# =====================================================

elif st.session_state.pantalla == "ASCE":

    st.title("Espectro ASCE 7-16")

    if st.button("⬅ Volver al inicio"):
        st.session_state.pantalla = "inicio"

    st.sidebar.header("Parámetros ASCE")

    suelo = st.sidebar.selectbox("Tipo de Suelo", ['A','B','C','D','E','F'])
    Ss = st.sidebar.number_input("Ss", value=0.65)
    S1 = st.sidebar.number_input("S1", value=0.35)
    TL = st.sidebar.number_input("TL", value=4.0)

    Fa_manual = None
    Fv_manual = None

    if suelo == "E" and Ss > 0.75:
        Fa_manual = st.sidebar.number_input("Fa (estudio suelo)", value=1.0)

    if suelo == "F":
        Fa_manual = st.sidebar.number_input("Fa (estudio específico)", value=1.0)
        Fv_manual = st.sidebar.number_input("Fv (estudio específico)", value=1.0)

    if suelo == "E" and S1 > 0.10:
        Fv_manual = st.sidebar.number_input("Fv (estudio suelo)", value=1.0)

    if st.sidebar.button("Calcular Espectro"):

        T, Sa, Fa, Fv, Sds, Sd1, Ts = espectro_ASCE(
            suelo, Ss, S1, TL, Fa_manual, Fv_manual
        )

        # =========================
        # RESULTADOS
        # =========================

        st.subheader("Parámetros Espectrales")

        col1, col2, col3 = st.columns(3)
        col1.metric("Fa", f"{Fa:.3f}")
        col2.metric("Fv", f"{Fv:.3f}")
        col3.metric("Sds", f"{Sds:.3f}")

        col4, col5 = st.columns(2)
        col4.metric("Sd1", f"{Sd1:.3f}")
        col5.metric("Ts (s)", f"{Ts:.3f}")

        # =========================
        # GRAFICA
        # =========================

        fig, ax = plt.subplots(figsize=(9,5))

        ax.plot(T, Sa, linewidth=2, label="Elástico")

        ax.set_xlabel("Periodo T (s)")
        ax.set_ylabel("Sa (g)")
        ax.set_title("Espectro ASCE 7-16")
        ax.legend()
        ax.grid(True)


        st.pyplot(fig)

        # =========================
        # TABLA + CSV
        # =========================

        import pandas as pd
        g = 9.806

        df = pd.DataFrame({
            "T (s)": T,
            "Sa (g)": Sa,
            "Sa (m/s²)": Sa * g
        })

        with st.expander("Ver Tabla Completa"):
            st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Descargar Tabla",
            data=csv,
            file_name="Espectro_ASCE_7_16.csv",
            mime="text/csv",
        )

#PAGINA NCH2745

elif st.session_state.pantalla == "NCh":

    st.title("Espectro NCh2745")

    if st.button("⬅ Volver al inicio"):
        st.session_state.pantalla = "inicio"

    # =========================================
    # SIDEBAR PARAMETROS
    # =========================================

    st.sidebar.header("Parámetros NCh2745")

    Beta = st.sidebar.number_input("Amortiguamiento Beta (%)", value=13.7)
    suelo = st.sidebar.selectbox("Tipo de Suelo", ["A","B","C"])
    zona = st.sidebar.selectbox("Zona Sísmica", [1,2,3])
    Tlimite_BD = st.sidebar.number_input("T límite BD (s)", value=2.0)
    paso_T = st.sidebar.number_input("Paso T", value=0.01)
    T_final = st.sidebar.number_input("Periodo máximo (s)", value=4.5)

    if st.sidebar.button("Calcular Espectro NCh"):

        g = 9.806
        T = np.arange(0, T_final + paso_T, paso_T)

        # =========================================
        # FACTOR Z
        # =========================================

        tablaZ = [0.75, 1.00, 1.25]
        Z = tablaZ[zona-1]

        # =========================================
        # PARAMETROS SUELO
        # =========================================

        P = cargarParametrosSueloABC(suelo, g)

        # =========================================
        # ESPECTRO BASE
        # =========================================

        Sa = construirEspectroBase5(T, P)
        Sa = Z * Sa

        # =========================================
        # FACTOR BD
        # =========================================

        BD = f_get_BD(Beta)

        Sa[T >= Tlimite_BD] = Sa[T >= Tlimite_BD] / BD

        # =========================================
        # RESULTADOS NUMERICOS
        # =========================================

        st.subheader("Resultados")

        col1, col2, col3 = st.columns(3)
        col1.metric("Z", f"{Z:.2f}")
        col2.metric("BD", f"{BD:.3f}")
        col3.metric("Sa_max (g)", f"{np.max(Sa)/g:.4f}")

        # =========================================
        # GRAFICA
        # =========================================

        fig, ax = plt.subplots(figsize=(9,5))
        ax.plot(T, Sa/g, linewidth=2)

        ax.set_xlabel("Periodo T (s)")
        ax.set_ylabel("Sa (g)")
        ax.set_title("Espectro NCh2745")
        ax.grid(True)

        st.pyplot(fig)

        # =========================================
        # TABLA + CSV
        # =========================================

        import pandas as pd

        df_nch = pd.DataFrame({
            "T (s)": T,
            "Sa (m/s²)": Sa,
            "Sa (g)": Sa/g
        })

        with st.expander("Ver Tabla Completa"):
            st.dataframe(df_nch, use_container_width=True)

        csv_nch = df_nch.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Descargar Tabla",
            data=csv_nch,
            file_name="Espectro_NCh2745.csv",
            mime="text/csv",
        )



