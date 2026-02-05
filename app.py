import io
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Tipos de actividad (Strava)
# -----------------------------

RUN_TYPES = {
    "run",
    "running",
    "carrera",
    "trail run",
    "treadmill",
    "virtual run",
}

BIKE_TYPES = {
    "ride",
    "cycling",
    "bicicleta",
    "bike",
    "road",
    "road cycling",
    "mountain bike",
    "mtb",
    "gravel ride",
    "virtual ride",
    "e-bike ride",
    "ebike ride",
}

# -----------------------------
# Librerías de sesiones (movilidad / core / fuerza)
# -----------------------------

MOBILITY_LIBRARY = {
    "movilidad_20": [
        "Caderas: 90/90 (2x60s por lado)",
        "Tobillos: rodilla a pared (2x12 por lado)",
        "Isquios: bisagra con banda (2x10)",
        "Columna torácica: rotaciones en cuadrupedia (2x8 por lado)",
        "Glúteo medio: monster walks con minibanda (2x12 pasos por lado)",
    ],
    "movilidad_10": [
        "Tobillos: rodilla a pared (2x10 por lado)",
        "Caderas: 90/90 (1x60s por lado)",
        "Torácica: rotaciones (1x8 por lado)",
    ],
}

CORE_LIBRARY = {
    "core_20": [
        "Plancha frontal 3x40s (20s descanso)",
        "Plancha lateral 3x30s por lado",
        "Dead bug 3x10 por lado",
        "Puente glúteo 3x12",
        "Bird-dog 3x10 por lado",
    ],
    "core_10": [
        "Plancha frontal 2x40s",
        "Dead bug 2x10 por lado",
        "Puente glúteo 2x12",
    ],
}

STRENGTH_LIBRARY = {
    "fuerza_35": [
        "Sentadilla goblet 4x8 (RPE 7)",
        "Peso muerto rumano 4x8 (RPE 7)",
        "Zancadas 3x10 por lado (RPE 7)",
        "Elevación de gemelos 4x12",
        "Remo con mancuerna 3x10 por lado",
    ],
    "fuerza_25": [
        "Sentadilla goblet 3x8 (RPE 7)",
        "Peso muerto rumano 3x8 (RPE 7)",
        "Gemelos 3x12",
        "Remo 2x10 por lado",
    ],
}

GOALS = {
    "correr": [
        ("maraton", "Maratón (en ~12 meses)"),
        ("resistencia", "Mejorar resistencia y bajar pulsaciones"),
        ("media", "Media maratón"),
        ("10k", "10K"),
        ("base", "Solo base aeróbica"),
    ],
    "bicicleta": [
        ("gran_fondo", "Gran fondo (larga distancia, en ~12 meses)"),
        ("resistencia", "Mejorar resistencia y bajar pulsaciones"),
        ("puerto", "Subidas / puertos (mejorar umbral en subida)"),
        ("criterium", "Potencia y cambios de ritmo (tipo criterium)"),
        ("base", "Solo base aeróbica"),
    ],
}

# -----------------------------
# Utilidades
# -----------------------------


def next_monday(d: date) -> date:
    return d + timedelta(days=(7 - d.weekday()) % 7)


def safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def parse_time_to_minutes(val) -> Optional[float]:
    """
    Acepta:
    - segundos numéricos
    - 'hh:mm:ss' o 'mm:ss'
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (int, float)):
        # Strava a veces usa segundos
        if val > 1000:
            return float(val) / 60.0
        return float(val)
    s = str(val).strip()
    if not s:
        return None
    if ":" in s:
        parts = s.split(":")
        try:
            parts = [int(p) for p in parts]
        except Exception:
            return None
        if len(parts) == 3:
            h, m, sec = parts
            return (h * 3600 + m * 60 + sec) / 60.0
        if len(parts) == 2:
            m, sec = parts
            return (m * 60 + sec) / 60.0
    return safe_float(s)


def normalise_colname(c: str) -> str:
    return (
        c.strip()
        .lower()
        .replace("(", "")
        .replace(")", "")
        .replace("/", " ")
        .replace("-", " ")
    )


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Intenta mapear columnas típicas del export de Strava.
    """
    cols = {normalise_colname(c): c for c in df.columns}

    def pick(*candidates):
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        return ""

    return {
        "type": pick("activity type", "type"),
        "date": pick("activity date", "date", "start date", "start_time", "start time"),
        "name": pick("activity name", "name", "title"),
        "distance": pick("distance", "distance km", "distance mi"),
        "moving_time": pick("moving time", "moving_time", "duration", "elapsed time"),
        "avg_hr": pick(
            "average heart rate", "avg heart rate", "avg_hr", "averageheartrate"
        ),
        "max_hr": pick("max heart rate", "max_hr", "maximum heart rate"),
    }


def is_activity_type(val: str, allowed: set) -> bool:
    if val is None:
        return False
    v = str(val).strip().lower()
    return v in allowed


def distance_to_km(val) -> Optional[float]:
    f = safe_float(val)
    if f is None:
        return None
    # Heurística: si parece estar en metros (val > 200), convertir a km
    if f > 200:
        return f / 1000.0
    return f


@dataclass
class UserProfile:
    age: int
    height_cm: float
    weight_kg: float
    days_per_week: int
    goal: str
    modality: str  # "correr" o "bicicleta"
    start_date: date


def estimate_hrmax(age: int) -> int:
    # Estimación general (no individual). Si tienes HRmáx real, es mejor usarla.
    return int(round(208 - 0.7 * age))


def hr_zones(hrmax: int) -> Dict[str, Tuple[int, int]]:
    return {
        "Z1": (int(0.60 * hrmax), int(0.70 * hrmax)),
        "Z2": (int(0.70 * hrmax), int(0.80 * hrmax)),
        "Z3": (int(0.80 * hrmax), int(0.87 * hrmax)),
        "Z4": (int(0.87 * hrmax), int(0.93 * hrmax)),
        "Z5": (int(0.93 * hrmax), hrmax),
    }


def weekly_baseline(target_df: pd.DataFrame) -> Dict[str, float]:
    """
    Métricas base últimas 6 semanas (si hay datos).
    """
    if target_df.empty:
        return {"w_km": 0.0, "long_km": 0.0, "sessions_w": 0.0}

    df = target_df.copy()
    df["date_only"] = pd.to_datetime(df["date_only"]).dt.date

    today = date.today()
    start = today - timedelta(days=42)
    df = df[df["date_only"] >= start]
    if df.empty:
        return {"w_km": 0.0, "long_km": 0.0, "sessions_w": 0.0}

    df["week"] = pd.to_datetime(df["date_only"]).dt.to_period("W-MON")
    w = df.groupby("week")["distance_km"].sum()
    sw = df.groupby("week")["distance_km"].count()
    w_km = float(w.mean()) if len(w) else 0.0
    sessions_w = float(sw.mean()) if len(sw) else 0.0
    long_km = float(df["distance_km"].max()) if df["distance_km"].notna().any() else 0.0

    return {"w_km": w_km, "long_km": long_km, "sessions_w": sessions_w}



# -----------------------------
# Feedback semanal (métricas + texto)
# -----------------------------

def compute_weekly_summaries(target_df: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    """
    Devuelve un dataframe con una fila por semana (semanas que empiezan en lunes).
    Columnas: week_start, km, sesiones, long_km, avg_hr_mean (si existe), avg_hr_weighted (si existe).
    """
    if target_df.empty:
        return pd.DataFrame()

    df = target_df.copy()
    df["date_only"] = pd.to_datetime(df["date_only"]).dt.date
    df["week_start"] = pd.to_datetime(df["date_only"]).dt.to_period("W-MON").apply(lambda p: p.start_time.date())

    # Distancia
    if "distance_km" not in df.columns:
        df["distance_km"] = np.nan

    # FC media (si existe)
    avg_hr_col = colmap.get("avg_hr", "")
    if avg_hr_col and avg_hr_col in df.columns:
        df["avg_hr"] = pd.to_numeric(df[avg_hr_col], errors="coerce")
    else:
        df["avg_hr"] = np.nan

    # Duración (para ponderar FC media si existe)
    if "moving_min" in df.columns:
        df["moving_min"] = pd.to_numeric(df["moving_min"], errors="coerce")
    else:
        df["moving_min"] = np.nan

    g = df.groupby("week_start", dropna=False)

    out = g.agg(
        km=("distance_km", "sum"),
        sesiones=("distance_km", "count"),
        long_km=("distance_km", "max"),
        avg_hr_mean=("avg_hr", "mean"),
        moving_min_sum=("moving_min", "sum"),
    ).reset_index()

    # FC ponderada por minutos, si hay datos suficientes
    def weighted_avg_hr(sub: pd.DataFrame) -> float:
        sub = sub.copy()
        sub = sub[sub["avg_hr"].notna() & sub["moving_min"].notna() & (sub["moving_min"] > 0)]
        if sub.empty:
            return float("nan")
        return float((sub["avg_hr"] * sub["moving_min"]).sum() / sub["moving_min"].sum())

    out["avg_hr_weighted"] = g.apply(weighted_avg_hr).values

    return out.sort_values("week_start")


def generate_weekly_feedback_text(
    weekly: pd.DataFrame,
    modality: str,
    goal: str,
    hrmax: int,
) -> str:
    """
    Genera un feedback semanal basado en reglas (sin IA).
    Si más adelante configuras IA, puedes sustituir esta función por una llamada a un modelo.
    """
    if weekly.empty:
        return "No hay suficientes datos de la modalidad seleccionada para generar un feedback semanal."

    last = weekly.iloc[-1]
    prev = weekly.iloc[-2] if len(weekly) >= 2 else None

    km = float(last["km"]) if pd.notna(last["km"]) else 0.0
    sesiones = int(last["sesiones"]) if pd.notna(last["sesiones"]) else 0
    long_km = float(last["long_km"]) if pd.notna(last["long_km"]) else 0.0
    hr_w = float(last["avg_hr_weighted"]) if pd.notna(last["avg_hr_weighted"]) else None
    hr_m = float(last["avg_hr_mean"]) if pd.notna(last["avg_hr_mean"]) else None

    # Comparativas
    km_change = None
    if prev is not None and pd.notna(prev["km"]) and float(prev["km"]) > 0:
        km_change = (km / float(prev["km"]) - 1.0) * 100.0

    # Hechos
    modality_name = "Correr" if modality == "correr" else "Bicicleta"
    goal_label = dict(GOALS[modality]).get(goal, goal)
    z2 = hr_zones(hrmax)["Z2"]

    facts = []
    facts.append(f"- Modalidad: **{modality_name}**")
    facts.append(f"- Objetivo: **{goal_label}**")
    facts.append(f"- Semana analizada (inicio lunes): **{last['week_start']}**")
    facts.append(f"- Volumen: **{km:.1f} km** en **{sesiones}** sesiones")
    facts.append(f"- Sesión más larga: **{long_km:.1f} km**")

    if hr_w is not None:
        facts.append(f"- FC media (ponderada por tiempo, si disponible): **{hr_w:.0f} ppm** (Z2 aprox {z2[0]}–{z2[1]} ppm)")
    elif hr_m is not None:
        facts.append(f"- FC media (si disponible): **{hr_m:.0f} ppm** (Z2 aprox {z2[0]}–{z2[1]} ppm)")

    if km_change is not None:
        sign = "+" if km_change >= 0 else ""
        facts.append(f"- Cambio de volumen vs. semana anterior: **{sign}{km_change:.0f}%**")

    # Interpretación (reglas simples)
    alerts = []
    positives = []
    actions = []

    # Volumen y progresión
    if km_change is not None and km_change > 12:
        alerts.append("Has subido el volumen bastante respecto a la semana anterior. Mantén la próxima semana más estable para reducir riesgo de sobrecarga.")
        actions.append("Reduce el volumen de la próxima semana un 5–10% o repite el volumen actual antes de volver a subir.")
    elif km_change is not None and km_change < -20:
        alerts.append("La carga ha bajado mucho vs. la semana anterior. Si ha sido por fatiga/enfermedad, bien; si no, intenta recuperar consistencia.")
        actions.append("Vuelve gradualmente al volumen habitual (no lo recuperes todo de golpe).")
    else:
        positives.append("La progresión de volumen parece razonable o estable.")
        actions.append("Mantén la progresión suave (subidas pequeñas y semana de descarga cada 4 semanas).")

    # Frecuencia
    if sesiones <= 2:
        alerts.append("Con 2 o menos sesiones semanales es difícil mejorar de forma consistente.")
        actions.append("Si puedes, sube a 3 sesiones/semana (aunque sean cortas) para ganar continuidad.")
    elif sesiones >= 6:
        positives.append("Muy buena frecuencia semanal.")
        actions.append("Asegura al menos 1 día muy suave o descanso real para absorber la carga.")
    else:
        positives.append("Frecuencia semanal adecuada para progresar.")

    # FC (muy aproximado)
    if hr_w is not None:
        if hr_w > z2[1] + 5:
            alerts.append("Tu FC media ponderada está por encima de Z2: es probable que muchas sesiones estén siendo demasiado intensas para una base aeróbica.")
            actions.append("En rodajes fáciles, baja el ritmo/potencia hasta mantenerte en Z2 la mayor parte del tiempo.")
        else:
            positives.append("La FC media ponderada está alineada con un trabajo aeróbico (Z2) en conjunto.")
    # Tirada larga / salida larga
    if modality == "correr":
        if goal == "maraton" and long_km < 12 and km > 0:
            actions.append("Prioriza una tirada larga semanal (progresiva) y un día fácil extra si encaja en tu agenda.")
    else:
        if goal == "gran_fondo" and long_km < 60 and km > 0:
            actions.append("Para gran fondo, añade una salida larga semanal y practica nutrición/hidratación.")

    # Construir texto en Markdown
    text = []
    text.append("### Feedback semanal")
    text.append("")
    text.append("#### Hechos de la semana")
    text.extend(facts)
    text.append("")
    if positives:
        text.append("#### Lo que va bien")
        for p in positives[:5]:
            text.append(f"- {p}")
        text.append("")
    if alerts:
        text.append("#### Alertas o ajustes recomendados")
        for a in alerts[:5]:
            text.append(f"- {a}")
        text.append("")
    if actions:
        text.append("#### Próximos pasos")
        # Lista numerada
        for i, act in enumerate(actions[:5], start=1):
            text.append(f"{i}. {act}")

    return "\n".join(text)
# -----------------------------
# Generación del plan
# -----------------------------


def phase_for_week(week_idx: int) -> str:
    """
    52 semanas:
    - Base: 1-16
    - Construcción: 17-32
    - Específico: 33-46
    - Afinado: 47-52
    """
    if week_idx <= 16:
        return "Base"
    if week_idx <= 32:
        return "Construcción"
    if week_idx <= 46:
        return "Específico"
    return "Afinado"


def weekly_volume_target(profile: UserProfile, week_idx: int, baseline_w: float) -> float:
    deload = (week_idx % 4 == 0)
    growth = 1.0 + 0.012 * week_idx

    if profile.modality == "correr":
        base = max(18.0, baseline_w) if baseline_w > 0 else 22.0
        if profile.goal == "maraton":
            cap = 70.0
        elif profile.goal == "media":
            cap = 55.0
        elif profile.goal == "10k":
            cap = 45.0
        else:
            cap = 45.0
    else:
        # Bici: km/semana suelen ser mayores (aprox). Si quieres hacerlo por horas, es una mejora fácil.
        base = max(60.0, baseline_w) if baseline_w > 0 else 90.0
        if profile.goal == "gran_fondo":
            cap = 300.0
        elif profile.goal == "puerto":
            cap = 240.0
        elif profile.goal == "criterium":
            cap = 220.0
        else:
            cap = 200.0

    target = min(base * growth, cap)
    if deload:
        target *= 0.78
    return round(target, 1)


def long_session_target(profile: UserProfile, week_idx: int, baseline_long: float) -> float:
    deload = (week_idx % 4 == 0)
    growth = 1.0 + 0.015 * week_idx

    if profile.modality == "correr":
        base = max(8.0, baseline_long * 0.85) if baseline_long > 0 else 10.0
        if profile.goal == "maraton":
            cap = 32.0
        elif profile.goal == "media":
            cap = 24.0
        elif profile.goal == "10k":
            cap = 18.0
        else:
            cap = 20.0
    else:
        base = max(25.0, baseline_long * 0.85) if baseline_long > 0 else 40.0
        if profile.goal == "gran_fondo":
            cap = 140.0
        elif profile.goal == "puerto":
            cap = 110.0
        elif profile.goal == "criterium":
            cap = 90.0
        else:
            cap = 100.0

    target = min(base * growth, cap)
    if deload:
        target *= 0.75
    return round(target, 1)


def workout_templates(phase: str, goal: str, hrmax: int, modality: str) -> Dict[str, Dict]:
    z = hr_zones(hrmax)

    if modality == "bicicleta":
        # En bici, el RPE (sensación) suele ser más estable que FC en intervalos cortos.
        return {
            "easy": {
                "title": "Rodaje Z2 (bici)",
                "details": f"Z2 aprox {z['Z2'][0]}–{z['Z2'][1]} ppm. Cadencia cómoda. RPE 3–4/10.",
            },
            "tempo": {
                "title": "Sweet spot (bici)",
                "details": "Calentar 15'. Luego 3x10' fuerte pero sostenible (RPE 7/10) con 5' suave. Enfriar 10'.",
            },
            "intervals": {
                "title": "VO2 (bici)",
                "details": "Calentar 15'. Luego 6x3' muy fuerte (RPE 8–9/10) con 3' suave. Enfriar 10'.",
            },
            "progressive": {
                "title": "Progresivo (bici)",
                "details": "60–90' empezando fácil y acabando 15–20' a ritmo sostenido (RPE 7/10).",
            },
            "long": {
                "title": "Salida larga (bici)",
                "details": "Mayormente Z2. Practica hidratación y comida cada 20–30'.",
            },
            "recovery": {
                "title": "Recuperación (bici)",
                "details": "30–45' muy suave (Z1–Z2) + movilidad 10'.",
            },
            "strength": {
                "title": "Fuerza + core",
                "details": "Fuerza total (pierna + tronco) y core. Cargas moderadas, técnica limpia.",
            },
        }

    # Correr
    return {
        "easy": {
            "title": "Rodaje suave",
            "details": f"Z2 (aprox {z['Z2'][0]}–{z['Z2'][1]} ppm). Ritmo cómodo, respiración controlada.",
        },
        "tempo": {
            "title": "Tempo",
            "details": f"Calentamiento 15'. Luego 3x8' en Z3 (aprox {z['Z3'][0]}–{z['Z3'][1]} ppm) con 3' suave. Enfriar 10'.",
        },
        "intervals": {
            "title": "Intervalos",
            "details": f"Calentamiento 15'. Luego 6x3' en Z4 (aprox {z['Z4'][0]}–{z['Z4'][1]} ppm) con 2' suave. Enfriar 10'.",
        },
        "progressive": {
            "title": "Progresivo",
            "details": "45–60' empezando en Z2 y acabando los últimos 10–15' en Z3.",
        },
        "long": {
            "title": "Tirada larga",
            "details": "Mayormente Z2. Si fase Específico y objetivo maratón: últimos 20' en Z3 si te encuentras bien.",
        },
        "recovery": {
            "title": "Recuperación",
            "details": "30–40' muy suave en Z1–Z2 + movilidad 10'.",
        },
        "strength": {
            "title": "Fuerza + core",
            "details": "Fuerza total (pierna + tronco) y core. Cargas moderadas, técnica limpia.",
        },
    }


def distribute_week_km(total_km: float, long_km: float, days: int) -> List[float]:
    """
    Reparte km en sesiones (rodajes + calidad + larga).
    """
    days = max(3, int(days))
    remain = max(0.0, total_km - long_km)

    # 1 calidad, 1 tempo/progresivo, resto suaves + larga
    if days == 3:
        parts = [remain * 0.45, remain * 0.55]
    elif days == 4:
        parts = [remain * 0.25, remain * 0.30, remain * 0.45]
    elif days == 5:
        parts = [remain * 0.18, remain * 0.20, remain * 0.25, remain * 0.37]
    else:
        parts = [remain * 0.12, remain * 0.15, remain * 0.18, remain * 0.20, remain * 0.35]

    km_sessions = [round(x, 1) for x in parts] + [round(long_km, 1)]
    return km_sessions


def build_plan(profile: UserProfile, baseline: Dict[str, float], hrmax: int) -> pd.DataFrame:
    start = next_monday(profile.start_date)
    rows = []

    for w in range(1, 53):
        phase = phase_for_week(w)

        w_km = weekly_volume_target(profile, w, baseline["w_km"])
        l_km = long_session_target(profile, w, baseline["long_km"])

        # Ajuste: si long > 45% del total, subimos total un poco
        if l_km > 0.45 * w_km:
            w_km = round(l_km / 0.45, 1)

        km_sessions = distribute_week_km(w_km, l_km, profile.days_per_week)
        templates = workout_templates(phase, profile.goal, hrmax, profile.modality)

        week_start = start + timedelta(weeks=w - 1)

        # Patrón semanal (L a D)
        # 6 días: L suave, M calidad, X suave, J tempo/progresivo, V recovery, S suave, D larga
        # 5 días: L suave, M calidad, J tempo/progresivo, S suave, D larga (+ fuerza 2 días)
        # 4 días: M calidad, J suave, S tempo/progresivo, D larga (+ fuerza 2 días)
        # 3 días: M calidad, J suave, D larga (+ fuerza 2 días)
        dmap = {}

        if profile.days_per_week >= 6:
            dmap = {
                0: ("easy", km_sessions[0]),
                1: ("intervals" if phase in ("Construcción", "Específico") else "progressive", km_sessions[1]),
                2: ("easy", km_sessions[2]),
                3: ("tempo" if phase != "Base" else "progressive", km_sessions[3]),
                4: ("recovery", max(4.0, km_sessions[4])),
                5: ("easy", km_sessions[5] if len(km_sessions) > 5 else max(6.0, (w_km - l_km) * 0.2)),
                6: ("long", km_sessions[-1]),
            }
            strength_days = [2, 4]  # X y V
        elif profile.days_per_week == 5:
            dmap = {
                0: ("easy", km_sessions[0]),
                1: ("intervals" if phase in ("Construcción", "Específico") else "progressive", km_sessions[1]),
                3: ("tempo" if phase != "Base" else "progressive", km_sessions[2]),
                5: ("easy", km_sessions[3]),
                6: ("long", km_sessions[-1]),
            }
            strength_days = [2, 4]  # X y V (sin cardio principal)
        elif profile.days_per_week == 4:
            dmap = {
                1: ("intervals" if phase in ("Construcción", "Específico") else "progressive", km_sessions[0]),
                3: ("easy", km_sessions[1]),
                5: ("tempo" if phase != "Base" else "progressive", km_sessions[2]),
                6: ("long", km_sessions[-1]),
            }
            strength_days = [0, 2]  # L y X
        else:  # 3
            dmap = {
                1: ("intervals" if phase in ("Construcción", "Específico") else "progressive", km_sessions[0]),
                3: ("easy", km_sessions[1]),
                6: ("long", km_sessions[-1]),
            }
            strength_days = [0, 4]  # L y V

        for dow in range(7):
            this_day = week_start + timedelta(days=dow)
            day_name = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"][dow]

            if dow in dmap:
                wtype, km = dmap[dow]
                t = templates[wtype]

                extra = ""
                if wtype == "recovery":
                    extra = "Movilidad 10': " + "; ".join(MOBILITY_LIBRARY["movilidad_10"])
                if wtype in ("easy", "long") and phase == "Base" and (w % 2 == 1):
                    extra = (extra + " | " if extra else "") + "Core 10': " + "; ".join(CORE_LIBRARY["core_10"])

                rows.append(
                    {
                        "Fecha": this_day,
                        "Día": day_name,
                        "Modalidad": "Correr" if profile.modality == "correr" else "Bicicleta",
                        "Fase": phase,
                        "Sesión": t["title"],
                        "Volumen (km)": float(km),
                        "Detalles": t["details"] + ((" | " + extra) if extra else ""),
                    }
                )
            elif dow in strength_days:
                strength_key = "fuerza_35" if phase in ("Construcción", "Específico") else "fuerza_25"
                rows.append(
                    {
                        "Fecha": this_day,
                        "Día": day_name,
                        "Modalidad": "Correr" if profile.modality == "correr" else "Bicicleta",
                        "Fase": phase,
                        "Sesión": "Fuerza + core",
                        "Volumen (km)": 0.0,
                        "Detalles": "Fuerza: "
                        + "; ".join(STRENGTH_LIBRARY[strength_key])
                        + " | Core: "
                        + "; ".join(CORE_LIBRARY["core_20"]),
                    }
                )
            else:
                rows.append(
                    {
                        "Fecha": this_day,
                        "Día": day_name,
                        "Modalidad": "Correr" if profile.modality == "correr" else "Bicicleta",
                        "Fase": phase,
                        "Sesión": "Descanso / movilidad",
                        "Volumen (km)": 0.0,
                        "Detalles": "Movilidad 20': " + "; ".join(MOBILITY_LIBRARY["movilidad_20"]),
                    }
                )

    return pd.DataFrame(rows)


def plan_to_excel_bytes(plan: pd.DataFrame) -> bytes:
    plan = plan.copy()
    plan["Fecha"] = pd.to_datetime(plan["Fecha"])
    plan["Año"] = plan["Fecha"].dt.year
    plan["Mes"] = plan["Fecha"].dt.month

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for (y, m), chunk in plan.groupby(["Año", "Mes"], sort=True):
            month_name = chunk["Fecha"].dt.strftime("%B").iloc[0]
            sheet_name = f"{month_name[:28]}"  # límite Excel
            chunk = chunk.sort_values("Fecha")[
                ["Fecha", "Día", "Modalidad", "Fase", "Sesión", "Volumen (km)", "Detalles"]
            ]
            chunk.to_excel(writer, index=False, sheet_name=sheet_name)

            ws = writer.book[sheet_name]
            ws.freeze_panes = "A2"

            widths = [12, 12, 12, 14, 22, 12, 80]
            for i, w in enumerate(widths, start=1):
                col_letter = chr(64 + i)
                ws.column_dimensions[col_letter].width = w

    return output.getvalue()


# -----------------------------
# App Streamlit
# -----------------------------

st.set_page_config(page_title="Planificador Strava", layout="wide")
st.title("Planificador de entrenamiento Strava")

st.info(
    "Sube tu CSV de Strava y genera un plan anual en Excel (una pestaña por mes) con sesiones detalladas."
)
with st.expander("¿No sabes cómo descargar tu CSV de Strava?"):
    st.markdown(
        """
        1. Inicia sesión en Strava desde un ordenador y haz clic en tu foto de perfil.  
           Selecciona **Ajustes**.

        2. En el menú lateral izquierdo, entra en **Mi cuenta**.

        3. Accede a **Descarga o elimina tu cuenta** y pulsa en **Solicita tu archivo**.

        4. Strava te enviará un correo electrónico con un archivo comprimido (.zip).  
           Descárgalo y descomprímelo.

        5. Dentro encontrarás varios archivos.  
           **Solo necesitas el archivo `activities.csv`.**
        """
    )

uploaded = st.file_uploader("Sube tu CSV exportado de Strava (activities.csv)", type=["csv"])

with st.sidebar:
    st.header("Modalidad")
    modality = st.selectbox("Selecciona modalidad", options=["correr", "bicicleta"])

    st.header("Perfil")
    age = st.number_input("Edad", min_value=12, max_value=90, value=36, step=1)
    height_cm = st.number_input("Altura (cm)", min_value=120, max_value=220, value=180, step=1)
    weight_kg = st.number_input("Peso (kg)", min_value=35.0, max_value=200.0, value=80.0, step=0.5)

    st.header("Objetivo")
    goal = st.selectbox(
        "Selecciona objetivo principal",
        options=GOALS[modality],
        format_func=lambda x: x[1],
    )[0]

    days_per_week = st.slider("Días de entrenamiento por semana", min_value=3, max_value=6, value=5)
    start_date = st.date_input("Fecha de inicio", value=date.today())

if uploaded is None:
    st.stop()

df_raw = pd.read_csv(uploaded)
colmap = detect_columns(df_raw)

missing = [k for k, v in colmap.items() if v == "" and k in ("type", "date", "distance")]
if missing:
    st.error(
        "No he podido identificar columnas clave en el CSV. "
        f"Faltan: {', '.join(missing)}. "
        "Prueba con el export estándar de Strava (activities.csv)."
    )
    st.stop()

df = df_raw.copy()

# Parse fecha
date_col = colmap["date"]
df["date_only"] = pd.to_datetime(df[date_col], errors="coerce")
df = df[df["date_only"].notna()]

# Filtrar por modalidad
type_col = colmap["type"]
allowed = RUN_TYPES if modality == "correr" else BIKE_TYPES
df["is_target"] = df[type_col].apply(lambda v: is_activity_type(v, allowed))
target_df = df[df["is_target"]].copy()

if target_df.empty:
    st.warning(
        "No he encontrado actividades de la modalidad seleccionada en el CSV. "
        "Generaré un plan base igualmente, pero la estimación inicial será menos precisa."
    )

# Distancia y tiempo
dist_col = colmap["distance"]
if not target_df.empty:
    target_df["distance_km"] = target_df[dist_col].apply(distance_to_km)
else:
    target_df["distance_km"] = np.nan

moving_col = colmap["moving_time"]
if moving_col:
    target_df["moving_min"] = target_df[moving_col].apply(parse_time_to_minutes)
else:
    target_df["moving_min"] = np.nan

baseline = weekly_baseline(target_df) if not target_df.empty else {"w_km": 0.0, "long_km": 0.0, "sessions_w": 0.0}
hrmax = estimate_hrmax(int(age))

st.subheader("Lectura rápida del CSV")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Actividades detectadas (modalidad)", int(target_df.shape[0]))
c2.metric("Km/semana (media ~6 semanas)", f"{baseline['w_km']:.1f}")
c3.metric("Sesión más larga aprox", f"{baseline['long_km']:.1f} km")
c4.metric("HRmáx estimada (aprox)", f"{hrmax} ppm")
st.caption("Si conoces tu HRmáx real, puedes modificar la función de estimación para usar tu valor.")


# Feedback semanal (botón)
weekly = compute_weekly_summaries(target_df, colmap)
if st.button("Generar feedback semanal"):
    feedback_md = generate_weekly_feedback_text(
        weekly=weekly,
        modality=modality,
        goal=goal,
        hrmax=hrmax,
    )
    st.markdown(feedback_md)
    st.divider()

profile = UserProfile(
    age=int(age),
    height_cm=float(height_cm),
    weight_kg=float(weight_kg),
    days_per_week=int(days_per_week),
    goal=str(goal),
    modality=str(modality),
    start_date=start_date,
)

plan = build_plan(profile, baseline, hrmax)

st.subheader("Vista previa del plan (primeras 21 filas)")
st.dataframe(plan.head(21), use_container_width=True)

excel_bytes = plan_to_excel_bytes(plan)

st.download_button(
    label="Descargar plan anual en Excel",
    data=excel_bytes,
    file_name=f"plan_entrenamiento_anual_{modality}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
