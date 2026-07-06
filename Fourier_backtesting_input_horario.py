# =============================================================================
# BACKTESTING: MODELO HÍBRIDO FOURIER v2.0 (RIDGE + XGBOOST) — INPUT: PRECIOS HORARIOS
# La regresión Ridge se ajusta sobre los %PRECIOS horarios crudos (celdas
# mes x hora x fin de semana, igual que el modelo híbrido original), NO sobre
# los coeficientes de Fourier. Es decir:
#
#   1) Ridge por celda (mes, hora, fin_semana) sobre el %precio horario -> trend(t)
#   2) residual(t) = %precio(t) - trend(t)                                  (24 valores/día)
#   3) Se descompone en Fourier (4 armónicos, fase fija por mes) el RESIDUAL,
#      no la curva cruda -> A0_res..A4_res (5 coeficientes/día)
#   4) XGBoost predice esos 5 coeficientes del residual (input = estadísticos
#      de la curva horaria de %precios pasada)
#   5) Reconstrucción final: %precio_pred(t) = trend(t) + residual_pred(t)
#
# Así, XGBoost sigue prediciendo sólo 5 variables/día en vez de 24, pero el
# Ridge opera directamente sobre la escala de %precios (como en el modelo
# original), y el Fourier se aplica sólo a lo que Ridge no explica.
#
# INPUT de XGBoost: estadísticos de la curva horaria de precios pasada, EN LA
# MISMA ESCALA que el target (% sobre la media mensual) — día anterior / mismo
# día de la semana anterior — tal como en el diagrama (t, x_n) -> XGBoost ->
# Â_k,t de la presentación.
#
# Predicción a ciegas de TODO 2025 (walk-forward): el modelo se entrena sólo
# con 2019-2024 y luego predice los 365 días de 2025 en orden cronológico. En
# ningún momento se usan precios reales de 2025: el input horario de cada día
# (lag 1d / lag 7d) se construye a partir de la propia curva RECONSTRUIDA
# (predicha) de días anteriores del propio walk-forward, arrancando sólo del
# histórico real de 2019-2024. Los precios reales de 2025 sólo se usan al
# final, para calcular las métricas de error.
#
# Target: % sobre la media mensual | Test out-of-sample: Año 2025
# Métricas: MAE, RMSE, WAPE, rMAE, rRMSE (según Weron, 2014) sobre la curva
# horaria reconstruida, para ser comparables con el modelo híbrido original.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

N_ARM      = 4     # nº de armónicos (4 según la presentación)
HORAS_DIA  = 24

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA Y PREPARACIÓN DE DATOS HORARIOS
# ─────────────────────────────────────────────────────────────────────────────
def cargar_y_preparar_datos(ruta_csv: str) -> pd.DataFrame:
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"¡ERROR! No encuentro el archivo '{ruta_csv}'.")

    df = pd.read_csv(ruta_csv)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df[df['Fecha'].dt.year >= 2019]
    df = df.drop_duplicates(subset=['Fecha', 'hora'], keep='first')
    df = df.sort_values(['Fecha', 'hora']).reset_index(drop=True)

    df['mes']           = df['Fecha'].dt.month
    df['dia_semana']    = df['Fecha'].dt.dayofweek
    df['es_fin_semana'] = (df['dia_semana'] >= 5).astype(int)

    df['precio_medio_mensual'] = df.groupby(['Año', 'mes'])['Precios'].transform('mean')
    df['target'] = df['Precios'] / df['precio_medio_mensual'] * 100

    df['timestamp'] = df['Fecha'] + pd.to_timedelta(df['hora'] - 1, unit='h')
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Naive horario (valor de hace exactamente 1 semana) para la comparación final
    df_naive = df[['timestamp', 'target']].copy()
    df_naive['timestamp'] = df_naive['timestamp'] + pd.Timedelta(weeks=1)
    df_naive = df_naive.rename(columns={'target': 'target_naive'})
    df = df.merge(df_naive, on='timestamp', how='left')

    return df


def construir_pivot_diario(df: pd.DataFrame, columna: str = 'target') -> pd.DataFrame:
    """Matriz días x horas (1..24) de la columna indicada (por defecto, el
    target crudo; también se usa con 'residual'). Descarta los días con
    cambio de hora (DST, 22/23/25 valores) para poder ajustar Fourier con
    N=24 fijo."""
    df = df.drop_duplicates(subset=['Fecha', 'hora'], keep='first')
    pivot = df.pivot(index='Fecha', columns='hora', values=columna)
    pivot = pivot[sorted(pivot.columns)]
    pivot = pivot.dropna(axis=0, how='any')
    return pivot


# ─────────────────────────────────────────────────────────────────────────────
# 2. RIDGE SOBRE LOS %PRECIOS HORARIOS (mismo esquema que el modelo original)
# ─────────────────────────────────────────────────────────────────────────────
RIDGE_ALPHA_HORARIO   = 1000
MIN_OBS_CELDA_HORARIO = 20

class TendenciaHoraria:
    """Ridge por celda (mes, hora, fin_semana) sobre el %precio horario crudo,
    tal cual el modelo híbrido original: captura la tendencia estructural
    multianual de la curva de pato en cada hora del día. Todo lo que esta
    tendencia NO explica (el residual) es lo que luego se comprime en 5
    coeficientes de Fourier para que XGBoost lo prediga."""
    def __init__(self):
        self.modelos = {}

    def _ajustar_celda(self, X_años: np.ndarray, y: np.ndarray, hora: int):
        X_norm = X_años - 2019
        m_cuad = make_pipeline(PolynomialFeatures(2), Ridge(alpha=RIDGE_ALPHA_HORARIO))
        m_cuad.fit(X_norm, y)

        p2025 = m_cuad.predict([[2025 - 2019]])[0]
        p2030 = m_cuad.predict([[2030 - 2019]])[0]

        if (p2030 > p2025) and (9 <= hora <= 17):
            m_lin = make_pipeline(PolynomialFeatures(1), Ridge(alpha=RIDGE_ALPHA_HORARIO))
            m_lin.fit(X_norm, y)
            return m_lin
        return m_cuad

    def fit(self, df_hora_train: pd.DataFrame):
        print("  -> Ajustando Ridge sobre el %precio horario (576 celdas mes x hora x fin de semana)...")
        for m in range(1, 13):
            for h in range(1, HORAS_DIA + 1):
                for f in [0, 1]:
                    mask = ((df_hora_train['mes'] == m) &
                            (df_hora_train['hora'] == h) &
                            (df_hora_train['es_fin_semana'] == f))
                    if mask.sum() < MIN_OBS_CELDA_HORARIO:
                        continue
                    X_lr = df_hora_train.loc[mask, ['Año']].values
                    y_lr = df_hora_train.loc[mask, 'target'].values
                    self.modelos[(m, h, f)] = self._ajustar_celda(X_lr, y_lr, h)

    def predict(self, df_hora: pd.DataFrame) -> np.ndarray:
        """Predicción vectorizada para un DataFrame horario con columnas
        Año, mes, hora, es_fin_semana."""
        trend_pred = np.full(len(df_hora), 100.0)
        for (m, h, f), modelo in self.modelos.items():
            mask = ((df_hora['mes'] == m) & (df_hora['hora'] == h) & (df_hora['es_fin_semana'] == f))
            if mask.any():
                trend_pred[mask.values] = modelo.predict(df_hora.loc[mask, ['Año']].values - 2019)
        return trend_pred

    def predict_dia(self, año: int, mes: int, es_fin_semana: int) -> np.ndarray:
        """Devuelve las 24 predicciones de tendencia (h=1..24) para un
        año/mes/tipo de día concretos, sin necesidad de construir un
        DataFrame (usado en la predicción recursiva día a día)."""
        trend = np.full(HORAS_DIA, 100.0)
        for h in range(1, HORAS_DIA + 1):
            clave = (mes, h, es_fin_semana)
            if clave in self.modelos:
                trend[h - 1] = self.modelos[clave].predict([[año - 2019]])[0]
        return trend


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRANSFORMADA DE FOURIER: FASE LIBRE (diagnóstico) Y FASE FIJA (target)
#    — aplicada sobre el RESIDUAL (tras restar la tendencia Ridge horaria) —
# ─────────────────────────────────────────────────────────────────────────────
def fft_fase_libre_matriz(Y: np.ndarray):
    """DTF por día (fase libre). Y: (n_dias, 24). Devuelve A0, amplitudes y
    fases libres para k=1..N_ARM, según X_k = sum_n x_n e^(-i 2pi k n/N)."""
    X = np.fft.rfft(Y, n=HORAS_DIA, axis=1)
    A0    = X[:, 0].real / HORAS_DIA
    amps  = 2 * np.abs(X[:, 1:N_ARM + 1]) / HORAS_DIA
    fases = np.angle(X[:, 1:N_ARM + 1])
    return A0, amps, fases


def calcular_tabla_fases_por_mes(pivot_train: pd.DataFrame) -> np.ndarray:
    """Fase constante por mes (1..12) y armónico (1..N_ARM), como media circular
    de las fases libres diarias observadas en entrenamiento (<=2024)."""
    meses = pivot_train.index.month.values
    _, _, fases = fft_fase_libre_matriz(pivot_train.values)

    tabla = np.zeros((12, N_ARM))
    print("\n  -> Fases phi_k del RESIDUAL fijadas por mes (grados), media circular 2019-2024:")
    for m in range(1, 13):
        mask = meses == m
        tabla[m - 1] = np.angle(np.mean(np.exp(1j * fases[mask]), axis=0))
        dispersion = np.degrees(np.std(np.angle(np.exp(1j * (fases[mask] - tabla[m - 1]))), axis=0))
        print(f"     Mes {m:2d}: phi(deg) = {np.round(np.degrees(tabla[m-1]), 1)}   "
              f"(dispersión intramensual ±{np.round(dispersion, 1)}°, n={mask.sum()} días)")
    return tabla


def matriz_diseno(t: np.ndarray, fases: np.ndarray) -> np.ndarray:
    cols = [np.ones_like(t, dtype=float)]
    for k in range(1, N_ARM + 1):
        cols.append(np.cos(2 * np.pi * k * t / HORAS_DIA + fases[k - 1]))
    return np.column_stack(cols)


def ajustar_amplitudes_fase_fija(pivot: pd.DataFrame, tabla_fases: np.ndarray) -> pd.DataFrame:
    """Para cada día, con las fases del mes ya fijadas, ajusta por mínimos
    cuadrados los 5 coeficientes A0..A4 (regresión lineal sobre la base de
    cosenos de fase fija) del RESIDUAL. Éstos son el nuevo target reducido
    de XGBoost."""
    t = np.arange(HORAS_DIA)
    Y = pivot.values
    meses = pivot.index.month.values
    coefs = np.zeros((len(pivot), N_ARM + 1))

    for m in range(1, 13):
        idx = np.where(meses == m)[0]
        if len(idx) == 0:
            continue
        D = matriz_diseno(t, tabla_fases[m - 1])
        pinvD = np.linalg.pinv(D)
        coefs[idx] = (pinvD @ Y[idx].T).T

    return pd.DataFrame(coefs, index=pivot.index, columns=[f'A{k}' for k in range(N_ARM + 1)])


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURES DIARIAS — INPUT: ESTADÍSTICOS DE LA CURVA HORARIA PASADA (% s/media mensual)
# ─────────────────────────────────────────────────────────────────────────────
def calcular_stats_diarias(pivot: pd.DataFrame) -> pd.DataFrame:
    """Estadísticos resumen de la curva horaria de %precios crudos de cada
    día. Sirven de input 'precios horarios' tanto para las features de
    entrenamiento como para sembrar el histórico real (pre-2025) de la
    predicción recursiva a ciegas."""
    horas_repr = [1, 9, 14, 20]  # medianoche, rampa mañana, valle solar, pico demanda
    stats = pd.DataFrame(index=pivot.index)
    stats['media_dia'] = pivot.mean(axis=1)
    stats['std_dia']   = pivot.std(axis=1)
    stats['max_dia']   = pivot.max(axis=1)
    stats['min_dia']   = pivot.min(axis=1)
    stats['hora_max']  = np.argmax(pivot.values, axis=1) + 1
    stats['hora_min']  = np.argmin(pivot.values, axis=1) + 1
    for h in horas_repr:
        stats[f'valor_h{h}'] = pivot[h].values
    return stats


def construir_dataset_diario(pivot_precios: pd.DataFrame, coefs_df_residual: pd.DataFrame) -> pd.DataFrame:
    """pivot_precios: curva horaria CRUDA (% s/media mensual), usada para las
    features de input. coefs_df_residual: A0..A4 del RESIDUAL (tras Ridge),
    usados como target reducido de XGBoost."""
    idx = pivot_precios.index
    df = pd.DataFrame({
        'Fecha':        idx,
        'Año':          idx.year,
        'mes':          idx.month,
        'dia_semana':   idx.dayofweek,
        'semana_anio':  idx.isocalendar().week.values.astype(int),
        'dia_anio':     idx.dayofyear,
    }).reset_index(drop=True)
    df['es_fin_semana'] = (df['dia_semana'] >= 5).astype(int)

    df['mes_sin']     = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos']     = np.cos(2 * np.pi * df['mes'] / 12)
    df['semana_sin']  = np.sin(2 * np.pi * df['semana_anio'] / 52)
    df['semana_cos']  = np.cos(2 * np.pi * df['semana_anio'] / 52)
    df['diaanio_sin'] = np.sin(2 * np.pi * df['dia_anio'] / 365.25)
    df['diaanio_cos'] = np.cos(2 * np.pi * df['dia_anio'] / 365.25)
    df['mes_raw']      = df['mes']
    df['diasem_raw']   = df['dia_semana']
    df['mes_x_diasem'] = df['mes'] * df['dia_semana']

    for k in range(N_ARM + 1):
        df[f'A{k}'] = coefs_df_residual[f'A{k}'].values

    # --- Estadísticos de la curva horaria CRUDA (input = precios horarios, % s/media mensual) ---
    stats = calcular_stats_diarias(pivot_precios)

    lag1 = stats.copy()
    lag1.index = lag1.index + pd.Timedelta(days=1)
    lag1 = lag1.add_prefix('lag1d_')

    lag7 = stats.copy()
    lag7.index = lag7.index + pd.Timedelta(days=7)
    lag7 = lag7.add_prefix('lag7d_')

    df = df.merge(lag1, left_on='Fecha', right_index=True, how='left')
    df = df.merge(lag7, left_on='Fecha', right_index=True, how='left')

    lag_cols = [c for c in df.columns if c.startswith(('lag1d_', 'lag7d_'))]
    df[lag_cols] = df[lag_cols].ffill()

    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    base = ['mes_sin', 'mes_cos', 'semana_sin', 'semana_cos', 'diaanio_sin', 'diaanio_cos',
            'es_fin_semana', 'mes_raw', 'diasem_raw', 'mes_x_diasem']
    lag  = [c for c in df.columns if c.startswith(('lag1d_', 'lag7d_'))]
    return base + lag


# ─────────────────────────────────────────────────────────────────────────────
# 5. XGBOOST SOBRE LOS 5 COEFICIENTES DE FOURIER DEL RESIDUAL
#    (el Ridge ya se aplicó en la sección 2, sobre los %precios horarios;
#    aquí XGBoost sólo modela lo que Ridge no explica, ya comprimido en A0..A4)
# ─────────────────────────────────────────────────────────────────────────────
class ModeloXGBoostFourier:
    def __init__(self, targets):
        self.targets  = targets
        self.baseline = {}
        self.xgb      = {}

    def fit(self, df: pd.DataFrame, feature_cols: list):
        df_fit = df.copy().dropna(subset=feature_cols + self.targets)

        for t in self.targets:
            self.baseline[t] = df_fit[t].mean()
            y_centrada = df_fit[t] - self.baseline[t]

            print(f"  -> Entrenando XGBoost sobre el coeficiente residual {t}...")
            mask_tr  = df_fit['Año'] <= 2023
            mask_val = df_fit['Año'] == 2024

            X_train, y_train = df_fit.loc[mask_tr, feature_cols].values, y_centrada.loc[mask_tr].values
            X_val, y_val     = df_fit.loc[mask_val, feature_cols].values, y_centrada.loc[mask_val].values

            modelo_xgb = xgb.XGBRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror',
                tree_method='hist', random_state=SEED, early_stopping_rounds=50,
            )
            modelo_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            self.xgb[t] = modelo_xgb

    def predict(self, df_pred: pd.DataFrame, feature_cols: list) -> np.ndarray:
        df_p = df_pred.copy()
        coefs = np.zeros((len(df_p), len(self.targets)))

        for i, t in enumerate(self.targets):
            coefs[:, i] = self.baseline[t] + self.xgb[t].predict(df_p[feature_cols].values)

        return coefs


# ─────────────────────────────────────────────────────────────────────────────
# 6. PREDICCIÓN A CIEGAS DE TODO 2025 (walk-forward, sin ver precios reales de 2025)
# ─────────────────────────────────────────────────────────────────────────────
def predecir_recursivo_2025(modelo: ModeloXGBoostFourier, tendencia_horaria: TendenciaHoraria,
                             feature_cols: list, tabla_fases_residual: np.ndarray,
                             historial_inicial: pd.DataFrame, fecha_inicio: str, fecha_fin: str) -> pd.DataFrame:
    """Genera la predicción de TODO el rango [fecha_inicio, fecha_fin] día a
    día, en orden cronológico, SIN usar en ningún momento datos reales de ese
    rango: el input 'precios horarios' de cada día (lag1d_/lag7d_) se
    construye a partir de la propia curva horaria RECONSTRUIDA (predicha) de
    días anteriores, arrancando únicamente de historial_inicial (histórico
    real anterior al rango, p.ej. <= 2024-12-31). Cada día: 1) la tendencia
    Ridge se evalúa de forma puramente extrapolativa (año, mes, hora, tipo de
    día), 2) XGBoost predice los 5 coeficientes de Fourier del residual a
    partir del histórico de %precios reconstruido, 3) se suman ambos."""
    horas_repr = [1, 9, 14, 20]
    t_horas = np.arange(HORAS_DIA)
    historial = historial_inicial.copy()

    filas_horas = []

    for fecha in pd.date_range(fecha_inicio, fecha_fin, freq='D'):
        año           = fecha.year
        mes           = fecha.month
        dia_semana    = fecha.dayofweek
        es_fin_semana = int(dia_semana >= 5)
        semana_anio   = int(fecha.isocalendar()[1])
        dia_anio      = fecha.dayofyear

        fila = {
            'Año': año, 'mes': mes, 'es_fin_semana': es_fin_semana,
            'mes_sin': np.sin(2 * np.pi * mes / 12),         'mes_cos': np.cos(2 * np.pi * mes / 12),
            'semana_sin': np.sin(2 * np.pi * semana_anio / 52), 'semana_cos': np.cos(2 * np.pi * semana_anio / 52),
            'diaanio_sin': np.sin(2 * np.pi * dia_anio / 365.25), 'diaanio_cos': np.cos(2 * np.pi * dia_anio / 365.25),
            'mes_raw': mes, 'diasem_raw': dia_semana, 'mes_x_diasem': mes * dia_semana,
        }

        fecha_ayer = fecha - pd.Timedelta(days=1)
        fecha_sem  = fecha - pd.Timedelta(days=7)
        for prefijo, f_ref in [('lag1d_', fecha_ayer), ('lag7d_', fecha_sem)]:
            fila_ref = historial.loc[f_ref] if f_ref in historial.index else historial.iloc[-1]
            for col in historial.columns:
                fila[f'{prefijo}{col}'] = fila_ref[col]

        df_fila        = pd.DataFrame([fila])
        coef_res_pred  = modelo.predict(df_fila, feature_cols)[0]

        fases      = tabla_fases_residual[mes - 1]
        D          = matriz_diseno(t_horas, fases)
        residual_hat = D @ coef_res_pred

        trend_hat = tendencia_horaria.predict_dia(año, mes, es_fin_semana)

        y_hat = np.clip(trend_hat + residual_hat, a_min=-50, a_max=None)

        historial.loc[fecha, 'media_dia'] = y_hat.mean()
        historial.loc[fecha, 'std_dia']   = y_hat.std()
        historial.loc[fecha, 'max_dia']   = y_hat.max()
        historial.loc[fecha, 'min_dia']   = y_hat.min()
        historial.loc[fecha, 'hora_max']  = np.argmax(y_hat) + 1
        historial.loc[fecha, 'hora_min']  = np.argmin(y_hat) + 1
        for h in horas_repr:
            historial.loc[fecha, f'valor_h{h}'] = y_hat[h - 1]

        for h in range(HORAS_DIA):
            filas_horas.append((fecha, h + 1, y_hat[h]))

    df_horas = pd.DataFrame(filas_horas, columns=['Fecha', 'hora', 'target_pred'])
    df_horas['timestamp'] = df_horas['Fecha'] + pd.to_timedelta(df_horas['hora'] - 1, unit='h')
    return df_horas


# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUACIÓN Y BACKTESTING (sobre la curva horaria reconstruida)
# ─────────────────────────────────────────────────────────────────────────────
def calcular_metricas(df_test: pd.DataFrame):
    df_eval = df_test.dropna(subset=['target', 'target_pred', 'target_naive'])

    y_true  = df_eval['target'].values
    y_pred  = df_eval['target_pred'].values
    y_naive = df_eval['target_naive'].values

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mae_naive  = mean_absolute_error(y_true, y_naive)
    rmse_naive = np.sqrt(mean_squared_error(y_true, y_naive))

    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

    rmae  = mae / mae_naive
    rrmse = rmse / rmse_naive

    print("\n" + "=" * 60)
    print(" RESULTADOS DEL BACKTESTING (AÑO 2025) — MODELO FOURIER (input horario)")
    print("=" * 60)
    print(f" MAE   (Error Absoluto Medio) : {mae:.2f} %")
    print(f" RMSE  (Error Cuadrático M.)  : {rmse:.2f} %")
    print(f" WAPE  (Error Pct. Ponderado) : {wape*100:.2f} %")
    print("-" * 60)
    print(f" rMAE  (vs Naive)             : {rmae:.3f}  <-- ¡Si es < 1, ganamos al Naive!")
    print(f" rRMSE (vs Naive)             : {rrmse:.3f}  <-- ¡Si es < 1, ganamos al Naive!")
    print("=" * 60)


def graficar_8760(df_test: pd.DataFrame):
    plt.figure(figsize=(20, 7))

    plt.plot(df_test['timestamp'], df_test['target'], color='black', alpha=0.5, lw=1, label='Precios Reales 2025')
    plt.plot(df_test['timestamp'], df_test['target_pred'], color='teal', alpha=0.7, lw=1,
             label='Predicción Fourier (input horario)')

    plt.title('Backtesting 2025: Modelo Fourier (input horario) — 8760 horas', fontsize=16, fontweight='bold')
    plt.ylabel('% sobre la media mensual', fontsize=12)
    plt.xlabel('Fecha', fontsize=12)

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.axhline(0, color='black', lw=1, linestyle='--')

    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('backtesting_fourier_input_horario_2025.png', dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 8. EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("[1] Cargando datos horarios y generando modelo Naive...")
    df_hora = cargar_y_preparar_datos('Dataset_Modelo_PrecioFV_Porcentaje.csv')
    df_hora_train = df_hora[df_hora['Año'] <= 2024].copy()

    print("\n[2] Ajustando Ridge sobre los %PRECIOS horarios (2019-2024)...")
    tendencia_horaria = TendenciaHoraria()
    tendencia_horaria.fit(df_hora_train)
    df_hora_train['trend']    = tendencia_horaria.predict(df_hora_train)
    df_hora_train['residual'] = df_hora_train['target'] - df_hora_train['trend']

    print("\n[3] Construyendo matrices diarias (24h) de %precios crudos y de residuales...")
    pivot_precios_train  = construir_pivot_diario(df_hora_train, columna='target')
    pivot_residual_train = construir_pivot_diario(df_hora_train, columna='residual')
    print(f"    {len(pivot_precios_train)} días completos de entrenamiento "
          f"({pivot_precios_train.index.min().date()} - {pivot_precios_train.index.max().date()})")

    print("\n[4] Calculando fases phi_k fijas por mes SOBRE EL RESIDUAL (sin fuga de datos)...")
    tabla_fases_residual = calcular_tabla_fases_por_mes(pivot_residual_train)

    print("\n[5] Proyectando el residual de cada día sobre la base de cosenos de fase fija -> A0..A4...")
    coefs_df_residual = ajustar_amplitudes_fase_fija(pivot_residual_train, tabla_fases_residual)
    print(f"    Variables por día para XGBoost: {N_ARM + 1} (A0..A{N_ARM} del residual) "
          f"en vez de {HORAS_DIA} precios horarios.")

    print("\n[6] Construyendo dataset de entrenamiento 2019-2024 (input = curva horaria de %precios pasada)...")
    df_diario_train = construir_dataset_diario(pivot_precios_train, coefs_df_residual)
    feature_cols     = get_feature_cols(df_diario_train)
    targets          = [f'A{k}' for k in range(N_ARM + 1)]

    print(f"\n[7] Entrenando XGBoost sobre los coeficientes de Fourier del residual ({len(df_diario_train)} días)...")
    modelo = ModeloXGBoostFourier(targets)
    modelo.fit(df_diario_train, feature_cols)

    print("\n[8] Prediciendo a ciegas TODO el año 2025, día a día, SIN usar precios reales de 2025...")
    historial_inicial = calcular_stats_diarias(pivot_precios_train)
    df_pred_horas = predecir_recursivo_2025(modelo, tendencia_horaria, feature_cols, tabla_fases_residual,
                                             historial_inicial, '2025-01-01', '2025-12-31')

    df_real_2025 = df_hora.loc[df_hora['Año'] == 2025, ['Fecha', 'hora', 'target', 'target_naive']]
    df_test_hora = df_pred_horas.merge(df_real_2025, on=['Fecha', 'hora'], how='left')

    print("\n[9] Calculando métricas de error sobre la curva horaria reconstruida (evaluación vs. precios reales)...")
    calcular_metricas(df_test_hora)

    print("\n[10] Generando gráfico de las 8760 horas...")
    graficar_8760(df_test_hora)


if __name__ == '__main__':
    main()
