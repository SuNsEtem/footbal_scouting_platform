from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
STATS_PATH = DATA_DIR / "stats.csv"
MARKET_VALUES_PATH = DATA_DIR / "market_values.csv"

POSITIONS = ["Нападающие", "Полузащитники", "Защитники", "Вратари"]
LEAGUES = ["АПЛ", "Ла Лига", "Серия А", "Бундеслига", "Лига 1", "Legends"]

FIELD_METRIC_COLUMNS = ["xG", "KP", "PrgC", "Tkl+Int"]
GK_METRIC_COLUMNS = ["Save%", "GA90", "SoTA", "CS%"]

METRIC_LABELS = {
    "xG": "xG (Ожидаемые голы)",
    "KP": "Ключевые передачи",
    "PrgC": "Прогрессивные ведения",
    "Tkl+Int": "Отборы + перехваты",
    "Save%": "Процент сейвов",
    "GA90": "Пропущено голов за 90'",
    "SoTA": "Удары в створ соперника",
    "CS%": "Процент сухих матчей",
}

POSITION_MAP = {
    "FW": "Нападающие",
    "MF": "Полузащитники",
    "DF": "Защитники",
    "GK": "Вратари",
}

LEAGUE_MAP = {
    "eng Premier League": "АПЛ",
    "es La Liga": "Ла Лига",
    "it Serie A": "Серия А",
    "de Bundesliga": "Бундеслига",
    "fr Ligue 1": "Лига 1",
    "Legends": "Legends",
}

# Легенды с предустановленным риском травм
LEGEND_PLAYERS = [
    {
        "Player": "Roberto Carlos",
        "Club": "Real Madrid Legends",
        "League": "Legends",
        "Age": 28,
        "Height": 168,
        "Foot": "Left",
        "Pos": "DF",
        "Price": 120_000_000,
        "injury_risk": "LOW",
        "Min": 2800,
        "Starts": 32,
        "MP": 34,
        "xG": 0.15, "KP": 1.8, "PrgC": 4.2, "Tkl+Int": 3.5,
        "Save%": 0.0, "GA90": 0.0, "SoTA": 0.0, "CS%": 0.0,
        "Image URL": "https://avatars.mds.yandex.net/get-kinopoisk-image/10703859/9ef8f3ff-59c5-4e84-8b4f-d24639d2bb48/360",
    },
    {
        "Player": "Neymar",
        "Club": "Al-Hilal / Legends",
        "League": "Legends",
        "Age": 24,
        "Height": 175,
        "Foot": "Right",
        "Pos": "FW",
        "Price": 150_000_000,
        "injury_risk": "HIGH",
        "Min": 900,
        "Starts": 10,
        "MP": 12,
        "xG": 0.65, "KP": 3.1, "PrgC": 6.8, "Tkl+Int": 0.8,
        "Save%": 0.0, "GA90": 0.0, "SoTA": 0.0, "CS%": 0.0,
        "Image URL": "https://img.a.transfermarkt.technology/portrait/big/68290-1692601435.jpg?lm=1",
    },
]

OUTPUT_COLUMNS = [
    "Player", "Club", "League", "Age", "Height", "Foot", "Pos", "Позиция",
    "Price", "injury_risk", "Min", "Starts", "MP",
    *FIELD_METRIC_COLUMNS, *GK_METRIC_COLUMNS, "Image URL"
]

def _normalize_name(series: pd.Series) -> pd.Series:
    import unicodedata
    def remove_accents(text):
        text = str(text)
        return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return series.astype(str).apply(remove_accents).str.lower().str.replace(" ", "", regex=False)

def _primary_position(pos: str) -> str:
    if pd.isna(pos): return "MF"
    return str(pos).split(",")[0].strip()

def _safe_per90(numerator: pd.Series, nineties: pd.Series) -> pd.Series:
    return np.where(nineties > 0, numerator / nineties, 0.0)

def _compute_injury_risk(minutes: pd.Series, starts: pd.Series) -> pd.Series:
    mins = pd.to_numeric(minutes, errors="coerce").fillna(0)
    sts = pd.to_numeric(starts, errors="coerce").fillna(0)
    mins_per_start = np.where(sts > 0, mins / sts, 90.0)
    risk = np.where(((mins < 600) & (mins > 0)) | ((sts >= 5) & (mins_per_start < 65.0)), "HIGH", "LOW")
    return pd.Series(risk, index=minutes.index)

def _derive_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    nineties = pd.to_numeric(df["90s"], errors="coerce").fillna(0)
    
    goals = pd.to_numeric(df["Gls"], errors="coerce").fillna(0)
    assists = pd.to_numeric(df["Ast"], errors="coerce").fillna(0)
    shots_per90 = pd.to_numeric(df["Sh/90"], errors="coerce").fillna(0)
    sot_per90 = pd.to_numeric(df["SoT/90"], errors="coerce").fillna(0)
    crosses_per90 = _safe_per90(pd.to_numeric(df["Crs"], errors="coerce").fillna(0), nineties)
    tackles = pd.to_numeric(df["TklW"], errors="coerce").fillna(0)
    interceptions = pd.to_numeric(df["Int"], errors="coerce").fillna(0)

    goals_per90 = np.round(_safe_per90(goals, nineties) * 0.85, 2)
    xg = np.round(shots_per90 * 0.104, 2)
    xg = np.where((xg == 0) & (nineties > 0), goals_per90, xg)

    df["xG"] = xg
    df["KP"] = np.round(_safe_per90(assists, nineties) * 2.2 + crosses_per90 * 0.15, 2)
    df["PrgC"] = np.round((shots_per90 + sot_per90) * 1.1 + crosses_per90 * 0.35, 2)
    df["Tkl+Int"] = np.round(_safe_per90(tackles + interceptions, nineties), 2)

    df["Save%"] = pd.to_numeric(df["Save%"], errors="coerce").fillna(0.0)
    df["GA90"] = pd.to_numeric(df["GA90"], errors="coerce").fillna(0.0)
    df["SoTA"] = pd.to_numeric(df["SoTA"], errors="coerce").fillna(0.0)
    df["CS%"] = pd.to_numeric(df["CS%"], errors="coerce").fillna(0.0)

    is_gk = df["Pos"].map(_primary_position) == "GK"
    for col in FIELD_METRIC_COLUMNS:
        df.loc[is_gk, col] = 0.0
    for col in GK_METRIC_COLUMNS:
        df.loc[~is_gk, col] = 0.0

    return df

def _fill_missing_market_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    position_avg = df.dropna(subset=["Price"]).groupby("Pos")["Price"].mean().to_dict()
    global_avg = float(df["Price"].dropna().mean()) if df["Price"].notna().any() else 1_000_000.0
    df["Price"] = df.apply(lambda r: float(r["Price"]) if pd.notna(r["Price"]) else float(position_avg.get(r["Pos"], global_avg)), axis=1)
    return df

def load_players(stats_path: Path = STATS_PATH, market_path: Path = MARKET_VALUES_PATH) -> pd.DataFrame:
    stats = pd.read_csv(stats_path)
    market = pd.read_csv(market_path)

    stats["merge_key"] = _normalize_name(stats["Player"])
    market["merge_key"] = _normalize_name(market["name"])

    market_subset = market[["merge_key", "market_value_in_eur", "height_in_cm", "foot", "image_url"]].drop_duplicates(subset=["merge_key"], keep="first")
    merged = stats.merge(market_subset, on="merge_key", how="left")
    
    merged["Pos"] = merged["Pos"].map(_primary_position)
    merged = _derive_metrics(merged)

    merged["Player"] = merged["Player"].astype(str).str.strip()
    merged["Club"] = merged["Squad"].astype(str).str.strip()
    merged["League"] = merged["Comp"].map(lambda c: LEAGUE_MAP.get(str(c), str(c)))
    merged["Позиция"] = merged["Pos"].map(lambda code: POSITION_MAP.get(code, "Полузащитники"))
    
    merged["Price"] = pd.to_numeric(merged["market_value_in_eur"], errors="coerce")
    merged["Height"] = pd.to_numeric(merged["height_in_cm"], errors="coerce")
    merged["Foot"] = merged["foot"].fillna("Unknown").astype(str).str.capitalize()
    merged["Image URL"] = merged["image_url"].fillna("").astype(str)
    merged["Age"] = pd.to_numeric(merged["Age"], errors="coerce")
    merged["Min"] = pd.to_numeric(merged["Min"], errors="coerce").fillna(0).astype(int)
    merged["Starts"] = pd.to_numeric(merged["Starts"], errors="coerce").fillna(0).astype(int)
    merged["MP"] = pd.to_numeric(merged["MP"], errors="coerce").fillna(0).astype(int)
    
    merged["injury_risk"] = _compute_injury_risk(merged["Min"], merged["Starts"])

    players = merged[OUTPUT_COLUMNS].copy()
    players = _fill_missing_market_values(players)

    legends = pd.DataFrame(LEGEND_PLAYERS)
    legends["Позиция"] = legends["Pos"].map(POSITION_MAP)
    
    players = players[~players["Player"].isin(legends["Player"])]
    players = pd.concat([players, legends], ignore_index=True)

    players["Age"] = players["Age"].fillna(0).astype(int)
    players["Height"] = players["Height"].fillna(0).astype(int)
    players["Price"] = players["Price"].astype(int)
    players["value_num"] = players["Price"].astype(float)

    players.loc[players["Player"] == "Neymar", "injury_risk"] = "HIGH"

    for metric in FIELD_METRIC_COLUMNS + GK_METRIC_COLUMNS:
        players[metric] = pd.to_numeric(players[metric], errors="coerce").round(2)

    return players.sort_values(["League", "Player"]).reset_index(drop=True)

def normalize_metrics(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = df.copy()
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val == min_val:
            normalized[col] = 0.5
        else:
            if col == "GA90":
                normalized[col] = (max_val - df[col]) / (max_val - min_val)
            else:
                normalized[col] = (df[col] - min_val) / (max_val - min_val)
    return normalized

def find_similar_players_weighted(
    df: pd.DataFrame, player_name: str, top_n: int = 3, weights: list[float] | None = None
) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    target_row = df[df["Player"] == player_name].iloc[0]
    
    is_gk = (target_row["Pos"] == "GK")
    metrics = GK_METRIC_COLUMNS if is_gk else FIELD_METRIC_COLUMNS

    if weights is None:
        weights = [1.0] * len(metrics)
    weights_arr = np.array(weights)

    if np.sum(weights_arr) == 0:
        empty_res = df[df["Player"] != player_name].head(top_n).copy()
        empty_res["Схожесть"] = 0.0
        target_df = df[df["Player"] == player_name].copy()
        target_df["Схожесть"] = 100.0
        return pd.concat([target_df, empty_res], ignore_index=True)

    normalized = normalize_metrics(df, metrics)
    target_idx = df.index[df["Player"] == player_name][0]
    
    target_v = normalized.loc[target_idx, metrics].values * weights_arr

    similarities = []
    for idx, row in normalized.iterrows():
        if idx == target_idx:
            continue
        
        vec = row[metrics].values * weights_arr
        dot_product = np.dot(target_v, vec)
        norm_product = np.linalg.norm(target_v) * np.linalg.norm(vec)
        
        cos_sim = float(dot_product / (norm_product + 1e-9))
        cos_sim = max(0.0, min(1.0, cos_sim))
        
        absolute_diff = np.abs(vec - target_v)
        penalty = 1.0 / (1.0 + 0.5 * np.sum(absolute_diff))
        
        final_sim = cos_sim * penalty
        similarities.append((idx, final_sim))

    similarities.sort(key=lambda item: item[1], reverse=True)
    top_indices = [idx for idx, _ in similarities[:top_n]]
    
    target_df = df.loc[[target_idx]].copy()
    target_df["Схожесть"] = 100.0
    
    analogs_df = df.loc[top_indices].copy()
    analogs_df["Схожесть"] = [round(sim * 100, 1) for _, sim in similarities[:top_n]]
    
    return pd.concat([target_df, analogs_df], ignore_index=True)
