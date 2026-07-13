from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import unicodedata

from data import (
    DATA_DIR,
    LEAGUES,
    MARKET_VALUES_PATH,
    FIELD_METRIC_COLUMNS,
    GK_METRIC_COLUMNS,
    METRIC_LABELS,
    POSITIONS,
    STATS_PATH,
    find_similar_players_weighted,
    load_players,
    normalize_metrics,
)

st.set_page_config(
    page_title="Футбольный скаутинг",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 3.5rem; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem; }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
        background: linear-gradient(90deg, #00D4AA, #4FC3F7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle { color: #9CA3AF; margin-bottom: 2rem; }
    .player-name { font-size: 1.2rem; font-weight: 700; color: #FAFAFA; }
    .badge { padding: 3px 8px; border-radius: 5px; font-size: 0.75rem; font-weight: 600; display: inline-block; }
    .badge-high { background-color: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid #EF4444; }
    .badge-low { background-color: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid #10B981; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_data_cached(stats_file: Path, market_file: Path) -> pd.DataFrame:
    return load_players(stats_path=stats_file, market_path=market_file)

def main() -> None:
    df = load_data_cached(STATS_PATH, MARKET_VALUES_PATH)

    st.markdown('<p class="hero-title">⚽ Профессиональная платформа скаутинг-аналитики</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Система продвинутой фильтрации игроков и поиска сходств на базе FBref</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("🔍 Настройки скаутинга")
        search_query = st.text_input("Поиск игрока или клуба", placeholder="Например, Roberto Carlos...")
        
        positions = st.multiselect("Глобальная позиция", options=POSITIONS, default=POSITIONS)
        leagues = st.multiselect("Лиги", options=sorted(df["League"].unique()), default=LEAGUES)
        
        # --- НОВЫЙ ФИЛЬТР ПО РИСКУ ТРАВМ ---
        injury_options = ["LOW", "HIGH"]
        selected_injury_risks = st.multiselect("Допустимый риск травм", options=injury_options, default=injury_options)
        
        age_range = st.slider("Возраст", 15, 45, (17, 40))
        height_range = st.slider("Рост (см)", 150, 210, (165, 210))
        min_minutes = st.slider("Минимум минут на поле", 0, 3000, 500, 100)
        
        available_feet = sorted([f for f in df["Foot"].unique() if f != "Unknown"])
        selected_feet = st.multiselect("Рабочая нога", options=available_feet, default=available_feet)
        
        st.subheader("Финансовые лимиты")
        max_val_m = float(df["value_num"].max() / 1_000_000) if not df.empty else 200.0
        price_range = st.slider("Рыночный бюджет (млн €)", 0.0, max_val_m, (0.0, max_val_m), 0.5)

        st.subheader("📊 Фильтры игровой статистики")
        stat_filters = {}
        
        if len(positions) == 1 and positions[0] == "Вратари":
            stat_filters["Save%"] = st.slider("Минимальный % Сейвов", 0.0, 1.0, 0.0, 0.05)
            stat_filters["GA90"] = st.slider("Максимальный пропущенный гол/90", 0.0, 4.0, 4.0, 0.1)
        else:
            stat_filters["xG"] = st.slider("Минимум xG per 90", 0.0, 1.5, 0.0, 0.05)
            stat_filters["KP"] = st.slider("Минимум Ключевых передач", 0.0, 5.0, 0.0, 0.1)

    if not positions or not leagues or not selected_injury_risks:
        st.warning("Пожалуйста, укажите хотя бы одну позицию, лигу и уровень риска травм в левом меню.")
        return

    min_price, max_price = price_range[0] * 1_000_000, price_range[1] * 1_000_000
    
    # Применяем фильтрацию, включая новый фильтр injury_risk
    f_df = df[
        (df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1]) &
        (df["Height"] >= height_range[0]) & (df["Height"] <= height_range[1]) &
        (df["Min"] >= min_minutes) & (df["Позиция"].isin(positions)) & (df["League"].isin(leagues)) &
        (df["value_num"] >= min_price) & (df["value_num"] <= max_price) &
        (df["injury_risk"].isin(selected_injury_risks))
    ]
    if selected_feet:
        f_df = f_df[f_df["Foot"].isin(selected_feet)]
        
    for k, val in stat_filters.items():
        if k == "GA90":
            f_df = f_df[f_df[k] <= val]
        else:
            f_df = f_df[f_df[k] >= val]

    if search_query:
        q = "".join(c for c in unicodedata.normalize("NFD", str(search_query)) if unicodedata.category(c) != "Mn").lower()
        f_df = f_df[f_df.apply(lambda r: q in str(r["Player"]).lower() or q in str(r["Club"]).lower(), axis=1)]

    f_df = f_df.sort_values(["Позиция", "Player"]).reset_index(drop=True)

    tab_search, tab_compare, tab_similar = st.tabs(["🕵️‍♂️ Поиск талантов", "📊 Сравнение игроков", "🧠 Поиск аналогов"])

    def get_player_meta_str(row) -> str:
        return f"{row['Club']} | {row['League']} | {row['Age']} лет | {row['Позиция']} ({row['Pos']})"

    # ВКЛАДКА 1: ПОИСК ТАЛАНТОВ
    with tab_search:
        st.caption(f"Найдено игроков по заданным критериям: {len(f_df)}")
        view_mode = st.radio("Формат данных:", ["Карточки", "Таблица"], horizontal=True, label_visibility="collapsed")

        if view_mode == "Таблица":
            metric_cols = GK_METRIC_COLUMNS if (len(positions) == 1 and positions[0] == "Вратари") else FIELD_METRIC_COLUMNS
            cols_to_show = ["Player", "Club", "Age", "Height", "Foot", "Позиция", "League", "Price", "injury_risk", "Min"] + metric_cols
            st.dataframe(f_df[cols_to_show], use_container_width=True, hide_index=True)
        else:
            if f_df.empty:
                st.info("По заданным фильтрам никто не подошел.")
            else:
                for idx, row in f_df.iterrows():
                    with st.container(border=True):
                        c1, c2, c3 = st.columns([1, 4, 3])
                        with c1:
                            if row["Image URL"]: st.image(row["Image URL"], width=100)
                            else: st.markdown("👤")
                        with c2:
                            badge = "badge-high" if row["injury_risk"] == "HIGH" else "badge-low"
                            st.markdown(f'<div class="player-name">{row["Player"]}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div style="color:#9CA3AF; font-size:0.9rem; margin-bottom:6px;">{get_player_meta_str(row)}</div>', unsafe_allow_html=True)
                            st.markdown(f'<span class="badge {badge}">Риск травм: {row["injury_risk"]}</span> <span style="margin-left:12px; font-size:0.9rem;">💰 {row["Price"]:,} €</span>'.replace(",", " "), unsafe_allow_html=True)
                        with c3:
                            st.markdown(f"**Характеристики:** {row['Height']} см | {row['Foot']} нога")
                            metrics = GK_METRIC_COLUMNS if row["Pos"] == "GK" else FIELD_METRIC_COLUMNS
                            m_cols = st.columns(len(metrics))
                            for m_idx, m in enumerate(metrics):
                                m_cols[m_idx].metric(m, f"{row[m]}")

    # ВКЛАДКА 2: СРАВНЕНИЕ ИГРОКОВ
    with tab_compare:
        all_names = df["Player"].tolist()
        c1, c2 = st.columns(2)
        p_a = c1.selectbox("Игрок 1", all_names, index=all_names.index("Neymar") if "Neymar" in all_names else 0)
        p_b = c2.selectbox("Игрок 2", all_names, index=all_names.index("Roberto Carlos") if "Roberto Carlos" in all_names else min(1, len(all_names)-1))

        if p_a == p_b:
            st.info("Выберите двух разных футболистов для сравнения.")
        else:
            row_a = df[df["Player"] == p_a].iloc[0]
            row_b = df[df["Player"] == p_b].iloc[0]
            
            is_gk_comparison = (row_a["Pos"] == "GK" and row_b["Pos"] == "GK")
            metrics = GK_METRIC_COLUMNS if is_gk_comparison else FIELD_METRIC_COLUMNS

            normalized = normalize_metrics(df, metrics)
            categories = [METRIC_LABELS[m] for m in metrics]
            categories_closed = categories + [categories[0]]

            fig = go.Figure()
            for p_name, (l_c, f_c) in zip([p_a, p_b], [("#00D4AA", "rgba(0, 212, 170, 0.2)"), ("#4FC3F7", "rgba(79, 195, 247, 0.2)")]):
                p_idx = df[df["Player"] == p_name].index[0]
                values = normalized.loc[p_idx, metrics].tolist()
                fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories_closed, fill="toself", name=p_name, line=dict(color=l_c, width=2), fillcolor=f_c))
            
            fig.update_layout(polar=dict(bgcolor="rgba(26,29,36,0.9)", radialaxis=dict(visible=True, range=[0,1])), paper_bgcolor="rgba(0,0,0,0)", height=450)
            st.plotly_chart(fig, use_container_width=True)

            col_left, col_right = st.columns(2)
            for col, r_data in zip([col_left, col_right], [row_a, row_b]):
                with col:
                    st.markdown(f"#### {r_data['Player']}")
                    if r_data["Image URL"]: st.image(r_data["Image URL"], width=120)
                    st.markdown(f'<div style="color:#9CA3AF; font-size:0.9rem; margin: 8px 0;">{get_player_meta_str(r_data)}</div>', unsafe_allow_html=True)
                    m_display = st.columns(len(metrics))
                    for m_i, m in enumerate(metrics):
                        m_display[m_i].metric(m, f"{r_data[m]}")

    # ВКЛАДКА 3: ПОИСК АНАЛОГОВ
    with tab_similar:
        ref_player = st.selectbox(
            "Игрок-ориентир для поиска аналогов", 
            df["Player"].tolist(), 
            index=df["Player"].tolist().index("Neymar") if "Neymar" in df["Player"].tolist() else 0,
            key="ref_player_select"
        )
        ref_row = df[df["Player"] == ref_player].iloc[0]
        
        is_gk = (ref_row["Pos"] == "GK")
        metrics = GK_METRIC_COLUMNS if is_gk else FIELD_METRIC_COLUMNS
        
        st.markdown(f"##### Настройка приоритетов скаутинга для позиции: **{ref_row['Позиция']}**")
        w_cols = st.columns(len(metrics))
        weights = []
        for w_idx, m in enumerate(metrics):
            weights.append(w_cols[w_idx].slider(f"Вес {m}", 0.0, 2.0, 1.0, 0.1, key=f"w_sim_{m}"))

        if sum(weights) == 0:
            st.warning("Укажите хотя бы один ненулевой вес.")
        else:
            calc_df = f_df.copy()
            if ref_player not in calc_df["Player"].values:
                calc_df = pd.concat([calc_df, df[df["Player"] == ref_player]]).drop_duplicates(subset=["Player"]).reset_index(drop=True)

            similar_df = find_similar_players_weighted(calc_df, ref_player, top_n=3, weights=weights)
            
            # --- БЛОК 1: ВИЗУАЛЬНЫЕ КАРТОЧКИ АНАЛОГОВ С ФОТОГРАФИЯМИ И РИСКОМ ---
            st.markdown("##### 🎯 ТОП-3 Найденных аналога")
            
            cards_df = similar_df[similar_df["Player"] != ref_player].head(3)
            card_cols = st.columns(3)
            
            if cards_df.empty:
                st.info("Аналоги не найдены. Попробуйте смягчить фильтры в левом меню.")
            else:
                for idx, (_, row) in enumerate(cards_df.reset_index().iterrows()):
                    with card_cols[idx]:
                        with st.container(border=True):
                            img_col, txt_col = st.columns([1, 2])
                            with img_col:
                                if row["Image URL"]:
                                    st.image(row["Image URL"], use_container_width=True)
                                else:
                                    st.markdown("<h1 style='text-align: center; margin: 0;'>👤</h1>", unsafe_allow_html=True)
                            with txt_col:
                                badge = "badge-high" if row["injury_risk"] == "HIGH" else "badge-low"
                                st.markdown(
                                    f"""
                                    <div style="padding-left: 5px;">
                                        <h4 style="margin:0 0 3px 0; color:#FAFAFA; font-size:1.1rem;">{row['Player']}</h4>
                                        <p style="margin:0 0 4px 0; font-size:13px; color:#9CA3AF; line-height:1.2;">{get_player_meta_str(row)}</p>
                                        <div style="margin-bottom: 4px;"><span class="badge {badge}">Риск: {row['injury_risk']}</span></div>
                                        <h3 style="margin:4px 0 0 0; color:#00D4AA; font-size:1.5rem;">{row['Схожесть']}%</h3>
                                        <p style="margin:0; font-size:12px; color:#9CA3AF;">Цена: {f"{int(row['Price']):,}".replace(",", " ")} €</p>
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                            
                            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                            m_cols = st.columns(len(metrics))
                            for m_idx, m in enumerate(metrics):
                                m_cols[m_idx].metric(m, f"{row[m]}")
            
            st.markdown("---")
            
            # --- БЛОК 2: ТАБЛИЦА С ДОБАВЛЕНИЕМ СТОЛБЦА injury_risk ---
            st.markdown(f"##### 📊 Детальное сравнение метрик")
            st.caption("Первая строка (голубая) — выбранный эталон. Зелёным цветом подсвечены параметры аналогов, максимально близкие к нему или превосходящие его.")

            cols_to_display = ["Player", "Club", "Age", "Price", "injury_risk", "Схожесть"] + metrics
            view_df = similar_df[cols_to_display].copy()
            
            def highlight_similarity(data):
                style_df = pd.DataFrame('', index=data.index, columns=data.columns)
                if data.empty:
                    return style_df
                
                style_df.iloc[0, :] = 'background-color: rgba(79, 195, 247, 0.15); font-weight: bold; border-bottom: 2px solid #4FC3F7;'
                
                target_values = data.iloc[0][metrics].astype(float)
                
                for i in range(1, len(data)):
                    for m in metrics:
                        val = float(data.iloc[i][m])
                        target_val = float(target_values[m])
                        
                        if target_val > 0:
                            delta = abs(val - target_val) / target_val
                            if delta <= 0.25 or val >= target_val:
                                style_df.iloc[i, style_df.columns.get_loc(m)] = 'background-color: rgba(16, 185, 129, 0.2); color: #34D399; font-weight: bold;'
                        else:
                            if val == target_val:
                                style_df.iloc[i, style_df.columns.get_loc(m)] = 'background-color: rgba(16, 185, 129, 0.2); color: #34D399; font-weight: bold;'
                return style_df

            styled_view = view_df.style.apply(highlight_similarity, axis=None).format({
                "Price": lambda x: f"{int(x):,} €".replace(",", " "),
                "Схожесть": "{:.1f}%"
            })
            
            st.dataframe(styled_view, use_container_width=True, hide_index=True)
            
            csv_data = similar_df.to_csv(index=False).encode('utf-8')
            st.sidebar.download_button(
                label="📥 Скачать полный отчет по аналогам (CSV)",
                data=csv_data,
                file_name=f"similars_to_{ref_player.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
