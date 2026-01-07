import streamlit as st
import random
import pandas as pd
import numpy as np

# ======================================
# KONFIGURASI HALAMAN
# ======================================
st.set_page_config(page_title="Cinema Price Optimizer", layout="wide")

st.title("🎬 Cinema Ticket Pricing Optimization")
st.markdown("""
Aplikasi ini menggunakan **Genetic Algorithm (GA)** untuk mencari harga tiket yang optimum bagi memaksimumkan hasil (revenue).
""")

# ======================================
# SIDEBAR - PARAMETER GA
# ======================================
st.sidebar.header("⚙️ Parameter Genetic Algorithm")
POP_SIZE = st.sidebar.slider("Population Size", 10, 200, 50)
GENERATIONS = st.sidebar.slider("Generations", 10, 300, 100)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
CROSSOVER_RATE = st.sidebar.slider("Crossover Rate", 0.5, 1.0, 0.8)

# ======================================
# 1. MUAT NAIK DATA & PEMPROSESAN
# ======================================
st.subheader("1. Muat Naik Data Jualan")
uploaded_file = st.file_uploader("Upload CSV (Mesti ada kolum Harga & Unit Terjual)", type=["csv"])

if uploaded_file:
    # Safe Loading dengan Reset Pointer (Seek 0)
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except Exception:
        uploaded_file.seek(0)  # Kembali ke permulaan fail
        try:
            df = pd.read_csv(uploaded_file, encoding='cp1252')
        except Exception as e:
            st.error(f"Gagal membaca fail: {e}")
            st.stop()

    if df.empty:
        st.error("Fail CSV kosong!")
        st.stop()

    st.success("Data berjaya dimuat naik!")
    
    col_a, col_b = st.columns(2)
    with col_a:
        price_col = st.selectbox("Pilih Kolum Harga (RM)", df.columns)
    with col_b:
        sold_col = st.selectbox("Pilih Kolum Unit Terjual", df.columns)

    # Pastikan data bersih dan tersusun untuk interpolasi
    df = df.dropna(subset=[price_col, sold_col])
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    df[sold_col] = pd.to_numeric(df[sold_col], errors='coerce')
    df = df.sort_values(by=price_col).reset_index(drop=True)

    PRICE_MIN = float(df[price_col].min())
    PRICE_MAX = float(df[price_col].max())

    # ======================================
    # 2. LOGIK EVOLUTIONARY ALGORITHM
    # ======================================
    
    # Fungsi Anggaran Permintaan menggunakan Linear Interpolation
    def get_demand(price):
        # np.interp memerlukan x-coordinates (harga) yang tersusun
        return np.interp(price, df[price_col], df[sold_col])

    # Fitness Function: Revenue = Price * Demand
    def fitness(price):
        return price * get_demand(price)

    # Inisialisasi Populasi Rawak
    def create_individual():
        return random.uniform(PRICE_MIN, PRICE_MAX)

    # Crossover (Arithmetic)
    def crossover(p1, p2):
        if random.random() < CROSSOVER_RATE:
            alpha = random.random()
            return (alpha * p1) + ((1 - alpha) * p2)
        return p1

    # Mutation (Gaussian)
    def mutate(individual):
        if random.random() < MUTATION_RATE:
            # Mutasi dalam range 10% daripada julat harga
            deviation = (PRICE_MAX - PRICE_MIN) * 0.1
            individual += random.gauss(0, deviation)
            individual = max(PRICE_MIN, min(PRICE_MAX, individual))
        return individual

    # ======================================
    # 3. PROSES OPTIMASI
    # ======================================
    if st.button("🚀 Jalankan Optimasi"):
        # Create Initial Population
        population = [create_individual() for _ in range(POP_SIZE)]
        
        history_revenue = []
        history_best_price = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for gen in range(GENERATIONS):
            # Sort mengikut fitness (terbaik di atas)
            population.sort(key=fitness, reverse=True)
            
            # Elitism (Kekalkan 2 terbaik)
            new_population = population[:2]

            while len(new_population) < POP_SIZE:
                # Selection (Tournament)
                tournament = random.sample(population[:max(5, POP_SIZE//2)], 3)
                parent1 = max(tournament, key=fitness)
                parent2 = selection_alt = random.choice(population[:10])

                # Breeding
                child = crossover(parent1, parent2)
                child = mutate(child)
                new_population.append(child)

            population = new_population
            best_ind = population[0]
            
            history_revenue.append(fitness(best_ind))
            history_best_price.append(best_ind)
            
            progress_bar.progress((gen + 1) / GENERATIONS)
            if gen % 10 == 0:
                status_text.text(f"Generasi {gen}: Harga Terbaik RM{best_ind:.2f}")

        # Keputusan Akhir
        best_price = population[0]
        final_revenue = fitness(best_price)
        final_demand = get_demand(best_price)

        # ======================================
        # 4. PAPARAN HASIL
        # ======================================
        st.divider()
        st.balloons()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Harga Tiket Optimum", f"RM {best_price:.2f}")
        col2.metric("Anggaran Unit Terjual", f"{int(final_demand)} unit")
        col3.metric("Maksimum Revenue", f"RM {final_revenue:.2f}")

        # Visualisasi
        tab1, tab2 = st.tabs(["📈 Graf Prestasi", "📋 Data Perjalanan GA"])
        
        with tab1:
            st.subheader("Peningkatan Revenue mengikut Generasi")
            st.line_chart(history_revenue)
            
            st.subheader("Perubahan Harga mengikut Generasi")
            st.line_chart(history_best_price)

        with tab2:
            df_history = pd.DataFrame({
                "Generasi": range(1, GENERATIONS + 1),
                "Harga (RM)": history_best_price,
                "Revenue (RM)": history_revenue
            })
            st.dataframe(df_history, use_container_width=True)

else:
    st.info("Sila muat naik fail CSV untuk memulakan.")
    # Preview format yang diperlukan
    st.markdown("""
    **Format CSV yang dicadangkan:**
    | harga | unit_terjual |
    |-------|--------------|
    | 12.0  | 100          |
    | 15.0  | 85           |
    | 18.0  | 60           |
    """)
