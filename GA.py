import streamlit as st
import random
import pandas as pd
import numpy as np
import io

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
# 1. MUAT NAIK DATA (ROBUST LOADING)
# ======================================
st.subheader("1. Muat Naik Data Jualan")
uploaded_file = st.file_uploader("Upload CSV (Mesti ada kolum Harga & Unit Terjual)", type=["csv"])

if uploaded_file:
    df = None
    # Senarai encoding untuk dicuba jika gagal
    encodings = ['utf-8', 'cp1252', 'latin-1', 'utf-16']
    
    for enc in encodings:
        try:
            uploaded_file.seek(0) # Reset setiap kali cubaan baru
            df = pd.read_csv(uploaded_file, encoding=enc)
            st.success(f"Data berjaya dibaca menggunakan format: {enc}")
            break
        except Exception:
            continue

    if df is None:
        st.error("Gagal membaca fail. Sila pastikan fail CSV anda tidak rosak dan menggunakan format yang betul.")
        st.stop()

    if df.empty:
        st.error("Fail CSV kosong!")
        st.stop()
    
    # Pilih Kolum
    col_a, col_b = st.columns(2)
    with col_a:
        price_col = st.selectbox("Pilih Kolum Harga (RM)", df.columns)
    with col_b:
        sold_col = st.selectbox("Pilih Kolum Unit Terjual", df.columns)

    # Pembersihan Data
    df = df.dropna(subset=[price_col, sold_col])
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    df[sold_col] = pd.to_numeric(df[sold_col], errors='coerce')
    df = df.dropna() # Buang jika ada baris yang gagal jadi nombor
    df = df.sort_values(by=price_col).reset_index(drop=True)

    if len(df) < 2:
        st.warning("Data tidak mencukupi untuk melakukan interpolasi. Sila pastikan ada sekurang-kurangnya 2 titik data harga yang berbeza.")
        st.stop()

    PRICE_MIN = float(df[price_col].min())
    PRICE_MAX = float(df[price_col].max())

    # ======================================
    # 2. LOGIK EVOLUTIONARY ALGORITHM
    # ======================================
    
    def get_demand(price):
        # Menggunakan linear interpolation untuk menganggar permintaan di antara titik data
        return np.interp(price, df[price_col], df[sold_col])

    def fitness(price):
        # Objektif: Maksimumkan Revenue (Harga * Permintaan)
        return price * get_demand(price)

    def create_individual():
        return random.uniform(PRICE_MIN, PRICE_MAX)

    def crossover(p1, p2):
        if random.random() < CROSSOVER_RATE:
            alpha = random.random()
            return (alpha * p1) + ((1 - alpha) * p2)
        return p1

    def mutate(individual):
        if random.random() < MUTATION_RATE:
            deviation = (PRICE_MAX - PRICE_MIN) * 0.1
            individual += random.gauss(0, deviation)
            individual = max(PRICE_MIN, min(PRICE_MAX, individual))
        return individual

    # ======================================
    # 3. PROSES OPTIMASI
    # ======================================
    if st.button("🚀 Jalankan Optimasi"):
        population = [create_individual() for _ in range(POP_SIZE)]
        history_revenue = []
        history_best_price = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for gen in range(GENERATIONS):
            # Sort mengikut fitness (turun)
            population.sort(key=fitness, reverse=True)
            
            # Elitism
            new_population = population[:2]

            while len(new_population) < POP_SIZE:
                # Tournament Selection
                tournament = random.sample(population[:max(5, len(population)//2)], 3)
                parent1 = max(tournament, key=fitness)
                parent2 = random.choice(population[:10])

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

        tab1, tab2 = st.tabs(["📈 Graf Prestasi", "📋 Data Perjalanan GA"])
        
        with tab1:
            st.subheader("Evolusi Hasil vs Generasi")
            st.line_chart(history_revenue)
            st.caption("Menunjukkan peningkatan hasil (revenue) seiring dengan evolusi harga.")

            st.subheader("Evolusi Harga vs Generasi")
            st.line_chart(history_best_price)
            st.caption("Menunjukkan bagaimana algoritma 'mencari' harga yang paling stabil dan menguntungkan.")

        with tab2:
            df_history = pd.DataFrame({
                "Generasi": range(1, GENERATIONS + 1),
                "Harga (RM)": history_best_price,
                "Revenue (RM)": history_revenue
            })
            st.dataframe(df_history, use_container_width=True)

else:
    st.info("Sila muat naik fail CSV untuk memulakan.")
