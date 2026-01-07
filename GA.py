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
Aplikasi ini menggunakan **Genetic Algorithm (GA)** untuk mencari harga tiket yang akan memaksimumkan hasil (revenue) 
berdasarkan data jualan sejarah anda.
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
# 1. MUAT NAIK DATA
# ======================================
st.subheader("1. Muat Naik Data Jualan")
uploaded_file = st.file_uploader("Upload CSV (Mesti ada kolum Harga & Unit Terjual)", type=["csv"])

if uploaded_file:
    # Load Data
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except:
        df = pd.read_csv(uploaded_file, encoding='cp1252')

    # Pembersihan Data Mudah
    df = df.dropna()

    col1, col2 = st.columns(2)
    with col1:
        price_col = st.selectbox("Pilih Kolum Harga (RM)", df.columns)
    with col2:
        sold_col = st.selectbox("Pilih Kolum Unit Terjual", df.columns)

    # Validasi Range
    PRICE_MIN = float(df[price_col].min())
    PRICE_MAX = float(df[price_col].max())

    # ======================================
    # 2. LOGIK GENETIC ALGORITHM
    # ======================================
    
    # Fungsi Anggaran Permintaan (Interpolation)
    # Ini lebih baik daripada 'closest match' kerana ia menghasilkan lengkung yang licin
    def get_demand(price):
        return np.interp(price, df[price_col].sort_values(), df[sold_col])

    def fitness(price):
        revenue = price * get_demand(price)
        return revenue

    # Operasi GA
    def create_individual():
        return random.uniform(PRICE_MIN, PRICE_MAX)

    def crossover(p1, p2):
        if random.random() < CROSSOVER_RATE:
            # Arithmetic Crossover
            alpha = random.random()
            return (alpha * p1) + ((1 - alpha) * p2)
        return p1

    def mutate(individual):
        if random.random() < MUTATION_RATE:
            # Tambah sedikit gangguan rawak (Gaussian Mutation)
            mutation_range = (PRICE_MAX - PRICE_MIN) * 0.1
            individual += random.uniform(-mutation_range, mutation_range)
            # Pastikan dalam range
            individual = max(PRICE_MIN, min(PRICE_MAX, individual))
        return individual

    # ======================================
    # 3. RUN SIMULATION
    # ======================================
    if st.button("🚀 Jalankan Optimasi"):
        # Initialization
        population = [create_individual() for _ in range(POP_SIZE)]
        history_revenue = []
        history_best_price = []

        progress_bar = st.progress(0)

        for gen in range(GENERATIONS):
            # Elitism: Simpan yang terbaik
            population.sort(key=fitness, reverse=True)
            new_population = population[:2] # Simpan 2 terbaik

            # Create next generation
            while len(new_population) < POP_SIZE:
                # Selection (Tournament)
                p1, p2 = random.sample(population[:10], 2) # Pilih dari 10 terbaik
                
                # Crossover
                child = crossover(p1, p2)
                
                # Mutation
                child = mutate(child)
                
                new_population.append(child)
            
            population = new_population
            best_ind = population[0]
            
            history_revenue.append(fitness(best_ind))
            history_best_price.append(best_ind)
            
            progress_bar.progress((gen + 1) / GENERATIONS)

        # HASIL AKHIR
        best_price = population[0]
        final_revenue = fitness(best_price)
        final_demand = get_demand(best_price)

        # ======================================
        # 4. PAPARAN KEPUTUSAN
        # ======================================
        st.divider()
        st.success("Optimasi Selesai!")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Harga Tiket Optimum", f"RM {best_price:.2f}")
        m2.metric("Anggaran Jualan", f"{int(final_demand)} Tiket")
        m3.metric("Jangkaan Hasil (Revenue)", f"RM {final_revenue:.2f}")

        tab1, tab2 = st.tabs(["Graf Prestasi", "Data Analitik"])
        
        with tab1:
            st.subheader("Evolusi Hasil vs Generasi")
            st.line_chart(history_revenue)
            st.caption("Graf ini menunjukkan bagaimana GA menemui harga yang lebih menguntungkan dari masa ke masa.")

        with tab2:
            st.subheader("Data Ringkasan")
            res_df = pd.DataFrame({
                "Generation": range(GENERATIONS),
                "Best Price (RM)": history_best_price,
                "Revenue (RM)": history_revenue
            })
            st.dataframe(res_df, use_container_width=True)

else:
    st.info("Sila muat naik fail CSV untuk memulakan.")
    # Contoh format fail
    st.write("Contoh format CSV yang diperlukan:")
    example_df = pd.DataFrame({
        "harga": [10, 15, 20, 25, 30],
        "tiket_terjual": [500, 450, 300, 200, 100]
    })
    st.table(example_df)
