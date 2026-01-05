import streamlit as st
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Title of the app
st.title("Cinema Ticket Pricing Optimization using Genetic Algorithm")

# ======================================
# 1. UPLOAD DATASET
# ======================================
uploaded_file = st.file_uploader("Upload your CSV file (e.g., cinema_hall_ticket_sales.csv)", type="csv")

if uploaded_file is not None:
    # ======================================
    # 3. LOAD DATASET (FIX ENCODING)
    # ======================================
    try:
        df = pd.read_csv(uploaded_file, encoding="latin1")
        st.success("Dataset successfully loaded ✅")
        st.write("Dataset Preview:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    # ======================================
    # 4. CHECK COLUMN NAMES
    # ======================================
    st.write("\nColumns in the dataset:")
    st.write(df.columns)

    # ======================================
    # 5. TETAPKAN COLUMN (UBAH JIKA PERLU)
    # ======================================
    # Allow user to select columns
    price_col_options = df.columns.tolist()
    sold_col_options = df.columns.tolist()
    
    price_col = st.selectbox("Select Price Column", price_col_options, index=0)
    sold_col = st.selectbox("Select Tickets Sold Column", sold_col_options, index=1 if len(sold_col_options) > 1 else 0)

    st.write(f"Selected Price Column: {price_col}")
    st.write(f"Selected Tickets Sold Column: {sold_col}")

    # Validate selected columns
    if price_col not in df.columns or sold_col not in df.columns:
        st.error("Selected columns do not exist in the dataset. Please choose valid columns.")
        st.stop()

    # Check if columns contain numeric data
    if not pd.api.types.is_numeric_dtype(df[price_col]) or not pd.api.types.is_numeric_dtype(df[sold_col]):
        st.error("Selected columns must contain numeric data.")
        st.stop()

    # ======================================
    # 6. PRICE RANGE
    # ======================================
    PRICE_MIN = df[price_col].min()
    PRICE_MAX = df[price_col].max()

    if PRICE_MIN >= PRICE_MAX:
        st.error("Price range is invalid (min >= max). Check your data.")
        st.stop()

    # ======================================
    # 7. ESTIMATE DEMAND FROM DATASET
    # ======================================
    def estimate_demand(price, df, price_col, sold_col):
        idx = (df[price_col] - price).abs().idxmin()
        return df.loc[idx, sold_col]

    # ======================================
    # 8. FITNESS FUNCTION
    # ======================================
    def fitness(price, df, price_col, sold_col):
        tickets_sold = estimate_demand(price, df, price_col, sold_col)
        return price * tickets_sold

    # ======================================
    # 9. GA PARAMETERS
    # ======================================
    st.sidebar.header("GA Parameters")
    POP_SIZE = st.sidebar.slider("Population Size", 10, 100, 50)
    GENERATIONS = st.sidebar.slider("Generations", 10, 200, 100)
    CROSSOVER_RATE = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.8)
    MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.05)
    ELITISM_SIZE = st.sidebar.slider("Elitism Size", 1, min(5, POP_SIZE-1), 2)

    # ======================================
    # 10. INITIAL POPULATION
    # ======================================
    def init_population(PRICE_MIN, PRICE_MAX, POP_SIZE):
        return [random.uniform(PRICE_MIN, PRICE_MAX) for _ in range(POP_SIZE)]

    # ======================================
    # 11. SELECTION
    # ======================================
    def selection(population, fitness, df, price_col, sold_col):
        if len(population) < 3:
            return max(population, key=lambda p: fitness(p, df, price_col, sold_col))
        tournament = random.sample(population, 3)
        return max(tournament, key=lambda p: fitness(p, df, price_col, sold_col))

    # ======================================
    # 12. CROSSOVER
    # ======================================
    def crossover(p1, p2, CROSSOVER_RATE):
        if random.random() < CROSSOVER_RATE:
            return (p1 + p2) / 2
        return p1

    # ======================================
    # 13. MUTATION
    # ======================================
    def mutation(price, MUTATION_RATE, PRICE_MIN, PRICE_MAX):
        if random.random() < MUTATION_RATE:
            return random.uniform(PRICE_MIN, PRICE_MAX)
        return price

    # Button to run the GA
    if st.button("Run Genetic Algorithm"):
        # ======================================
        # 14. MAIN GA LOOP (WITH ELITISM)
        # ======================================
        population = init_population(PRICE_MIN, PRICE_MAX, POP_SIZE)
        best_fitness_history = []
        best_price_history = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for gen in range(GENERATIONS):
            # Sort population by fitness
            population = sorted(population, key=lambda p: fitness(p, df, price_col, sold_col), reverse=True)
            
            # Elitism
            new_population = population[:ELITISM_SIZE]

            # Generate new population
            while len(new_population) < POP_SIZE:
                p1 = selection(population, fitness, df, price_col, sold_col)
                p2 = selection(population, fitness, df, price_col, sold_col)
                child = crossover(p1, p2, CROSSOVER_RATE)
                child = mutation(child, MUTATION_RATE, PRICE_MIN, PRICE_MAX)
                new_population.append(child)

            population = new_population

            best_price = population[0]
            best_price_history.append(best_price)
            best_fitness = fitness(best_price, df, price_col, sold_col)
            best_fitness_history.append(best_fitness)

            # Update progress
            progress = (gen + 1) / GENERATIONS
            progress_bar.progress(progress)
            if gen % 10 == 0:
                status_text.text(f"Gen {gen} | Best Price RM {best_price:.2f} | Revenue RM {best_fitness:.2f}")

        # ======================================
        # 15. FINAL RESULT
        # ======================================
        best_price = population[0]
        best_fitness = fitness(best_price, df, price_col, sold_col)
        st.write("\n===== FINAL RESULT =====")
        st.write(f"Optimal Ticket Price : RM {best_price:.2f}")
        st.write(f"Estimated Tickets Sold : {estimate_demand(best_price, df, price_col, sold_col)}")
        st.write(f"Maximum Revenue : RM {best_fitness:.2f}")

        # ======================================
        # 16. PLOT OPTIMIZATION GRAPH
        # ======================================
        fig1, ax1 = plt.subplots()
        ax1.plot(best_fitness_history)
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Best Revenue (RM)")
        ax1.set_title("GA Optimization of Cinema Ticket Pricing")
        st.pyplot(fig1)

        # ======================================
        # 17. PLOT PRICE EVOLUTION
        # ======================================
        fig2, ax2 = plt.subplots()
        ax2.plot(best_price_history)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Ticket Price (RM)")
        ax2.set_title("Evolution of Ticket Price")
        st.pyplot(fig2)
else:
    st.info("Please upload a CSV file to proceed.")
