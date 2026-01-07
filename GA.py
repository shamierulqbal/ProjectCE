import streamlit as st
import random
import numpy as np
import pandas as pd

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Cinema Ticket Pricing Optimization",
    page_icon="🎬",
    layout="centered"
)

# ----------------------------------
# Streamlit UI
# ----------------------------------
st.title("🎬 Optimizing Cinema Ticket Pricing")
st.markdown("### Genetic Algorithm with Dataset Upload")

st.write(
    "This application allows users to upload a **real or simulated dataset** "
    "containing ticket price and demand. A **Genetic Algorithm** is then used "
    "to find the optimal ticket price that maximizes cinema revenue."
)

# ----------------------------------
# Dataset Upload (FINAL FIX)
# ----------------------------------
st.subheader("📤 Upload Price–Demand Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    df = None

    try:
        # Try Excel first
        if uploaded_file.name.lower().endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        # Try CSV with multiple encodings & separators
        else:
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except:
                try:
                    df = pd.read_csv(uploaded_file, encoding="latin1")
                except:
                    df = pd.read_csv(uploaded_file, sep=";", encoding="latin1")

    except Exception as e:
        st.error("❌ Unable to read file. Please check file format.")
        st.stop()

    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()

    # Try to auto-detect columns
    price_col = None
    demand_col = None

    for col in df.columns:
        if "price" in col:
            price_col = col
        if "demand" in col:
            demand_col = col

    if price_col is None or demand_col is None:
        st.error(
            "❌ Could not detect 'price' and 'demand' columns.\n\n"
            "Please ensure your dataset contains columns like:\n"
            "`price`, `ticket_price`, `demand`, `customers`"
        )
        st.stop()

    # Keep only needed columns
    df = df[[price_col, demand_col]]
    df.columns = ["price", "demand"]

    # Convert to numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    df = df.dropna()

    st.success("✅ Dataset loaded successfully!")
    st.dataframe(df)

    # ----------------------------------
    # Interpolated Demand Function
    # ----------------------------------
    def demand(price):
        return np.interp(price, df["price"], df["demand"])

    # ----------------------------------
    # Fitness Function
    # ----------------------------------
    def fitness(price):
        return price * demand(price)

    # ----------------------------------
    # GA Parameters
    # ----------------------------------
    st.sidebar.header("🧬 Genetic Algorithm Parameters")

    population_size = st.sidebar.slider("Population Size", 20, 200, 80)
    generations = st.sidebar.slider("Number of Generations", 20, 200, 100)
    mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)

    # ----------------------------------
    # Genetic Algorithm
    # ----------------------------------
    def genetic_algorithm():
        population = np.random.uniform(
            df["price"].min(),
            df["price"].max(),
            population_size
        )

        best_fitness_history = []

        for _ in range(generations):
            fitness_scores = np.array([fitness(p) for p in population])
            best_fitness_history.append(fitness_scores.max())

            # Selection (Tournament)
            selected = []
            for _ in range(population_size):
                i, j = np.random.choice(len(population), 2, replace=False)
                selected.append(
                    population[i] if fitness_scores[i] > fitness_scores[j] else population[j]
                )

            # Crossover
            offspring = []
            for i in range(0, population_size, 2):
                p1 = selected[i]
                p2 = selected[min(i + 1, population_size - 1)]
                alpha = random.random()
                offspring.extend([
                    alpha * p1 + (1 - alpha) * p2,
                    alpha * p2 + (1 - alpha) * p1
                ])

            # Mutation
            for i in range(len(offspring)):
                if random.random() < mutation_rate:
                    offspring[i] += random.uniform(-2, 2)

            population = np.clip(
                offspring,
                df["price"].min(),
                df["price"].max()
            )

        best_price = population[np.argmax([fitness(p) for p in population])]
        return best_price, fitness(best_price), best_fitness_history

    # ----------------------------------
    # Run Optimization
    # ----------------------------------
    if st.button("🚀 Run Optimization"):
        with st.spinner("Running Genetic Algorithm..."):
            best_price, best_revenue, history = genetic_algorithm()

        st.success("Optimization Completed!")

        # Results
        col1, col2, col3 = st.columns(3)
        col1.metric("🎟️ Optimal Ticket Price (RM)", f"{best_price:.2f}")
        col2.metric("👥 Expected Demand", int(demand(best_price)))
        col3.metric("💰 Maximum Revenue (RM)", f"{best_revenue:.2f}")

        # Convergence plot
        st.subheader("📈 GA Convergence Curve")
        st.line_chart(history)

        # Price vs Revenue
        st.subheader("✨ Ticket Price vs Revenue")
        prices = np.linspace(df["price"].min(), df["price"].max(), 100)
        revenues = [fitness(p) for p in prices]

        chart_data = {
            "Price (RM)": prices,
            "Revenue (RM)": revenues
        }

        st.line_chart(chart_data)

        st.info(
            "📌 The optimal ticket price is determined using real demand data "
            "rather than a predefined mathematical function."
        )

else:
    st.warning("Please upload a dataset to proceed.")
