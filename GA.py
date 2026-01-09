import streamlit as st
import pandas as pd
import numpy as np
import random

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Cinema Ticket Pricing Optimization (GA)",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Cinema Ticket Pricing Optimization Using Genetic Algorithm")
st.write(
    "This application applies a Genetic Algorithm (GA) to determine the optimal "
    "cinema ticket price that maximizes total revenue based on historical demand data."
)

# ======================================================
# FILE UPLOAD
# ======================================================
uploaded_file = st.file_uploader(
    "📂 Upload cinema ticket dataset (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

# ======================================================
# LOAD DATA
# ======================================================
try:
    df = pd.read_csv(uploaded_file)
except Exception:
    df = pd.read_csv(uploaded_file, engine="python")

st.success("Dataset loaded successfully ✅")
st.dataframe(df, use_container_width=True)

# ======================================================
# COLUMN SELECTION (AUTO: PRICE & PERSON ONLY)
# ======================================================
st.subheader("📌 Column Selection")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error("Dataset must contain at least two numeric columns.")
    st.stop()

price_col = numeric_cols[0]
sold_col = numeric_cols[1]

st.write(f"🎟 **Ticket Price Column:** `{price_col}`")
st.write(f"👥 **Number of Persons Column:** `{sold_col}`")

# ======================================================
# BASIC DATA CHECK
# ======================================================
df = df[[price_col, sold_col]].dropna()

# ======================================================
# DATA AUGMENTATION (IF DATA TOO SMALL)
# ======================================================
if len(df) < 5:
    st.warning("Dataset is small. Performing data augmentation.")

    price_range = np.linspace(df[price_col].min(), df[price_col].max(), 15)
    demand_interp = np.interp(
        price_range,
        df[price_col],
        df[sold_col]
    )

    df = pd.DataFrame({
        price_col: price_range,
        sold_col: demand_interp.astype(int)
    })

    st.success("Data augmentation completed.")
    st.dataframe(df, use_container_width=True)

# ======================================================
# PRICE RANGE
# ======================================================
PRICE_MIN = float(df[price_col].min())
PRICE_MAX = float(df[price_col].max())

# ======================================================
# DEMAND ESTIMATION FUNCTION
# ======================================================
def estimate_demand(price):
    idx = (df[price_col] - price).abs().idxmin()
    return df.loc[idx, sold_col]

# ======================================================
# FITNESS FUNCTION
# ======================================================
def fitness(price):
    return price * estimate_demand(price)

# ======================================================
# GA PARAMETERS
# ======================================================
st.sidebar.header("⚙️ Genetic Algorithm Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 30, 200, 80)
GENERATIONS = st.sidebar.slider("Number of Generations", 50, 300, 150)
CROSSOVER_RATE = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.8)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.05)
ELITISM_SIZE = st.sidebar.slider("Elitism Size", 1, 5, 2)

# ======================================================
# GA OPERATORS
# ======================================================
def init_population():
    return [random.uniform(PRICE_MIN, PRICE_MAX) for _ in range(POP_SIZE)]

def selection(population):
    tournament = random.sample(population, 3)
    return max(tournament, key=fitness)

def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        return (p1 + p2) / 2
    return p1

def mutation(price):
    if random.random() < MUTATION_RATE:
        return random.uniform(PRICE_MIN, PRICE_MAX)
    return price

# ======================================================
# RUN GENETIC ALGORITHM
# ======================================================
if st.button("🚀 Run Genetic Algorithm"):

    population = init_population()
    best_prices = []
    best_revenues = []

    progress = st.progress(0.0)

    for gen in range(GENERATIONS):
        population = sorted(population, key=fitness, reverse=True)
        new_population = population[:ELITISM_SIZE]

        while len(new_population) < POP_SIZE:
            p1 = selection(population)
            p2 = selection(population)
            child = crossover(p1, p2)
            child = mutation(child)
            new_population.append(child)

        population = new_population
        best_price = population[0]
        best_prices.append(best_price)
        best_revenues.append(fitness(best_price))

        progress.progress((gen + 1) / GENERATIONS)

    # ======================================================
    # FINAL RESULT
    # ======================================================
    st.subheader("🏆 Optimization Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Optimal Ticket Price",
        f"RM {best_price:.2f}"
    )

    col2.metric(
        "Estimated Number of Persons",
        int(estimate_demand(best_price))
    )

    col3.metric(
        "Maximum Revenue",
        f"RM {fitness(best_price):.2f}"
    )

    # ======================================================
    # VISUALIZATION 1: PRICE vs DEMAND
    # ======================================================
    st.markdown("## 📊 Ticket Price vs Number of Persons")

    st.scatter_chart(
        df,
        x=price_col,
        y=sold_col
    )

    # ======================================================
    # VISUALIZATION 2: PRICE vs REVENUE CURVE
    # ======================================================
    st.markdown("## 💰 Ticket Price vs Revenue")

    price_range = np.linspace(PRICE_MIN, PRICE_MAX, 100)
    revenue_curve = [p * estimate_demand(p) for p in price_range]

    revenue_df = pd.DataFrame({
        "Ticket Price": price_range,
        "Revenue": revenue_curve
    })

    st.line_chart(revenue_df, x="Ticket Price", y="Revenue")

    # ======================================================
    # VISUALIZATION 3: GA LEARNING CURVE
    # ======================================================
    st.markdown("## 🧬 Genetic Algorithm Learning Curve")

    ga_df = pd.DataFrame({
        "Generation": range(1, GENERATIONS + 1),
        "Best Revenue": best_revenues
    })

    st.line_chart(ga_df, x="Generation", y="Best Revenue")

    st.success("🎉 Genetic Algorithm optimization completed successfully!")
