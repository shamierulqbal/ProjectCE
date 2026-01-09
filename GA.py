import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Cinema Ticket Pricing Optimization (GA)",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Cinema Ticket Price Optimization using Genetic Algorithm")
st.write(
    """
    This application applies a **Genetic Algorithm (GA)** to determine the
    optimal cinema ticket price that maximizes revenue based on historical demand data.
    """
)

# ======================================================
# FILE UPLOAD
# ======================================================
uploaded_file = st.file_uploader(
    "📂 Upload cinema ticket sales dataset (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

# ======================================================
# LOAD DATA
# ======================================================
try:
    df = pd.read_csv(uploaded_file)
except:
    df = pd.read_csv(uploaded_file, encoding="latin1")

st.success("Dataset loaded successfully ✅")
st.dataframe(df)

# ======================================================
# AUTO-DETECT REQUIRED COLUMNS
# ======================================================
st.subheader("📌 Auto-detected Columns")

price_keywords = ["price", "ticket"]
demand_keywords = ["person", "sold", "demand", "quantity"]

def find_column(keywords):
    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            return col
    return None

price_col = find_column(price_keywords)
sold_col = find_column(demand_keywords)

if price_col is None or sold_col is None:
    st.error("❌ Cannot detect ticket price or number of persons column.")
    st.stop()

st.success(f"🎫 Ticket Price Column: {price_col}")
st.success(f"👥 Number of Persons Column: {sold_col}")

# ======================================================
# DATA PREPARATION
# ======================================================
st.subheader("🧪 Data Preparation")

if len(df) < 5:
    st.warning("Dataset too small. Generating synthetic data.")

    price_min = df[price_col].min()
    price_max = df[price_col].max()

    new_prices = np.linspace(price_min, price_max, 10)
    new_demand = np.interp(new_prices, df[price_col], df[sold_col])

    df = pd.DataFrame({
        price_col: new_prices,
        sold_col: new_demand.astype(int)
    })

    st.success("Data augmented successfully.")
    st.dataframe(df)

# ======================================================
# GA SETTINGS
# ======================================================
PRICE_MIN = float(df[price_col].min())
PRICE_MAX = float(df[price_col].max())

def estimate_demand(price):
    idx = (df[price_col] - price).abs().idxmin()
    return df.loc[idx, sold_col]

def fitness(price):
    return price * estimate_demand(price)

st.sidebar.header("⚙️ Genetic Algorithm Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 20, 200, 60)
GENERATIONS = st.sidebar.slider("Generations", 50, 300, 150)
CROSSOVER_RATE = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.8)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.05)
ELITISM_SIZE = st.sidebar.slider("Elitism Size", 1, 5, 2)

# ======================================================
# GA OPERATORS
# ======================================================
def init_population():
    return [random.uniform(PRICE_MIN, PRICE_MAX) for _ in range(POP_SIZE)]

def selection(pop):
    tournament = random.sample(pop, min(3, len(pop)))
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
    # RESULTS
    # ======================================================
    st.markdown("## 🏆 Optimization Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Optimal Ticket Price", f"RM {best_price:.2f}")
    col2.metric("Estimated Tickets Sold", int(estimate_demand(best_price)))
    col3.metric("Maximum Revenue", f"RM {fitness(best_price):.2f}")

    # ======================================================
    # PLOT 1: PRICE vs DEMAND
    # ======================================================
    st.markdown("## 📊 Ticket Price vs Number of Persons")

    fig1, ax1 = plt.subplots()
    ax1.scatter(df[price_col], df[sold_col])
    ax1.axvline(best_price, linestyle="--", label="Optimal Price")

    ax1.set_xlabel("Ticket Price (RM)")
    ax1.set_ylabel("Number of Persons")
    ax1.set_title("Relationship Between Ticket Price and Demand")
    ax1.legend()

    st.pyplot(fig1)

    # ======================================================
    # PLOT 2: PRICE vs REVENUE
    # ======================================================
    st.markdown("## 💰 Ticket Price vs Revenue")

    price_range = np.linspace(PRICE_MIN, PRICE_MAX, 100)
    revenue_curve = [p * estimate_demand(p) for p in price_range]

    fig2, ax2 = plt.subplots()
    ax2.plot(price_range, revenue_curve)
    ax2.axvline(best_price, linestyle="--", label="Optimal Price")
    ax2.scatter(best_price, fitness(best_price))

    ax2.set_xlabel("Ticket Price (RM)")
    ax2.set_ylabel("Revenue (RM)")
    ax2.set_title("Revenue Optimization Curve")
    ax2.legend()

    st.pyplot(fig2)

    # ======================================================
    # PLOT 3: GA LEARNING CURVE
    # ======================================================
    st.markdown("## 🧬 Genetic Algorithm Learning Curve")

    fig3, ax3 = plt.subplots()
    ax3.plot(range(1, GENERATIONS + 1), best_revenues)

    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Best Revenue (RM)")
    ax3.set_title("GA Convergence Curve")

    st.pyplot(fig3)

    st.success("🎉 Genetic Algorithm optimization completed successfully!")
