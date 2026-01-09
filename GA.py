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

# ======================================================
# FILE UPLOAD
# ======================================================
uploaded_file = st.file_uploader("📂 Upload cinema ticket dataset (CSV)", type=["csv"])
if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
st.dataframe(df, use_container_width=True)

# ======================================================
# AUTO COLUMN DETECTION
# ======================================================
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
    st.error("Required columns not found.")
    st.stop()

# ======================================================
# 🔴 CRITICAL FIX: FORCE NUMERIC
# ======================================================
df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
df[sold_col] = pd.to_numeric(df[sold_col], errors="coerce")

df = df[[price_col, sold_col]].dropna()

if df.empty:
    st.error("No valid numeric data after cleaning.")
    st.stop()

st.success("Numeric data cleaned successfully")

# ======================================================
# DATA AUGMENTATION
# ======================================================
if len(df) < 5:
    prices = np.linspace(df[price_col].min(), df[price_col].max(), 15)
    demand = np.interp(prices, df[price_col], df[sold_col])
    df = pd.DataFrame({price_col: prices, sold_col: demand.astype(int)})

# ======================================================
# GA SETUP (SAFE FLOAT)
# ======================================================
PRICE_MIN = float(df[price_col].min())
PRICE_MAX = float(df[price_col].max())

def estimate_demand(price):
    idx = (df[price_col] - price).abs().idxmin()
    return int(df.loc[idx, sold_col])

def fitness(price):
    return price * estimate_demand(price)

# ======================================================
# PARAMETERS
# ======================================================
POP_SIZE = 80
GENERATIONS = 150
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.05
ELITISM_SIZE = 2

# ======================================================
# GA OPERATORS
# ======================================================
def init_population():
    return [random.uniform(PRICE_MIN, PRICE_MAX) for _ in range(POP_SIZE)]

def selection(pop):
    return max(random.sample(pop, 3), key=fitness)

def crossover(p1, p2):
    return (p1 + p2) / 2 if random.random() < CROSSOVER_RATE else p1

def mutation(price):
    return random.uniform(PRICE_MIN, PRICE_MAX) if random.random() < MUTATION_RATE else price

# ======================================================
# RUN GA
# ======================================================
if st.button("🚀 Run Genetic Algorithm"):

    population = init_population()
    history = []

    for gen in range(GENERATIONS):
        population = sorted(population, key=fitness, reverse=True)
        new_population = population[:ELITISM_SIZE]

        while len(new_population) < POP_SIZE:
            child = mutation(crossover(selection(population), selection(population)))
            new_population.append(child)

        population = new_population
        best = population[0]

        history.append({
            "Generation": gen + 1,
            "Price": best,
            "Demand": estimate_demand(best),
            "Fitness": fitness(best)
        })

    result_df = pd.DataFrame(history)
    best_row = result_df.loc[result_df["Fitness"].idxmax()]

    st.markdown("## 🏆 Best Solution")
    st.metric("Optimal Price", f"RM {best_row['Price']:.2f}")
    st.metric("Best Fitness (Revenue)", f"RM {best_row['Fitness']:.2f}")

    st.markdown("## 📈 GA Learning Curve")
    st.line_chart(result_df.set_index("Generation")["Fitness"])

    st.markdown("## 🔝 Top 5 Solutions")
    st.dataframe(
        result_df.sort_values("Fitness", ascending=False).head(5),
        use_container_width=True
    )
