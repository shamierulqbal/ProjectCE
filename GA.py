import streamlit as st
import pandas as pd
import numpy as np
import random

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Cinema Ticket Pricing Optimization (GA)",
    layout="wide"
)

st.title("Cinema Ticket Pricing Optimization Using Genetic Algorithm")
st.write("""
This application applies a Genetic Algorithm (GA) to determine the optimal cinema
ticket price based on customer demand.
""")

# ======================================================
# FILE UPLOAD (ONLY CSV)
# ======================================================
uploaded_file = st.file_uploader(
    "Upload cinema ticket dataset (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file with columns such as 'ticket_price' and 'number_person'.")
    st.stop()

df = pd.read_csv(uploaded_file)

# ======================================================
# AUTO COLUMN DETECTION
# ======================================================
price_keywords = ["price", "ticket"]
demand_keywords = ["sold", "demand", "attendance", "quantity", "person", "number"]

def find_column(keywords):
    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            return col
    return None

price_col = find_column(price_keywords)
demand_col = find_column(demand_keywords)

if price_col is None or demand_col is None:
    st.error(
        "Price or Demand column not detected.\n\n"
        "Expected column names similar to:\n"
        "- ticket_price\n"
        "- number_person"
    )
    st.stop()

# ======================================================
# CLEAN DATA
# ======================================================
df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
df[demand_col] = pd.to_numeric(df[demand_col], errors="coerce")
df = df[[price_col, demand_col]].dropna().sort_values(by=price_col).reset_index(drop=True)

if len(df) < 5:
    st.error("Dataset too small. Please upload a dataset with more records.")
    st.stop()

st.success(f"Detected columns → Price: **{price_col}**, Demand: **{demand_col}**")
st.markdown("### Dataset Preview")
st.dataframe(df.head(10))

# ======================================================
# SIDEBAR – GA PARAMETERS ONLY
# ======================================================
st.sidebar.header("Genetic Algorithm Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 20, 200, 100)
GENERATIONS = st.sidebar.slider("Number of Generations", 50, 500, 200)
CROSSOVER_RATE = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.8)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.1)
ELITISM_SIZE = st.sidebar.slider("Elitism Size", 1, 10, 2)

st.sidebar.markdown("---")

objective_type = st.sidebar.selectbox(
    "Objective Function",
    ["Single Objective (Maximize Revenue)",
     "Multi Objective (Revenue + Demand)"]
)

w1, w2 = 1.0, 0.0
if "Multi" in objective_type:
    st.sidebar.subheader("Objective Weights")
    w1 = st.sidebar.slider("Revenue Weight (w1)", 0.0, 1.0, 0.7)
    w2 = st.sidebar.slider("Demand Weight (w2)", 0.0, 1.0, 0.3)

# ======================================================
# GA SUPPORT FUNCTIONS
# ======================================================
PRICE_MIN = float(df[price_col].min())
PRICE_MAX = float(df[price_col].max())

price_arr = df[price_col].values
demand_arr = df[demand_col].values

def estimate_demand(price):
    price = np.clip(price, PRICE_MIN, PRICE_MAX)
    return float(np.interp(price, price_arr, demand_arr))

def fitness_single(price):
    demand = estimate_demand(price)
    return price * demand

def fitness_multi(price):
    demand = estimate_demand(price)
    revenue = price * demand
    return (w1 * revenue) + (w2 * demand)

def init_population():
    return [random.uniform(PRICE_MIN, PRICE_MAX) for _ in range(POP_SIZE)]

def selection(pop, fitness_func):
    return max(random.sample(pop, 3), key=fitness_func)

def crossover(p1, p2):
    return (p1 + p2) / 2 if random.random() < CROSSOVER_RATE else p1

def mutation(price):
    if random.random() < MUTATION_RATE:
        price += random.uniform(-2, 2)
    return np.clip(price, PRICE_MIN, PRICE_MAX)

# ======================================================
# RUN GA
# ======================================================
if st.button("Run Genetic Algorithm"):

    fitness_func = fitness_single if "Single" in objective_type else fitness_multi

    population = init_population()
    best_history = []

    progress_bar = st.progress(0)

    for gen in range(GENERATIONS):
        population = sorted(population, key=fitness_func, reverse=True)
        best_history.append(population[0])

        new_population = population[:ELITISM_SIZE]

        while len(new_population) < POP_SIZE:
            p1 = selection(population, fitness_func)
            p2 = selection(population, fitness_func)
            child = crossover(p1, p2)
            child = mutation(child)
            new_population.append(child)

        population = new_population
        progress_bar.progress((gen + 1) / GENERATIONS)

    # ======================================================
    # RESULTS
    # ======================================================
    best_price = max(population, key=fitness_func)
    best_demand = estimate_demand(best_price)
    best_revenue = best_price * best_demand

    st.markdown("## Optimization Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Optimal Ticket Price", f"RM {best_price:.2f}")
    col2.metric("Estimated Demand", f"{int(best_demand)} people")
    col3.metric("Total Revenue", f"RM {best_revenue:,.2f}")

    # ======================================================
    # VISUALIZATION
    # ======================================================
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Price vs Fitness Curve**")
        prices = np.linspace(PRICE_MIN, PRICE_MAX, 100)
        fitness_values = [fitness_func(p) for p in prices]

        st.line_chart(pd.DataFrame({
            "Price": prices,
            "Fitness": fitness_values
        }).set_index("Price"))

    with c2:
        st.markdown("**GA Convergence Curve**")
        best_fitness = [fitness_func(p) for p in best_history]

        st.line_chart(pd.DataFrame({
            "Generation": range(1, len(best_fitness) + 1),
            "Best Fitness": best_fitness
        }).set_index("Generation"))

    st.success("Genetic Algorithm Optimization Completed Successfully 🎉")
