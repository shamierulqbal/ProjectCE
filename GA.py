import streamlit as st
import pandas as pd
import numpy as np
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Cinema Ticket Pricing Optimization (GA)",
    layout="wide"
)

st.title("Cinema Ticket Pricing Optimization Using Genetic Algorithm")
st.write("""
This application applies a Genetic Algorithm (GA) to determine the optimal cinema ticket price
by maximizing revenue and balancing customer demand.
""")

# ======================================================
# FILE UPLOAD
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
demand_keywords = ["person", "sold", "demand", "quantity", "attendance", "number"]

def find_column(keywords):
    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            return col
    return None

price_col = find_column(price_keywords)
demand_col = find_column(demand_keywords)

if price_col is None or demand_col is None:
    st.error("Price or Demand column not detected.")
    st.stop()

# ======================================================
# CLEAN DATA
# ======================================================
df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
df[demand_col] = pd.to_numeric(df[demand_col], errors="coerce")
df = df[[price_col, demand_col]].dropna().sort_values(by=price_col).reset_index(drop=True)

st.success(f"Detected columns → Price: {price_col}, Demand: {demand_col}")
st.dataframe(df.head(10))

# ======================================================
# SIDEBAR – GA PARAMETERS
# ======================================================
st.sidebar.header("Genetic Algorithm Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 30, 200, 100)
GENERATIONS = st.sidebar.slider("Generations", 50, 500, 200)
CROSSOVER_RATE = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.8)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.1)
ELITISM_SIZE = st.sidebar.slider("Elitism Size", 1, 10, 2)

objective_type = st.sidebar.selectbox(
    "Optimization Objective",
    ["Single Objective (Maximize Revenue)",
     "Multi Objective (Revenue + Demand Balance)"]
)

w1, w2 = 1.0, 0.0
if "Multi" in objective_type:
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
    return float(np.interp(
        np.clip(price, PRICE_MIN, PRICE_MAX),
        price_arr,
        demand_arr
    ))

def fitness_single(price):
    return price * estimate_demand(price)

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

    best_fitness_history = []
    avg_fitness_history = []

    for gen in range(GENERATIONS):

        fitness_values = [fitness_func(p) for p in population]

        best_fitness_history.append(max(fitness_values))
        avg_fitness_history.append(np.mean(fitness_values))

        # elitism
        population_sorted = [
            p for _, p in sorted(
                zip(fitness_values, population),
                reverse=True
            )
        ]

        new_population = population_sorted[:ELITISM_SIZE]

        while len(new_population) < POP_SIZE:
            p1 = selection(population_sorted, fitness_func)
            p2 = selection(population_sorted, fitness_func)
            child = crossover(p1, p2)
            child = mutation(child)
            new_population.append(child)

        population = new_population

    # ======================================================
    # FINAL RESULTS
    # ======================================================
    best_price = max(population, key=fitness_func)
    best_demand = estimate_demand(best_price)
    best_revenue = best_price * best_demand

    st.markdown("## Optimization Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal Ticket Price", f"RM {best_price:.2f}")
    c2.metric("Estimated Demand", f"{int(best_demand)} customers")
    c3.metric("Total Revenue", f"RM {best_revenue:,.2f}")

    # ======================================================
    # GA LEARNING CURVE
    # ======================================================
    st.markdown("## GA Learning Curve")

    st.line_chart(pd.DataFrame({
        "Best Fitness": best_fitness_history,
        "Average Fitness": avg_fitness_history
    }))

    st.success("Genetic Algorithm Optimization Completed Successfully ")
