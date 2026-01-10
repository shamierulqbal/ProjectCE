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
This application applies a Genetic Algorithm (GA) to optimize cinema ticket pricing
based on customer demand and revenue performance.
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
# COLUMN DETECTION
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
    best_history = []

    for _ in range(GENERATIONS):
        population = sorted(population, key=fitness_func, reverse=True)
        best_history.append(population[0])

        new_population = population[:ELITISM_SIZE]
        while len(new_population) < POP_SIZE:
            p1, p2 = selection(population, fitness_func), selection(population, fitness_func)
            child = mutation(crossover(p1, p2))
            new_population.append(child)

        population = new_population

    # ======================================================
    # FINAL RESULT
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
    # LOGICAL EXPLANATION OUTPUT
    # ======================================================
    if "Single" in objective_type:
        st.markdown("### 🔍 Interpretation (Single Objective)")
        st.write(
            f"""
            The Genetic Algorithm aims to **maximize total revenue** by evaluating different ticket prices.
            The optimal ticket price of **RM {best_price:.2f}** achieves the highest revenue by balancing
            ticket price and customer demand.

            Although lower prices attract more customers and higher prices increase ticket value,
            this price point provides the **maximum revenue of RM {best_revenue:,.2f}**
            with an estimated demand of **{int(best_demand)} customers**.
            """
        )

    else:
        st.markdown("### 🔍 Interpretation (Multi Objective)")
        st.write(
            f"""
            The multi-objective Genetic Algorithm optimizes cinema ticket pricing by **maximizing total revenue
            while balancing ticket price and customer demand using evolutionary optimization**.

            The selected ticket price of **RM {best_price:.2f}** represents a trade-off solution that maintains
            healthy customer attendance (**{int(best_demand)} customers**) while achieving a strong revenue
            performance of **RM {best_revenue:,.2f}**.

            This solution is suitable for cinemas aiming for **long-term sustainability rather than short-term profit maximization**.
            """
        )

    st.success("Genetic Algorithm Optimization Completed Successfully 🎉")
