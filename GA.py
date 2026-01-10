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
This system uses a Genetic Algorithm (GA) to find the optimal cinema ticket price
based on customer demand and revenue.
""")

# ======================================================
# DATA SOURCE OPTION
# ======================================================
st.sidebar.header("Dataset Options")
data_option = st.sidebar.radio(
    "Choose Dataset Source",
    ["Upload CSV", "Auto Generate Dataset"]
)

# ======================================================
# LOAD / GENERATE DATASET
# ======================================================
if data_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload cinema ticket dataset (CSV)", type=["csv"])
    if uploaded_file is None:
        st.info("Please upload a CSV file.")
        st.stop()
    df = pd.read_csv(uploaded_file)

else:
    st.sidebar.subheader("Auto Dataset Settings")
    min_price = st.sidebar.slider("Minimum Price (RM)", 5, 10, 8)
    max_price = st.sidebar.slider("Maximum Price (RM)", 30, 50, 35)
    step = st.sidebar.selectbox("Price Interval", [0.5, 1, 2])
    base_demand = st.sidebar.slider("Base Demand", 300, 600, 450)

    prices = np.arange(min_price, max_price + step, step)
    demand = np.maximum(
        0,
        base_demand - (prices * 10) + np.random.randint(-15, 15, len(prices))
    )

    df = pd.DataFrame({
        "Ticket_Price": prices,
        "Tickets_Sold": demand
    })

    st.success("Auto-generated dataset created")

# ======================================================
# COLUMN DETECTION
# ======================================================
price_keywords = ["price", "ticket"]
demand_keywords = ["sold", "demand", "attendance", "quantity"]

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
df = df[[price_col, demand_col]].dropna().sort_values(by=price_col)

st.markdown("### Dataset Preview")
st.dataframe(df.head(10))

# ======================================================
# GA PARAMETERS
# ======================================================
st.sidebar.header("GA Parameters")
POP_SIZE = st.sidebar.slider("Population Size", 20, 200, 100)
GENERATIONS = st.sidebar.slider("Generations", 50, 500, 200)
CROSSOVER_RATE = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.8)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.1)
ELITISM_SIZE = st.sidebar.slider("Elitism Size", 1, 10, 2)

objective_type = st.sidebar.selectbox(
    "Objective Function",
    ["Single Objective (Max Revenue)", "Multi Objective (Revenue + Demand)"]
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
    price = np.clip(price, PRICE_MIN, PRICE_MAX)
    return float(np.interp(price, price_arr, demand_arr))

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

    progress = st.progress(0)

    for gen in range(GENERATIONS):
        population = sorted(population, key=fitness_func, reverse=True)
        best_history.append(population[0])

        new_pop = population[:ELITISM_SIZE]
        while len(new_pop) < POP_SIZE:
            p1, p2 = selection(population, fitness_func), selection(population, fitness_func)
            child = mutation(crossover(p1, p2))
            new_pop.append(child)

        population = new_pop
        progress.progress((gen + 1) / GENERATIONS)

    best_price = max(population, key=fitness_func)
    best_demand = estimate_demand(best_price)
    best_revenue = best_price * best_demand

    # ======================================================
    # RESULTS
    # ======================================================
    st.markdown("## Optimization Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal Price", f"RM {best_price:.2f}")
    c2.metric("Estimated Demand", f"{int(best_demand)} customers")
    c3.metric("Revenue", f"RM {best_revenue:,.2f}")

    # ======================================================
    # VISUALIZATION
    # ======================================================
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Price vs Fitness**")
        prices = np.linspace(PRICE_MIN, PRICE_MAX, 100)
        fitness_vals = [fitness_func(p) for p in prices]
        st.line_chart(pd.DataFrame({
            "Price": prices,
            "Fitness": fitness_vals
        }).set_index("Price"))

    with col2:
        st.markdown("**GA Convergence Curve**")
        best_fit = [fitness_func(p) for p in best_history]
        st.line_chart(pd.DataFrame({
            "Generation": range(1, len(best_fit) + 1),
            "Best Fitness": best_fit
        }).set_index("Generation"))

    st.success("Genetic Algorithm Optimization Completed 🎉")
