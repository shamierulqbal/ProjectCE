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
st.write(
    """
    This application applies a Genetic Algorithm (GA) to determine the optimal cinema ticket price.
    Choose between maximizing pure revenue or balancing revenue with customer demand.
    """
)

# ======================================================
# FILE UPLOAD
# ======================================================
uploaded_file = st.file_uploader("Upload cinema ticket dataset (CSV)", type=["csv"])
if uploaded_file is None:
    st.info("Please upload a CSV file to proceed. Ensure it contains 'Price' and 'Demand/Sold' columns.")
    st.stop()

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(uploaded_file)

# ======================================================
# AUTO COLUMN DETECTION
# ======================================================
price_keywords = ["price", "ticket"]
demand_keywords = ["person", "sold", "demand", "quantity", "attendance"]

def find_column(keywords):
    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            return col
    return None

price_col = find_column(price_keywords)
sold_col = find_column(demand_keywords)

if price_col is None or sold_col is None:
    st.error("Unable to auto-detect price or demand column. Please rename your CSV columns.")
    st.stop()

# ======================================================
# CLEAN DATA
# ======================================================
df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
df[sold_col] = pd.to_numeric(df[sold_col], errors="coerce")
df = df[[price_col, sold_col]].dropna().sort_values(by=price_col).reset_index(drop=True)

if len(df) == 0:
    st.error("Dataset has no valid numeric data.")
    st.stop()

st.success(f"Detected Price Column: **{price_col}** | Demand Column: **{sold_col}**")
st.dataframe(df.head())

# ======================================================
# SIDEBAR: GA PARAMETERS & OBJECTIVE
# ======================================================
st.sidebar.header("GA Parameters")
POP_SIZE = st.sidebar.slider("Population Size", 20, 200, 100)
GENERATIONS = st.sidebar.slider("Generations", 50, 500, 200)
CROSSOVER_RATE = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.8)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.1)
ELITISM_SIZE = st.sidebar.slider("Elitism Size", 1, 10, 2)

st.sidebar.markdown("---")
objective_type = st.sidebar.selectbox(
    "Select GA Objective",
    ["Single Objective (Maximize Revenue)", "Multi-Objective (Revenue & Demand Balance)"]
)

# Initialize weights
w1, w2 = 1.0, 0.0
if "Multi-Objective" in objective_type:
    st.sidebar.subheader("Multi-Objective Weights")
    w1 = st.sidebar.slider("w1 (Revenue Weight)", 0.0, 1.0, 0.7)
    w2 = st.sidebar.slider("w2 (Demand Weight)", 0.0, 1.0, 0.3)
    
    if round(w1 + w2, 2) != 1.0:
        st.sidebar.warning(f"Weights sum to {round(w1+w2,2)}. (Ideal: 1.0)")

# ======================================================
# PRICE RANGE & INTERPOLATION
# ======================================================
PRICE_MIN = float(df[price_col].min())
PRICE_MAX = float(df[price_col].max())

price_arr = df[price_col].values
demand_arr = df[sold_col].values

def estimate_demand(price):
    price = np.clip(price, PRICE_MIN, PRICE_MAX)
    return float(np.interp(price, price_arr, demand_arr))

def fitness(price):
    demand = estimate_demand(price)
    revenue = price * demand
    
    if objective_type.startswith("Single"):
        return revenue
    else:
        # Multi-Objective: (w1 * Revenue) + (w2 * Demand)
        return (w1 * revenue) + (w2 * demand)

# ======================================================
# GA OPERATORS
# ======================================================
def init_population():
    return [random.uniform(PRICE_MIN, PRICE_MAX) for _ in range(POP_SIZE)]

def selection(pop):
    candidates = random.sample(pop, min(3, len(pop)))
    return max(candidates, key=fitness)

def crossover(p1, p2):
    return (p1 + p2)/2 if random.random() < CROSSOVER_RATE else p1

def mutation(price):
    if random.random() < MUTATION_RATE:
        # Small mutation range around current price
        return np.clip(price + random.uniform(-2, 2), PRICE_MIN, PRICE_MAX)
    return price

# ======================================================
# RUN GA
# ======================================================
if st.button("Run Genetic Algorithm"):
    population = init_population()
    best_history = []
    
    progress_bar = st.progress(0)

    for gen in range(GENERATIONS):
        # Sort by fitness (Descending)
        population = sorted(population, key=fitness, reverse=True)
        
        # Keep track of best individual
        best_history.append(population[0])
        
        # Create new population using Elitism
        new_pop = population[:ELITISM_SIZE]
        
        while len(new_pop) < POP_SIZE:
            p1 = selection(population)
            p2 = selection(population)
            child = crossover(p1, p2)
            child = mutation(child)
            new_pop.append(child)
            
        population = new_pop
        
        # Update progress
        progress_bar.progress((gen + 1) / GENERATIONS)

    # ======================================================
    # FINAL RESULTS
    # ======================================================
    best_price = sorted(population, key=fitness, reverse=True)[0]
    final_demand = estimate_demand(best_price)
    final_revenue = best_price * final_demand

    st.markdown("### Optimization Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Optimal Ticket Price", f"RM {best_price:.2f}")
    col2.metric("Estimated Attendance", f"{int(final_demand)} pax")
    col3.metric("Total Revenue", f"RM {final_revenue:,.2f}")

    # ======================================================
    # CHARTS
    # ======================================================
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Price vs. Fitness Curve**")
        p_plot = np.linspace(PRICE_MIN, PRICE_MAX, 100)
        f_plot = [fitness(p) for p in p_plot]
        chart_data = pd.DataFrame({"Price": p_plot, "Fitness Score": f_plot})
        st.line_chart(chart_data.set_index("Price"))

    with c2:
        st.markdown("**GA Learning Curve (Convergence)**")
        learning_data = pd.DataFrame({
            "Generation": range(len(best_history)),
            "Best Fitness": [fitness(p) for p in best_history]
        })
        st.line_chart(learning_data.set_index("Generation"))

    # ======================================================
    # TOP SOLUTIONS TABLE
    # ======================================================
    st.markdown("### Top 5 Candidate Solutions")
    top_5 = sorted(list(set(population)), key=fitness, reverse=True)[:5]
    top_df = pd.DataFrame({
        "Rank": range(1, len(top_5) + 1),
        "Ticket Price (RM)": [round(p, 2) for p in top_5],
        "Est. Demand": [int(estimate_demand(p)) for p in top_5],
        "Est. Revenue (RM)": [round(p * estimate_demand(p), 2) for p in top_5],
        "Total Fitness": [round(fitness(p), 2) for p in top_5]
    })
    st.table(top_df)

    # ======================================================
    # STRATEGY SUMMARY
    # ======================================================
    st.info(f"""
    **Analysis Summary:**
    - **Objective:** {objective_type}
    - **Weighting:** Revenue ($w_1$) = {w1}, Demand ($w_2$) = {w2}
    - The Genetic Algorithm suggests a price of **RM {best_price:.2f}** to achieve the best balance between profit and occupancy.
    """)
