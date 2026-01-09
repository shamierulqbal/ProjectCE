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

st.title("🎬 Cinema Ticket Pricing Optimization Using Genetic Algorithm (GA)")
st.write(
    """
    This app uses a **Genetic Algorithm (GA)** to identify the **optimal cinema ticket price**.
    You can select **Single Objective** (maximize revenue) or **Multi-Objective** (maximize revenue + balance price/demand).
    """
)

# ======================================================
# FILE UPLOAD
# ======================================================
uploaded_file = st.file_uploader("📂 Upload cinema ticket dataset (CSV)", type=["csv"])
if uploaded_file is None:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(uploaded_file)
st.success("Dataset loaded successfully ✅")
st.dataframe(df)

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
    st.error("Unable to auto-detect price or demand column.")
    st.stop()

# Force numeric
df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
df[sold_col] = pd.to_numeric(df[sold_col], errors="coerce")
df = df[[price_col, sold_col]].dropna()

st.success(f"🎫 Price Column: {price_col} | 👥 Demand Column: {sold_col}")

# ======================================================
# SIDEBAR: GA PARAMETERS & OBJECTIVE
# ======================================================
st.sidebar.header("⚙️ GA Parameters")
POP_SIZE = st.sidebar.slider("Population Size", 20, 200, 60)
GENERATIONS = st.sidebar.slider("Generations", 50, 300, 150)
CROSSOVER_RATE = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.8)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.05)
ELITISM_SIZE = st.sidebar.slider("Elitism Size", 1, 5, 2)

objective_type = st.sidebar.selectbox(
    "Select GA Objective",
    ["Single Objective (Maximize Revenue)", "Multi-Objective (Revenue & Price Balance)"]
)

if objective_type.startswith("Single"):
    st.sidebar.info("✅ Single objective: maximize revenue only")
else:
    st.sidebar.info("✅ Multi-objective: balance revenue and price")

# ======================================================
# PRICE RANGE
# ======================================================
PRICE_MIN = float(df[price_col].min())
PRICE_MAX = float(df[price_col].max())

# ======================================================
# DEMAND ESTIMATION
# ======================================================
def estimate_demand(price):
    idx = (df[price_col] - price).abs().idxmin()
    return float(df.loc[idx, sold_col])

# ======================================================
# FITNESS FUNCTION
# ======================================================
def fitness(price):
    demand = estimate_demand(price)
    if objective_type.startswith("Single"):
        return price * demand
    else:
        # Weighted sum: maximize revenue, penalize very high prices
        revenue = price * demand
        price_penalty = 0.1 * price  # weight adjustable
        return revenue - price_penalty

# ======================================================
# GA OPERATORS
# ======================================================
def init_population():
    return [random.uniform(PRICE_MIN, PRICE_MAX) for _ in range(POP_SIZE)]

def selection(pop):
    candidates = random.sample(pop, 3)
    return max(candidates, key=fitness)

def crossover(p1, p2):
    return (p1 + p2)/2 if random.random() < CROSSOVER_RATE else p1

def mutation(price):
    return random.uniform(PRICE_MIN, PRICE_MAX) if random.random() < MUTATION_RATE else price

# ======================================================
# RUN GA
# ======================================================
if st.button("🚀 Run Genetic Algorithm"):

    population = init_population()
    best_history = []

    for gen in range(GENERATIONS):
        population = sorted(population, key=fitness, reverse=True)
        new_pop = population[:ELITISM_SIZE]

        while len(new_pop) < POP_SIZE:
            p1 = selection(population)
            p2 = selection(population)
            child = mutation(crossover(p1, p2))
            new_pop.append(child)

        population = new_pop
        best_history.append(population[0])

    # ======================================================
    # TOP 3 SOLUTIONS
    # ======================================================
    population = sorted(population, key=fitness, reverse=True)
    top_solutions = population[:3]

    results_df = pd.DataFrame({
        "Rank": [1, 2, 3],
        "Ticket Price (RM)": [round(p,2) for p in top_solutions],
        "Estimated Demand": [int(estimate_demand(p)) for p in top_solutions],
        "Fitness Value (Revenue)": [round(fitness(p),2) for p in top_solutions]
    })

    best_price = top_solutions[0]

    # ======================================================
    # RESULT SUMMARY
    # ======================================================
    st.markdown("## 🏆 Best Solution")
    col1, col2, col3 = st.columns(3)
    col1.metric("🎫 Best Ticket Price", f"RM {best_price:.2f}")
    col2.metric("👥 Estimated Demand", int(estimate_demand(best_price)))
    col3.metric("💰 Fitness Value (Revenue)", f"RM {fitness(best_price):.2f}")

    # ======================================================
    # TOP 3 TABLE
    # ======================================================
    st.markdown("## 🥇 Top 3 GA Solutions")
    st.dataframe(results_df, use_container_width=True)

    # ======================================================
    # PRICE vs REVENUE CURVE
    # ======================================================
    st.markdown("## 📈 Ticket Price vs Revenue")
    price_range = np.linspace(PRICE_MIN, PRICE_MAX, 50)
    revenue_curve = [fitness(p) for p in price_range]

    revenue_df = pd.DataFrame({
        "Ticket Price": price_range,
        "Revenue": revenue_curve
    })

    st.line_chart(revenue_df.set_index("Ticket Price"))
    st.success(f"🔴 Maximum revenue occurs at RM {best_price:.2f}")

    # ======================================================
    # GA LEARNING CURVE
    # ======================================================
    st.markdown("## 🧬 Genetic Algorithm Learning Curve")
    learning_df = pd.DataFrame({
        "Generation": range(1, GENERATIONS+1),
        "Best Fitness": [fitness(p) for p in best_history]
    })
    st.line_chart(learning_df.set_index("Generation"))

    # ======================================================
    # INTERPRETATION
    # ======================================================
    st.markdown("## 🧠 Interpretation")
    st.write(
        f"""
        • The GA selects the ticket price with the **highest fitness value**.
        • Top 3 solutions show how fitness varies with price.
        • Learning curve indicates convergence over generations.
        • Multi-objective GA balances revenue with price to avoid very high ticket prices.
        """
    )
