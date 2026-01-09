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

st.title("🎬 Optimizing Cinema Ticket Pricing Using Genetic Algorithm")
st.write(
    """
    This application uses a **Genetic Algorithm (GA)** to determine the **optimal cinema ticket price**
    that maximizes total revenue based on historical ticket demand.
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
except Exception:
    df = pd.read_csv(uploaded_file, engine="python")

st.success("Dataset loaded successfully ✅")
st.dataframe(df, use_container_width=True)

# ======================================================
# COLUMN AUTO-DETECTION (PRICE & PERSON ONLY)
# ======================================================
st.subheader("📌 Ticket Price & Demand Columns (Auto Detected)")

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
    st.error("Unable to detect ticket price or number of persons column.")
    st.stop()

st.success(f"🎫 Ticket Price Column: **{price_col}**")
st.success(f"👥 Number of Persons Column: **{sold_col}**")

df = df[[price_col, sold_col]].dropna()

# ======================================================
# DATA AUGMENTATION (IF DATA TOO SMALL)
# ======================================================
if len(df) < 5:
    st.warning("Dataset too small. Generating synthetic data for GA stability.")

    prices = np.linspace(df[price_col].min(), df[price_col].max(), 15)
    demand = np.interp(prices, df[price_col], df[sold_col])

    df = pd.DataFrame({
        price_col: prices,
        sold_col: demand.astype(int)
    })

    st.success("Data augmentation completed.")
    st.dataframe(df, use_container_width=True)

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
    return max(random.sample(pop, 3), key=fitness)

def crossover(p1, p2):
    return (p1 + p2) / 2 if random.random() < CROSSOVER_RATE else p1

def mutation(price):
    return random.uniform(PRICE_MIN, PRICE_MAX) if random.random() < MUTATION_RATE else price

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
            child = mutation(crossover(selection(population), selection(population)))
            new_population.append(child)

        population = new_population
        best_prices.append(population[0])
        best_revenues.append(fitness(population[0]))

        progress.progress((gen + 1) / GENERATIONS)

    best_price = best_prices[-1]

    # ======================================================
    # RESULT SUMMARY
    # ======================================================
    st.markdown("## 🏆 Optimization Result Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal Ticket Price", f"RM {best_price:.2f}")
    c2.metric("Estimated Demand", int(estimate_demand(best_price)))
    c3.metric("Maximum Revenue", f"RM {fitness(best_price):.2f}")

    # ======================================================
    # PLOT 1: PRICE vs DEMAND
    # ======================================================
    st.markdown("## 📊 Ticket Price vs Number of Persons")

    st.scatter_chart(df, x=price_col, y=sold_col)

    st.caption(
        "This plot shows the inverse relationship between ticket price and customer demand. "
        "As ticket prices increase, the number of persons generally decreases."
    )

    # ======================================================
    # PLOT 2: PRICE vs REVENUE (WHAT-IF ANALYSIS)
    # ======================================================
    st.markdown("## 💰 Ticket Price vs Revenue (What-if Analysis)")

    price_range = np.linspace(PRICE_MIN, PRICE_MAX, 100)
    revenue_curve = [p * estimate_demand(p) for p in price_range]

    revenue_df = pd.DataFrame({
        "Ticket Price": price_range,
        "Revenue": revenue_curve
    })

    st.line_chart(revenue_df, x="Ticket Price", y="Revenue")

    st.caption(
        "This curve represents a what-if analysis of revenue for different ticket prices. "
        "The peak of the curve indicates the price that generates the highest revenue."
    )

    # ======================================================
    # PLOT 3: GA LEARNING CURVE
    # ======================================================
    st.markdown("## 🧬 Genetic Algorithm Learning Curve")

    ga_df = pd.DataFrame({
        "Generation": range(1, GENERATIONS + 1),
        "Best Revenue": best_revenues
    })

    st.line_chart(ga_df, x="Generation", y="Best Revenue")

    st.caption(
        "The learning curve shows how the best revenue improves over generations. "
        "Early rapid improvement indicates exploration, while later stabilization indicates convergence."
    )

    # ======================================================
    # DECISION INTERPRETATION
    # ======================================================
    st.markdown("## 🧠 Decision Interpretation")

    avg_price = df[price_col].mean()
    relation = "higher" if best_price > avg_price else "lower"

    st.success(
        f"""
        **Interpretation:**  
        The Genetic Algorithm recommends a ticket price that is **{relation} than the historical average price**.

        This indicates that maximizing revenue does not necessarily mean selling more tickets, 
        but finding a balance between **price per ticket** and **customer demand**.

        The what-if revenue curve further supports this result by showing that prices beyond the optimal point 
        reduce total revenue due to significant drops in demand.
        """
    )
