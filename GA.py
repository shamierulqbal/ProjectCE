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
    "This application applies a Genetic Algorithm (GA) to identify the optimal "
    "cinema ticket price that maximizes total revenue (fitness value)."
)

# ======================================================
# FILE UPLOAD
# ======================================================
uploaded_file = st.file_uploader("📂 Upload cinema ticket dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success("Dataset loaded successfully ✅")
st.dataframe(df, use_container_width=True)

# ======================================================
# COLUMN AUTO-DETECTION
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
    st.error("Ticket price or demand column not detected.")
    st.stop()

df = df[[price_col, sold_col]].dropna()

st.success(f"Ticket Price Column: {price_col}")
st.success(f"Demand Column: {sold_col}")

# ======================================================
# DATA AUGMENTATION
# ======================================================
if len(df) < 5:
    prices = np.linspace(df[price_col].min(), df[price_col].max(), 15)
    demand = np.interp(prices, df[price_col], df[sold_col])
    df = pd.DataFrame({price_col: prices, sold_col: demand.astype(int)})

# ======================================================
# GA SETUP
# ======================================================
PRICE_MIN = df[price_col].min()
PRICE_MAX = df[price_col].max()

def estimate_demand(price):
    idx = (df[price_col] - price).abs().idxmin()
    return df.loc[idx, sold_col]

def fitness(price):
    return price * estimate_demand(price)

# ======================================================
# GA PARAMETERS
# ======================================================
st.sidebar.header("⚙️ GA Parameters")
POP_SIZE = st.sidebar.slider("Population Size", 30, 200, 80)
GENERATIONS = st.sidebar.slider("Generations", 50, 300, 150)
CROSSOVER_RATE = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.8)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.05)
ELITISM_SIZE = st.sidebar.slider("Elitism Size", 1, 5, 2)

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

    progress = st.progress(0.0)

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
            "Ticket Price": best,
            "Demand": estimate_demand(best),
            "Fitness (Revenue)": fitness(best)
        })

        progress.progress((gen + 1) / GENERATIONS)

    result_df = pd.DataFrame(history)
    best_solution = result_df.iloc[result_df["Fitness (Revenue)"].idxmax()]

    # ======================================================
    # RESULT SUMMARY
    # ======================================================
    st.markdown("## 🏆 Best Solution Identified")

    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal Ticket Price", f"RM {best_solution['Ticket Price']:.2f}")
    c2.metric("Estimated Demand", int(best_solution["Demand"]))
    c3.metric("Fitness Value (Revenue)", f"RM {best_solution['Fitness (Revenue)']:.2f}")

    # ======================================================
    # TOP 5 SOLUTIONS TABLE
    # ======================================================
    st.markdown("## 🔝 Top 5 Best Ticket Prices (Based on Fitness Value)")

    top5 = (
        result_df
        .sort_values("Fitness (Revenue)", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )
    top5.index += 1
    st.dataframe(top5, use_container_width=True)

    # ======================================================
    # LEARNING CURVE
    # ======================================================
    st.markdown("## 📈 Genetic Algorithm Learning Curve")

    st.line_chart(
        result_df.set_index("Generation")["Fitness (Revenue)"]
    )

    # ======================================================
    # WHAT-IF REVENUE CURVE
    # ======================================================
    st.markdown("## 💰 What-if Analysis: Ticket Price vs Revenue")

    price_range = np.linspace(PRICE_MIN, PRICE_MAX, 100)
    revenue_curve = [p * estimate_demand(p) for p in price_range]

    revenue_df = pd.DataFrame({
        "Ticket Price": price_range,
        "Revenue": revenue_curve
    })

    st.line_chart(revenue_df, x="Ticket Price", y="Revenue")

    # ======================================================
    # INTERPRETATION
    # ======================================================
    avg_price = df[price_col].mean()
    improvement = (
        (best_solution["Fitness (Revenue)"] -
         (avg_price * df[sold_col].mean()))
        / (avg_price * df[sold_col].mean())
    ) * 100

    st.markdown("## 🧠 Interpretation")

    st.success(
        f"""
        • The **best solution** is selected based on the **highest fitness value**, 
        which represents maximum revenue.

        • The optimal ticket price (**RM {best_solution['Ticket Price']:.2f}**) produces 
        the **highest revenue among all evaluated solutions**.

        • Compared to the average pricing strategy, this solution improves revenue by 
        approximately **{improvement:.2f}%**.

        • This confirms that the Genetic Algorithm successfully identifies the most effective
        price–demand balance.
        """
    )
