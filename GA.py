import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# ======================================================
# APP CONFIG
# ======================================================
st.set_page_config(
    page_title="Cinema Ticket Pricing Optimization (GA)",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Optimizing Cinema Ticket Pricing Using Genetic Algorithm")
st.markdown("This application finds the **optimal ticket price** that maximizes **revenue** using a **Genetic Algorithm**.")

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
# LOAD DATA (ROBUST)
# ======================================================
try:
    df = pd.read_csv(uploaded_file)
except UnicodeDecodeError:
    df = pd.read_csv(uploaded_file, encoding="latin1")
except Exception:
    df = pd.read_csv(uploaded_file, engine="python", on_bad_lines="skip")

st.success("Dataset loaded successfully ✅")
st.dataframe(df.head())

# ======================================================
# COLUMN SELECTION
# ======================================================
st.subheader("📌 Select Relevant Columns")

cols = df.columns.tolist()

price_col = st.selectbox("Select Ticket Price Column", cols)
sold_col = st.selectbox("Select Tickets Sold Column", cols)

if price_col == sold_col:
    st.error("Price column and Tickets Sold column must be different.")
    st.stop()

if not pd.api.types.is_numeric_dtype(df[price_col]) or not pd.api.types.is_numeric_dtype(df[sold_col]):
    st.error("Selected columns must be numeric.")
    st.stop()

# ======================================================
# PRICE RANGE
# ======================================================
PRICE_MIN = float(df[price_col].min())
PRICE_MAX = float(df[price_col].max())

if PRICE_MIN >= PRICE_MAX:
    st.error("Invalid price range in dataset.")
    st.stop()

# ======================================================
# DEMAND ESTIMATION
# ======================================================
def estimate_demand(price):
    idx = (df[price_col] - price).abs().idxmin()
    return df.loc[idx, sold_col]

# ======================================================
# FITNESS FUNCTION (REVENUE)
# ======================================================
def fitness(price):
    return price * estimate_demand(price)

# ======================================================
# SIDEBAR – GA PARAMETERS
# ======================================================
st.sidebar.header("⚙️ Genetic Algorithm Settings")

POP_SIZE = st.sidebar.slider("Population Size", 20, 200, 60)
GENERATIONS = st.sidebar.slider("Generations", 50, 500, 150)
CROSSOVER_RATE = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.8)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.05)
ELITISM_SIZE = st.sidebar.slider("Elitism Size", 1, 5, 2)

# ======================================================
# GA FUNCTIONS
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
# RUN GA
# ======================================================
if st.button("🚀 Run Genetic Algorithm"):
    population = init_population()

    best_prices = []
    best_revenues = []

    progress = st.progress(0.0)
    status = st.empty()

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
        best_rev = fitness(best_price)

        best_prices.append(best_price)
        best_revenues.append(best_rev)

        progress.progress((gen + 1) / GENERATIONS)
        if gen % 10 == 0:
            status.text(
                f"Generation {gen} | Best Price: RM {best_price:.2f} | Revenue: RM {best_rev:.2f}"
            )

    # ======================================================
    # FINAL RESULT
    # ======================================================
    st.subheader("🏆 Optimization Result")

    col1, col2, col3 = st.columns(3)
    col1.metric("Optimal Ticket Price", f"RM {best_price:.2f}")
    col2.metric("Estimated Tickets Sold", int(estimate_demand(best_price)))
    col3.metric("Maximum Revenue", f"RM {best_rev:.2f}")

    # ======================================================
    # VISUALIZATION
    # ======================================================
    st.subheader("📈 Optimization Progress")

    fig1, ax1 = plt.subplots()
    ax1.plot(best_revenues)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Revenue (RM)")
    ax1.set_title("Revenue Optimization")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(best_prices)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Ticket Price (RM)")
    ax2.set_title("Ticket Price Evolution")
    st.pyplot(fig2)

    st.success("🎉 Optimization completed successfully!")
