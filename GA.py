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
st.write("This app finds the optimal cinema ticket price that maximizes revenue using a Genetic Algorithm.")

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
except UnicodeDecodeError:
    df = pd.read_csv(uploaded_file, encoding="latin1")
except Exception:
    df = pd.read_csv(uploaded_file, engine="python", on_bad_lines="skip")

st.success("Dataset loaded successfully ✅")
st.dataframe(df)

# ======================================================
# COLUMN SELECTION
# ======================================================
st.subheader("📌 Select Columns")

columns = df.columns.tolist()
price_col = st.selectbox("Ticket Price Column", columns)
sold_col = st.selectbox("Tickets Sold Column", columns)

if price_col == sold_col:
    st.error("Columns must be different.")
    st.stop()

if not pd.api.types.is_numeric_dtype(df[price_col]) or not pd.api.types.is_numeric_dtype(df[sold_col]):
    st.error("Selected columns must be numeric.")
    st.stop()

# ======================================================
# DATA AUGMENTATION (IF DATA TOO SMALL)
# ======================================================
st.subheader("🧪 Data Augmentation")

if len(df) < 5:
    st.warning("Dataset too small. Automatically generating additional data based on existing trend.")

    price_min = df[price_col].min()
    price_max = df[price_col].max()

    # Generate new prices
    new_prices = np.linspace(price_min, price_max, 10)

    # Estimate demand using linear interpolation
    new_demand = np.interp(
        new_prices,
        df[price_col],
        df[sold_col]
    )

    augmented_df = pd.DataFrame({
        price_col: new_prices,
        sold_col: new_demand.astype(int)
    })

    df = augmented_df
    st.success("Data successfully expanded ✅")
    st.dataframe(df)

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
st.sidebar.header("⚙️ GA Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 20, 200, 60)
GENERATIONS = st.sidebar.slider("Generations", 50, 300, 150)
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
        best_prices.append(best_price)
        best_revenues.append(fitness(best_price))
        progress.progress((gen + 1) / GENERATIONS)

    # ======================================================
    # RESULT
    # ======================================================
    st.subheader("🏆 Optimization Result")

    st.metric("Optimal Ticket Price", f"RM {best_price:.2f}")
    st.metric("Estimated Tickets Sold", int(estimate_demand(best_price)))
    st.metric("Maximum Revenue", f"RM {fitness(best_price):.2f}")

    st.subheader("📈 Optimization Progress")
    st.line_chart(pd.DataFrame({
        "Revenue": best_revenues,
        "Ticket Price": best_prices
    }))

    st.success("🎉 Optimization completed successfully!")
