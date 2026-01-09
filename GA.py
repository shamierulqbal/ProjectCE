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
    This application uses a **Genetic Algorithm (GA)** to identify the **optimal cinema ticket price**
    that maximizes total revenue based on historical demand data.
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
except UnicodeDecodeError:
    df = pd.read_csv(uploaded_file, encoding="latin1")
except Exception:
    df = pd.read_csv(uploaded_file, engine="python", on_bad_lines="skip")

st.success("Dataset loaded successfully ✅")
st.dataframe(df)

# ======================================================
# COLUMN SELECTION (AUTO-DETECT & RESTRICTED)
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
    st.error("❌ Unable to detect required columns (ticket price / number of persons).")
    st.stop()

if not pd.api.types.is_numeric_dtype(df[price_col]) or not pd.api.types.is_numeric_dtype(df[sold_col]):
    st.error("Detected columns must be numeric.")
    st.stop()

st.success(f"🎫 Ticket Price Column: **{price_col}**")
st.success(f"👥 Number of Person Column: **{sold_col}**")

# ======================================================
# DATA AUGMENTATION (IF DATA TOO SMALL)
# ======================================================
st.subheader("🧪 Data Preparation")

if len(df) < 5:
    st.warning("Dataset too small. Generating additional synthetic data for GA stability.")

    price_min = df[price_col].min()
    price_max = df[price_col].max()

    new_prices = np.linspace(price_min, price_max, 10)
    new_demand = np.interp(new_prices, df[price_col], df[sold_col])

    df = pd.DataFrame({
        price_col: new_prices,
        sold_col: new_demand.astype(int)
    })

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
# FITNESS FUNCTION (REVENUE)
# ======================================================
def fitness(price):
    return price * estimate_demand(price)

# ======================================================
# GA PARAMETERS
# ======================================================
st.sidebar.header("⚙️ Genetic Algorithm Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 20, 200, 60)
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
    # RESULT DASHBOARD
    # ======================================================
    st.markdown("## 🏆 Optimization Result Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("🎫 Optimal Ticket Price", f"RM {best_price:.2f}")
    col2.metric("👥 Estimated Tickets Sold", int(estimate_demand(best_price)))
    col3.metric("💰 Maximum Revenue", f"RM {fitness(best_price):.2f}")

    # ======================================================
    # REAL DATA VISUALIZATION
    # ======================================================
    st.markdown("## 📊 Ticket Price vs Demand (Real Data)")

    plot_df = df.copy()
    plot_df["Revenue"] = plot_df[price_col] * plot_df[sold_col]

    st.scatter_chart(plot_df, x=price_col, y=sold_col)

    st.info(
        f"🔴 **Optimal price identified by GA: RM {best_price:.2f}**"
    )

    # ======================================================
    # GA CONVERGENCE
    # ======================================================
    st.markdown("## 📈 Genetic Algorithm Learning Curve")

    conv_df = pd.DataFrame({
        "Generation": range(1, GENERATIONS + 1),
        "Best Revenue": best_revenues,
        "Best Price": best_prices
    })

    st.line_chart(conv_df.set_index("Generation")[["Best Revenue"]])

    # ======================================================
    # DECISION INSIGHT
    # ======================================================
    st.markdown("## 🧠 Optimization Insight")

    avg_price = df[price_col].mean()
    price_relation = "higher" if best_price > avg_price else "lower"

    st.success(
        f"""
        📌 **Key Insight**  
        The Genetic Algorithm suggests setting the ticket price **{price_relation} than the average historical price**
        to achieve maximum revenue.

        💡 Although higher prices may reduce demand, the **increase in price per ticket compensates for the loss in volume**,
        resulting in higher overall revenue.

        🧬 This demonstrates GA's ability to effectively balance the **price–demand trade-off** using evolutionary optimization.
        """
    )

    st.balloons()
