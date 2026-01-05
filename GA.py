import streamlit as st
import random
import pandas as pd

# ======================================
# APP TITLE
# ======================================
st.title("🎬 Cinema Ticket Pricing Optimization using Genetic Algorithm")

# ======================================
# UPLOAD DATASET
# ======================================
uploaded_file = st.file_uploader(
    "Upload your CSV file (e.g., cinema_hall_ticket_sales.csv)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

# ======================================
# LOAD DATASET
# ======================================
try:
    df = pd.read_csv(uploaded_file, encoding="latin1")
    st.success("Dataset successfully loaded ✅")
    st.write("Dataset Preview:")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# ======================================
# COLUMN SELECTION
# ======================================
st.subheader("Select Dataset Columns")

price_col = st.selectbox("Select Ticket Price Column", df.columns)
sold_col = st.selectbox("Select Tickets Sold Column", df.columns)

if price_col == sold_col:
    st.error("Price column and Tickets Sold column must be different.")
    st.stop()

if not pd.api.types.is_numeric_dtype(df[price_col]) or not pd.api.types.is_numeric_dtype(df[sold_col]):
    st.error("Selected columns must contain numeric values.")
    st.stop()

# ======================================
# PRICE RANGE
# ======================================
PRICE_MIN = float(df[price_col].min())
PRICE_MAX = float(df[price_col].max())

if PRICE_MIN >= PRICE_MAX:
    st.error("Invalid price range in dataset.")
    st.stop()

# ======================================
# DEMAND ESTIMATION
# ======================================
def estimate_demand(price):
    idx = (df[price_col] - price).abs().idxmin()
    return df.loc[idx, sold_col]

# ======================================
# FITNESS FUNCTION
# ======================================
def fitness(price):
    return price * estimate_demand(price)

# ======================================
# GA PARAMETERS
# ======================================
st.sidebar.header("Genetic Algorithm Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 20, 100, 50)
GENERATIONS = st.sidebar.slider("Generations", 20, 200, 100)
CROSSOVER_RATE = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.8)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.05)
ELITISM_SIZE = st.sidebar.slider(
    "Elitism Size",
    1,
    min(5, POP_SIZE - 1),
    2
)

# ======================================
# GA FUNCTIONS
# ======================================
def init_population():
    return [random.uniform(PRICE_MIN, PRICE_MAX) for _ in range(POP_SIZE)]

def selection(population):
    tournament = random.sample(population, 3)
    return max(tournament, key=fitness)

def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        return (p1 + p2) / 2
    return p1

def mutation(price):
    if random.random() < MUTATION_RATE:
        return random.uniform(PRICE_MIN, PRICE_MAX)
    return price

# ======================================
# RUN GA
# ======================================
if st.button("🚀 Run Genetic Algorithm"):
    population = init_population()
    best_fitness_history = []
    best_price_history = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for gen in range(GENERATIONS):
        population.sort(key=fitness, reverse=True)

        # Elitism
        new_population = population[:ELITISM_SIZE]

        while len(new_population) < POP_SIZE:
            p1 = selection(population)
            p2 = selection(population)
            child = crossover(p1, p2)
            child = mutation(child)
            new_population.append(child)

        population = new_population

        best_price = population[0]
        best_revenue = fitness(best_price)

        best_price_history.append(best_price)
        best_fitness_history.append(best_revenue)

        progress_bar.progress((gen + 1) / GENERATIONS)

        if gen % 10 == 0:
            status_text.text(
                f"Generation {gen} | Best Price RM {best_price:.2f} | Revenue RM {best_revenue:.2f}"
            )

    # ======================================
    # FINAL RESULT
    # ======================================
    st.subheader("📊 Final Optimization Result")

    st.metric("Optimal Ticket Price (RM)", f"{best_price:.2f}")
    st.metric("Estimated Tickets Sold", int(estimate_demand(best_price)))
    st.metric("Maximum Revenue (RM)", f"{best_revenue:.2f}")

    # ======================================
    # VISUALIZATION (STREAMLIT NATIVE)
    # ======================================
    st.subheader("📈 Revenue Optimization Progress")
    st.line_chart(best_fitness_history)

    st.subheader("💰 Ticket Price Evolution")
    st.line_chart(best_price_history)
