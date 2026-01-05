import streamlit as st
import random
import pandas as pd

# ======================================
# APP TITLE
# ======================================
st.title("Cinema Ticket Pricing Optimization using Genetic Algorithm")

# ======================================
# FILE UPLOAD
# ======================================
uploaded_file = st.file_uploader(
    "Upload your CSV file (e.g., cinema_ticket_sales.csv)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

# ======================================
# SAFE CSV LOADER (ENCODING FALLBACK)
# ======================================
def load_csv(file):
    try:
        # Try UTF-8 first
        return pd.read_csv(
            file,
            sep=",",
            engine="python",
            on_bad_lines="skip"
        )
    except UnicodeDecodeError:
        # Fallback for Excel / Windows CSV
        return pd.read_csv(
            file,
            sep=",",
            engine="python",
            on_bad_lines="skip",
            encoding="cp1252"
        )

try:
    df = load_csv(uploaded_file)
    st.success("Dataset successfully loaded")
    st.write("Dataset Preview:")
    st.dataframe(df.head())
    st.write(f"Total rows loaded: {len(df)}")
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
    st.error("Price column and tickets sold column must be different.")
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
    st.error("Invalid price range detected in the dataset.")
    st.stop()

# ======================================
# DEMAND ESTIMATION
# ======================================
def estimate_demand(price):
    closest_index = (df[price_col] - price).abs().idxmin()
    return df.loc[closest_index, sold_col]

# ======================================
# FITNESS FUNCTION
# ======================================
def fitness(price):
    return price * estimate_demand(price)

# ======================================
# GENETIC ALGORITHM PARAMETERS
# ======================================
st.sidebar.header("Genetic Algorithm Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 20, 100, 50)
GENERATIONS = st.sidebar.slider("Number of Generations", 20, 200, 100)
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
def initialize_population():
    return [random.uniform(PRICE_MIN, PRICE_MAX) for _ in range(POP_SIZE)]

def selection(population):
    tournament = random.sample(population, 3)
    return max(tournament, key=fitness)

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        return (parent1 + parent2) / 2
    return parent1

def mutation(price):
    if random.random() < MUTATION_RATE:
        return random.uniform(PRICE_MIN, PRICE_MAX)
    return price

# ======================================
# RUN GENETIC ALGORITHM
# ======================================
if st.button("Run Genetic Algorithm"):
    population = initialize_population()
    best_revenue_history = []
    best_price_history = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for generation in range(GENERATIONS):
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
        best_revenue_history.append(best_revenue)

        progress_bar.progress((generation + 1) / GENERATIONS)

        if generation % 10 == 0:
            status_text.text(
                f"Generation {generation} | Best Price: RM {best_price:.2f} | Revenue: RM {best_revenue:.2f}"
            )

    # ======================================
    # FINAL RESULTS
    # ======================================
    st.subheader("Final Optimization Results")

    st.metric("Optimal Ticket Price (RM)", f"{best_price:.2f}")
    st.metric("Estimated Tickets Sold", int(estimate_demand(best_price)))
    st.metric("Maximum Revenue (RM)", f"{best_revenue:.2f}")

    # ======================================
    # VISUALIZATION (STREAMLIT NATIVE)
    # ======================================
    st.subheader("Revenue Optimization Over Generations")
    st.line_chart(best_revenue_history)

    st.subheader("Ticket Price Evolution Over Generations")
    st.line_chart(best_price_history)
