import streamlit as st
import pandas as pd
import random

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("cinema_ticket_pricing_clean.csv")

df = load_data()

st.title("🎬 Optimizing Cinema Ticket Pricing using Genetic Algorithm")

st.write("This application optimizes cinema ticket prices to maximize revenue using a Genetic Algorithm.")

# -----------------------------
# USER PARAMETERS
# -----------------------------
st.sidebar.header("Genetic Algorithm Parameters")

population_size = st.sidebar.slider("Population Size", 10, 100, 30)
generations = st.sidebar.slider("Number of Generations", 10, 200, 50)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)

min_price = st.sidebar.number_input("Minimum Ticket Price", value=5.0)
max_price = st.sidebar.number_input("Maximum Ticket Price", value=30.0)

# -----------------------------
# FITNESS FUNCTION
# -----------------------------
def fitness(price):
    """
    Fitness = Total Revenue
    """
    revenue = 0
    for _, row in df.iterrows():
        revenue += price * row['number_of_person']
    return revenue

# -----------------------------
# GENETIC ALGORITHM
# -----------------------------
def genetic_algorithm():
    # Initialize population
    population = [random.uniform(min_price, max_price) for _ in range(population_size)]

    best_prices = []
    best_fitness = []

    for _ in range(generations):
        # Evaluate fitness
        scores = [(price, fitness(price)) for price in population]
        scores.sort(key=lambda x: x[1], reverse=True)

        best_prices.append(scores[0][0])
        best_fitness.append(scores[0][1])

        # Selection (top 50%)
        selected = scores[:population_size // 2]
        selected_prices = [p[0] for p in selected]

        # Crossover
        children = []
        while len(children) < population_size:
            p1, p2 = random.sample(selected_prices, 2)
            child = (p1 + p2) / 2
            children.append(child)

        # Mutation
        for i in range(len(children)):
            if random.random() < mutation_rate:
                children[i] += random.uniform(-2, 2)

        # Ensure price bounds
        population = [max(min_price, min(max_price, p)) for p in children]

    return best_prices, best_fitness

# -----------------------------
# RUN BUTTON
# -----------------------------
if st.button("Run Optimization"):
    prices, fitness_values = genetic_algorithm()

    st.success("Optimization Completed!")

    st.metric(
        label="Optimal Ticket Price (RM)",
        value=f"{prices[-1]:.2f}"
    )

    st.metric(
        label="Maximum Revenue",
        value=f"{fitness_values[-1]:,.2f}"
    )

    # -----------------------------
    # VISUALIZATION
    # -----------------------------
    chart_data = pd.DataFrame({
        "Generation": range(len(fitness_values)),
        "Revenue": fitness_values
    })

    st.line_chart(chart_data, x="Generation", y="Revenue")
