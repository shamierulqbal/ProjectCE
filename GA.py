import streamlit as st
import random
import numpy as np

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Cinema Ticket Pricing Optimization",
    page_icon="🎬",
    layout="centered"
)

# ----------------------------------
# Demand Function
# ----------------------------------
def demand(price):
    base_demand = 300      # maximum customers
    sensitivity = 8        # demand reduction per RM
    return max(0, base_demand - sensitivity * price)

# ----------------------------------
# Fitness Function (Revenue)
# ----------------------------------
def fitness(price):
    return price * demand(price)

# ----------------------------------
# Genetic Algorithm
# ----------------------------------
def genetic_algorithm(pop_size, generations, mutation_rate):
    population = [random.uniform(5, 30) for _ in range(pop_size)]
    best_fitness_history = []

    for _ in range(generations):
        fitness_scores = [fitness(p) for p in population]
        best_fitness_history.append(max(fitness_scores))

        # Tournament Selection
        selected = []
        for _ in range(pop_size):
            a, b = random.sample(population, 2)
            selected.append(a if fitness(a) > fitness(b) else b)

        # Crossover
        offspring = []
        for i in range(0, pop_size, 2):
            parent1 = selected[i]
            parent2 = selected[min(i + 1, pop_size - 1)]
            alpha = random.random()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = alpha * parent2 + (1 - alpha) * parent1
            offspring.extend([child1, child2])

        # Mutation
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                offspring[i] += random.uniform(-2, 2)

        population = np.clip(offspring, 5, 30)

    best_price = max(population, key=fitness)
    return best_price, fitness(best_price), best_fitness_history

# ----------------------------------
# Streamlit UI
# ----------------------------------
st.title("🎬 Optimizing Cinema Ticket Pricing")
st.markdown("### Using Genetic Algorithm (GA)")
st.write(
    "This application uses a **Genetic Algorithm** to find the optimal cinema "
    "ticket price that **maximizes revenue** by balancing ticket price and customer demand."
)

# Sidebar parameters
st.sidebar.header("🧬 Genetic Algorithm Parameters")

population_size = st.sidebar.slider("Population Size", 20, 200, 80)
generations = st.sidebar.slider("Number of Generations", 20, 200, 100)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)

# Run GA
if st.button("🚀 Run Optimization"):
    with st.spinner("Running Genetic Algorithm..."):
        best_price, best_revenue, history = genetic_algorithm(
            population_size, generations, mutation_rate
        )

    st.success("Optimization Completed Successfully!")

    # Results
    col1, col2, col3 = st.columns(3)

    col1.metric("🎟️ Optimal Ticket Price (RM)", f"{best_price:.2f}")
    col2.metric("👥 Expected Demand", int(demand(best_price)))
    col3.metric("💰 Maximum Revenue (RM)", f"{best_revenue:.2f}")

    # Convergence Plot
    st.subheader("📈 GA Convergence Curve")
    st.line_chart(history)

    # Price vs Revenue Curve
    st.subheader("✨ Ticket Price vs Revenue")
    prices = np.linspace(5, 30, 100)
    revenues = [fitness(p) for p in prices]

    chart_data = {
        "Price (RM)": prices,
        "Revenue (RM)": revenues
    }

    st.line_chart(chart_data)

    st.info(
        "📌 The peak of the curve represents the optimal pricing point discovered "
        "by the Genetic Algorithm."
    )

# Footer
st.markdown("---")
st.caption("📊 Evolutionary Algorithm | Genetic Algorithm | Streamlit App")
