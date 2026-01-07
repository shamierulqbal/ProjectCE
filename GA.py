import streamlit as st
import random
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Demand Function (Realistic)
# -------------------------------
def demand(price):
    base_demand = 300
    sensitivity = 8
    return max(0, base_demand - sensitivity * price)

# -------------------------------
# Fitness Function
# -------------------------------
def fitness(price):
    return price * demand(price)

# -------------------------------
# Genetic Algorithm
# -------------------------------
def genetic_algorithm(pop_size, generations, mutation_rate):
    population = [random.uniform(5, 30) for _ in range(pop_size)]
    best_fitness_history = []

    for _ in range(generations):
        fitness_scores = [fitness(p) for p in population]

        # Save best fitness
        best_fitness_history.append(max(fitness_scores))

        # Selection (Tournament)
        selected = []
        for _ in range(pop_size):
            i, j = random.sample(range(pop_size), 2)
            selected.append(population[i] if fitness(population[i]) > fitness(population[j]) else population[j])

        # Crossover
        offspring = []
        for i in range(0, pop_size, 2):
            p1, p2 = selected[i], selected[min(i+1, pop_size-1)]
            alpha = random.random()
            child1 = alpha * p1 + (1 - alpha) * p2
            child2 = alpha * p2 + (1 - alpha) * p1
            offspring.extend([child1, child2])

        # Mutation
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                offspring[i] += random.uniform(-2, 2)

        population = np.clip(offspring, 5, 30)

    best_price = max(population, key=fitness)
    return best_price, fitness(best_price), best_fitness_history

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎬 Optimizing Cinema Ticket Pricing using Genetic Algorithm")
st.markdown("Adjust GA parameters and find the **optimal ticket price** that maximizes revenue.")

# Sidebar controls
st.sidebar.header("🧬 GA Parameters")
pop_size = st.sidebar.slider("Population Size", 20, 200, 80)
generations = st.sidebar.slider("Generations", 20, 200, 100)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)

# Run button
if st.button("🚀 Run Optimization"):
    best_price, best_revenue, history = genetic_algorithm(pop_size, generations, mutation_rate)

    st.success("Optimization Completed!")

    # Display results
    col1, col2, col3 = st.columns(3)
    col1.metric("🎟️ Best Ticket Price (RM)", f"{best_price:.2f}")
    col2.metric("👥 Expected Demand", int(demand(best_price)))
    col3.metric("💰 Maximum Revenue (RM)", f"{best_revenue:.2f}")

    # Fitness convergence plot
    st.subheader("📈 Fitness Convergence")
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Revenue")
    ax.set_title("GA Convergence Curve")
    st.pyplot(fig)

    # Wow factor: Price vs Revenue Curve
    st.subheader("✨ Price vs Revenue Curve")
    prices = np.linspace(5, 30, 100)
    revenues = [fitness(p) for p in prices]

    fig2, ax2 = plt.subplots()
    ax2.plot(prices, revenues)
    ax2.axvline(best_price, linestyle="--")
    ax2.set_xlabel("Ticket Price (RM)")
    ax2.set_ylabel("Revenue (RM)")
    ax2.set_title("Revenue Landscape")
    st.pyplot(fig2)
