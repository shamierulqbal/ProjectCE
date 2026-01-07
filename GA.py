import streamlit as st
import numpy as np
import random
import pandas as pd
from io import StringIO

# Streamlit app title
st.title("Cinema Ticket Pricing Optimizer using Genetic Algorithm")

# Sidebar for parameter settings
st.sidebar.header("Set Genetic Algorithm Parameters")
population_size = st.sidebar.slider("Population Size", min_value=10, max_value=200, value=50, step=10)
generations = st.sidebar.slider("Number of Generations", min_value=10, max_value=500, value=100, step=10)
mutation_rate = st.sidebar.slider("Mutation Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
crossover_rate = st.sidebar.slider("Crossover Rate", min_value=0.5, max_value=1.0, value=0.8, step=0.05)

# Default model parameters
base_demand = 1000
sensitivity = 10
max_price = 50

# Dataset upload section
st.header("Upload Dataset")
st.write("Upload a CSV file with 'price' and 'demand' columns to fit a custom demand model. Defaults will be used if no file is uploaded.")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

data = None
if uploaded_file is not None:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data.head())
    
    # Fit linear model if required columns exist
    if 'price' in data.columns and 'demand' in data.columns:
        # Simple linear fit: demand = intercept + slope * price
        # We expect slope to be negative (higher price -> lower demand)
        prices = data['price'].values
        demands = data['demand'].values
        if len(prices) > 1:  # Need at least 2 points for fit
            slope, intercept = np.polyfit(prices, demands, 1)
            sensitivity = -slope if slope < 0 else sensitivity  # Ensure positive sensitivity
            base_demand = intercept
            st.success(f"Fitted Demand Model: Base Demand = {base_demand:.2f}, Sensitivity = {sensitivity:.2f}")
        else:
            st.warning("Not enough data points for fitting. Using defaults.")
    else:
        st.error("CSV must have 'price' and 'demand' columns. Using defaults.")

# Define the fitness function (objective: maximize revenue based on price and demand)
# Demand model: demand = base_demand - sensitivity * price
def fitness(individual, base_demand, sensitivity, max_price):
    price = individual[0]  # Individual is a list with one gene: ticket price
    if price < 0 or price > max_price:
        return 0  # Penalize invalid prices
    demand = max(0, base_demand - sensitivity * price)
    revenue = price * demand
    return revenue

# Initialize population
def create_population(size):
    return [[random.uniform(5, 30)] for _ in range(size)]  # Prices between \$5 and \$30

# Selection: Tournament selection
def select(population, fitnesses, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        competitors = random.sample(range(len(population)), tournament_size)
        winner = max(competitors, key=lambda i: fitnesses[i])
        selected.append(population[winner])
    return selected

# Crossover: Single point crossover for single gene (arithmetic mean)
def crossover(parent1, parent2, rate):
    if random.random() < rate:
        child = [(parent1[0] + parent2[0]) / 2]
        return child
    return parent1

# Mutation: Gaussian mutation
def mutate(individual, rate, sigma=1.0):
    if random.random() < rate:
        individual[0] += random.gauss(0, sigma)
    return individual

# Run GA button
if st.sidebar.button("Run Optimization"):
    # Initialize
    population = create_population(population_size)
    best_fitnesses = []
    
    with st.spinner("Optimizing..."):
        for gen in range(generations):
            fitnesses = [fitness(ind, base_demand, sensitivity, max_price) for ind in population]
            best_fitness = max(fitnesses)
            best_fitnesses.append(best_fitness)
            
            # Selection
            selected = select(population, fitnesses)
            
            # Create next generation
            next_population = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[i+1] if i+1 < len(selected) else selected[0]
                child1 = crossover(parent1, parent2, crossover_rate)
                child2 = crossover(parent2, parent1, crossover_rate)
                next_population.append(mutate(child1, mutation_rate))
                next_population.append(mutate(child2, mutation_rate))
            
            population = next_population[:population_size]  # Trim to size
            # Simple elitism: replace worst with best from previous
            best_ind = max(population, key=lambda ind: fitness(ind, base_demand, sensitivity, max_price))
            population[0] = best_ind
            
            # Progress bar
            st.progress((gen + 1) / generations)
    
    # Find best individual
    best_individual = max(population, key=lambda ind: fitness(ind, base_demand, sensitivity, max_price))
    best_price = best_individual[0]
    best_demand = max(0, base_demand - sensitivity * best_price)
    best_revenue = best_price * best_demand
    
    st.success(f"Optimization Complete! Best Ticket Price: ${best_price:.2f} | Best Demand: {best_demand:.2f} | Best Revenue: ${best_revenue:.2f}")

    # Wow factor: Interactive plot of fitness evolution using Streamlit's built-in chart
    st.header("Fitness Evolution Plot")
    fitness_df = pd.DataFrame({
        "Generation": range(generations),
        "Best Revenue": best_fitnesses
    })
    st.line_chart(fitness_df.set_index("Generation"))

    # Additional wow: 2D Plots for Revenue and Demand vs Price (replacing 3D for compatibility)
    st.header("Revenue and Demand Curves (Wow Factor!)")
    prices = np.linspace(5, 30, 100)
    demands = [max(0, base_demand - sensitivity * p) for p in prices]
    revenues = [p * d for p, d in zip(prices, demands)]
    
    # Demand vs Price
    demand_df = pd.DataFrame({
        "Price": prices,
        "Demand": demands
    })
    st.subheader("Demand vs Price")
    st.line_chart(demand_df.set_index("Price"))
    
    # Revenue vs Price with optimal point
    revenue_df = pd.DataFrame({
        "Price": prices,
        "Revenue": revenues
    })
    st.subheader("Revenue vs Price")
    st.line_chart(revenue_df.set_index("Price"))
    
    # Highlight optimal point (note: Streamlit line_chart doesn't support direct markers, so describe it)
    st.info(f"Optimal Point: Price ${best_price:.2f}, Revenue ${best_revenue:.2f}")

    # If data uploaded, show historical points in a table or additional chart
    if data is not None and 'price' in data.columns and 'demand' in data.columns:
        hist_prices = data['price'].values
        hist_demands = data['demand'].values
        hist_revenues = hist_prices * hist_demands
        hist_df = pd.DataFrame({
            "Price": hist_prices,
            "Demand": hist_demands,
            "Revenue": hist_revenues
        })
        st.subheader("Historical Data Points")
        st.dataframe(hist_df)

# Citation notes (based on relevant references for GA concepts and file upload)
# Genetic Algorithm concepts inspired by tutorials and explanations from sources like MATLAB GA tutorials<sup>1</sup> and GeeksforGeeks comparisons<sup>10</sup>.
# Streamlit integration for apps similar to movie recommendation examples<sup>4</sup><sup>5</sup><sup>7</sup>.
# File upload functionality using st.file_uploader as per Streamlit documentation<sup>6</sup> and community examples<sup>5</sup><sup>8</sup><sup>14</sup><sup>15</sup>.
# Built-in charting with st.line_chart for compatibility without external plotting libraries<sup>6</sup>.
