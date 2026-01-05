# ======================================
# 1. UPLOAD DATASET
# ======================================
from google.colab import files
files.upload()   # Upload cinema_hall_ticket_sales (1).csv


# ======================================
# 2. IMPORT LIBRARY
# ======================================
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================
# 3. LOAD DATASET (FIX ENCODING)
# ======================================
# GUNA encoding='latin1' (PALING SELAMAT)
df = pd.read_csv("cinema_hall_ticket_sales (1).csv", encoding="latin1")

print("Dataset berjaya dibaca ✅")
print(df.head())


# ======================================
# 4. CHECK COLUMN NAMES
# ======================================
print("\nColumn dalam dataset:")
print(df.columns)


# ======================================
# 5. TETAPKAN COLUMN (UBAH JIKA PERLU)
# ======================================
# 🔴 JIKA NAMA COLUMN BERBEZA, TUKAR DI SINI
PRICE_COL = df.columns[0]   # contoh: ticket_price
SOLD_COL  = df.columns[1]   # contoh: tickets_sold

print(f"\nHarga column: {PRICE_COL}")
print(f"Tiket sold column: {SOLD_COL}")


# ======================================
# 6. PRICE RANGE
# ======================================
PRICE_MIN = df[PRICE_COL].min()
PRICE_MAX = df[PRICE_COL].max()


# ======================================
# 7. ESTIMATE DEMAND FROM DATASET
# ======================================
def estimate_demand(price, df):
    idx = (df[PRICE_COL] - price).abs().idxmin()
    return df.loc[idx, SOLD_COL]


# ======================================
# 8. FITNESS FUNCTION
# ======================================
def fitness(price):
    tickets_sold = estimate_demand(price, df)
    return price * tickets_sold


# ======================================
# 9. GA PARAMETERS
# ======================================
POP_SIZE = 50
GENERATIONS = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.05
ELITISM_SIZE = 2


# ======================================
# 10. INITIAL POPULATION
# ======================================
def init_population():
    return [random.uniform(PRICE_MIN, PRICE_MAX) for _ in range(POP_SIZE)]


# ======================================
# 11. SELECTION
# ======================================
def selection(population):
    tournament = random.sample(population, 3)
    return max(tournament, key=fitness)


# ======================================
# 12. CROSSOVER
# ======================================
def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        return (p1 + p2) / 2
    return p1


# ======================================
# 13. MUTATION
# ======================================
def mutation(price):
    if random.random() < MUTATION_RATE:
        return random.uniform(PRICE_MIN, PRICE_MAX)
    return price


# ======================================
# 14. MAIN GA LOOP (WITH ELITISM)
# ======================================
population = init_population()
best_fitness_history = []
best_price_history = []

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
    best_price_history.append(best_price)
    best_fitness_history.append(fitness(best_price))

    if gen % 10 == 0:
        print(f"Gen {gen} | Best Price RM {best_price:.2f} | Revenue RM {fitness(best_price):.2f}")


# ======================================
# 15. FINAL RESULT
# ======================================
best_price = population[0]

print("\n===== FINAL RESULT =====")
print(f"Optimal Ticket Price : RM {best_price:.2f}")
print(f"Estimated Tickets Sold : {estimate_demand(best_price, df)}")
print(f"Maximum Revenue : RM {fitness(best_price):.2f}")


# ======================================
# 16. PLOT OPTIMIZATION GRAPH
# ======================================
plt.figure()
plt.plot(best_fitness_history)
plt.xlabel("Generation")
plt.ylabel("Best Revenue (RM)")
plt.title("GA Optimization of Cinema Ticket Pricing")
plt.show()


# ======================================
# 17. PLOT PRICE EVOLUTION
# ======================================
plt.figure()
plt.plot(best_price_history)
plt.xlabel("Generation")
plt.ylabel("Ticket Price (RM)")
plt.title("Evolution of Ticket Price")
plt.show()
