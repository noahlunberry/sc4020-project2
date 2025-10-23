"""# code source: https://www.datacamp.com/tutorial/apriori-algorithm

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv('dataset.csv')
df.head()

# freq itemsets
freq_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# gen association rule
rules = association_rules(freq_itemsets, metric='confidence', min_threshold=0.6)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

import matplotlib.pyplot as plt
import networkx as nx
# Scatter plot of confidence vs lift
plt.figure(figsize=(8,6))
plt.scatter(rules['confidence'], rules['lift'], alpha=0.7, color='b')
plt.xlabel('Confidence')
plt.ylabel('Lift')
plt.title('Confidence vs Lift in Association Rules')
plt.grid()
plt.show()
# Visualizing association rules as a network graph
G = nx.DiGraph()
for _, row in rules.iterrows():
    G.add_edge(tuple(row['antecedents']), tuple(row['consequents']), weight=row['confidence'])
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10)
edge_labels = {(tuple(row['antecedents']), tuple(row['consequents'])): f"{row['confidence']:.2f}" 
               for _, row in rules.iterrows()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Association Rules Network")
plt.show()"""

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Load dataset
df = pd.read_csv('dataset.csv')
print("Columns:", df.columns)
print("Shape:", df.shape)
df.head()

# Step 2: Prepare transactions
transactions = []

for _, row in df.iterrows():
    # Extract non-null symptom names and normalize
    symptoms = [str(s).strip().lower() for s in row[1:].dropna().tolist()]
    
    # Handle synonym normalization (custom mapping)
    synonym_map = {'pyrexia': 'fever', 'coughing': 'cough'}
    symptoms = [synonym_map.get(s, s) for s in symptoms]
    
    transactions.append(list(set(symptoms)))  # remove duplicates per disease

print(f"Number of transactions (diseases): {len(transactions)}")

# Step 3: One-hot encode transactions
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_symptoms = pd.DataFrame(te_ary, columns=te.columns_)

# Step 4: Apply Apriori algorithm
# Adapted logic based on DataCamp Apriori tutorial
frequent_itemsets = apriori(df_symptoms, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Step 5: Adapt your 'inspect' visualization for mlxtend output
def inspect(rules_df):
    lhs = [', '.join(list(x)) for x in rules_df['antecedents']]
    rhs = [', '.join(list(x)) for x in rules_df['consequents']]
    supports = rules_df['support'].tolist()
    confidences = rules_df['confidence'].tolist()
    lifts = rules_df['lift'].tolist()
    return list(zip(lhs, rhs, supports, confidences, lifts))

resultsinDataFrame = pd.DataFrame(
    inspect(rules),
    columns=["Left hand side", "Right hand side", "Support", "Confidence", "Lift"]
)

# Step 6: Display results
print("Top 10 strongest symptom associations:")
print(resultsinDataFrame.nlargest(10, "Lift"))

# Step 7: Quick scatter visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.scatter(resultsinDataFrame["Support"], resultsinDataFrame["Lift"], alpha=0.6)
plt.title("Symptom Co-occurrence (Support vs Lift)")
plt.xlabel("Support")
plt.ylabel("Lift")
plt.grid(True)
plt.show()
