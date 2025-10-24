# code source: https://www.datacamp.com/tutorial/apriori-algorithm

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: load dataset
df = pd.read_csv("dataset.csv")

# Step 2: Prepare transactions (clean nulls)
transactions = []
for _, row in df.iterrows():
    symptoms = [str(s).strip().lower() for s in row[1:].dropna().tolist() if str(s).strip() != '']
    if len(symptoms) > 0:
        transactions.append(list(set(symptoms)))  # remove duplicates

# Step 3: One-hot encode
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Step 4: Find frequent itemsets
frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)

# Step 5: Generate rules (support + confidence only)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Step 6: Show result
print("\nFrequent Itemsets:")
print(frequent_itemsets.head())

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence']].head())
