# code source: https://www.datacamp.com/tutorial/apriori-algorithm

support = 0.01
threshold = 0.7

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules # source: https://www.datacamp.com/tutorial/apriori-algorithm

# 1. load dataset
df = pd.read_csv("dataset.csv")

# 2. prepare data, clear null values
transactions = []
for _, row in df.iterrows():
    symptoms = [str(s).strip().lower() for s in row[1:].dropna().tolist() if str(s).strip() != '']
    if len(symptoms) > 0:
        transactions.append(list(set(symptoms)))  # to remove duplicates

# 3. one hot encode
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# 4. find frequent itemsets and generate rules [source: https://www.datacamp.com/tutorial/apriori-algorithm]
frequent_itemsets = apriori(df_encoded, min_support = support, use_colnames = True)
rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = threshold)

# 5. results
print("\nFrequent Itemsets:")
print(frequent_itemsets.head())

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence']].head())