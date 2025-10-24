# code source: https://www.datacamp.com/tutorial/apriori-algorithm

support = 0.1
threshold = 0.6

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

"""
Frequent Itemsets:
    support          itemsets
0  0.209756  (abdominal_pain)
1  0.141463      (chest_pain)
2  0.162195          (chills)
3  0.114634           (cough)
4  0.115854      (dark_urine)

Association Rules:
           antecedents         consequents   support  confidence
0         (dark_urine)    (abdominal_pain)  0.110976    0.957895
1     (abdominal_pain)  (loss_of_appetite)  0.132927    0.633721
2     (abdominal_pain)          (vomiting)  0.176829    0.843023
3  (yellowing_of_eyes)    (abdominal_pain)  0.114634    0.691176
4     (yellowish_skin)    (abdominal_pain)  0.154878    0.835526
"""