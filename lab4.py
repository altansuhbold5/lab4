import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

df = pd.read_csv("StudentsPerformance_sltJocu.csv")
scores = ['math score', 'reading score', 'writing score']

print(df.shape)
print(df.isnull().sum().to_dict())
print("duplicates:", df.duplicated().sum())
print(df[scores].agg(['min', 'max']))
for col in ['gender', 'lunch', 'test preparation course']:
    print(col, df[col].unique())

rng = np.random.default_rng(42)
df_miss = df.copy()
for col in ['math score', 'reading score']:
    df_miss[col] = df_miss[col].astype(float)
    idx = rng.choice(df_miss.index, size=int(0.10 * len(df_miss)), replace=False)
    df_miss.loc[idx, col] = np.nan
idx = rng.choice(df_miss.index, size=int(0.05 * len(df_miss)), replace=False)
df_miss.loc[idx, 'lunch'] = np.nan
print((df_miss.isnull().mean() * 100).round(2).to_dict())

df_a = df_miss.dropna()
df_b = df_miss.copy()
df_b['math score'] = df_b['math score'].fillna(df_b['math score'].mean())
df_c = df_miss.copy()
df_c['reading score'] = df_c['reading score'].fillna(df_c['reading score'].median())
df_d = df_miss.copy()
df_d['math score'] = df_d['math score'].fillna(df_d.groupby('gender')['math score'].transform('mean'))
df_e = df_miss.copy()
df_e['lunch'] = df_e['lunch'].fillna(df_e['lunch'].mode()[0])
print("rows after dropna:", len(df_a))
print(pd.DataFrame({
    'original': df['math score'].agg(['mean', 'std']),
    'ignore_tuple': df_a['math score'].agg(['mean', 'std']),
    'mean_fill': df_b['math score'].agg(['mean', 'std']),
    'class_mean_fill': df_d['math score'].agg(['mean', 'std'])
}).round(3))

prices = pd.Series([4, 8, 15, 21, 21, 24, 25, 28, 34])
bins = [prices[i:i + 3] for i in range(0, len(prices), 3)]
smooth_mean = [[round(b.mean())] * len(b) for b in bins]
smooth_bound = []
for b in bins:
    lo, hi = b.min(), b.max()
    smooth_bound.append([int(lo) if (v - lo) <= (hi - v) else int(hi) for v in b])
print(smooth_mean, smooth_bound)

q1 = df['math score'].quantile(0.25)
q3 = df['math score'].quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
iqr_out = df[(df['math score'] < lower) | (df['math score'] > upper)]
print("IQR:", lower, upper, len(iqr_out))

z = (df['math score'] - df['math score'].mean()) / df['math score'].std()
z_out = df[z.abs() > 2]
print("z-score outliers:", len(z_out))

students = df[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']].copy()
students.insert(0, 'student_id', range(1, len(df) + 1))
scores_tbl = df[scores].copy()
scores_tbl.insert(0, 'student_id', range(1, len(df) + 1))
scores_dup = pd.concat([scores_tbl, scores_tbl.sample(5, random_state=1)], ignore_index=True)
scores_clean = scores_dup.drop_duplicates(subset=['student_id'])
merged = students.merge(scores_clean, on='student_id', how='inner')
print(len(scores_dup), len(scores_clean), merged.shape)
print(merged[scores].corr().round(4))

ct = pd.crosstab(merged['gender'], merged['lunch'])
chi2, p, dof, expected = chi2_contingency(ct)
print(ct)
print("chi2 =", round(chi2, 3), "p =", round(p, 4), "dof =", dof)

x = df['math score']
df['math_minmax'] = (x - x.min()) / (x.max() - x.min())
df['math_z'] = (x - x.mean()) / x.std()
j = len(str(int(x.abs().max())))
df['math_decimal'] = x / 10 ** j
print(df[['math score', 'math_minmax', 'math_z', 'math_decimal']].agg(['min', 'max', 'mean', 'std']).round(4))

df['math_width'] = pd.cut(df['math score'], bins=3, labels=['Low', 'Medium', 'High'])
df['math_freq'] = pd.qcut(df['math score'], q=3, labels=['Low', 'Medium', 'High'])
df['math_level'] = pd.cut(df['math score'], bins=[0, 60, 80, 101], labels=['Low', 'Medium', 'High'], right=False)
print(df['math_width'].value_counts().sort_index())
print(df['math_freq'].value_counts().sort_index())
print(df['math_level'].value_counts().sort_index())