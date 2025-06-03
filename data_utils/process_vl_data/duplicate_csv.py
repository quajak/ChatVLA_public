import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('/home/jz08/LMUData/MMRO_mini.tsv')

df_repeated = df.loc[df.index.repeat(5)].reset_index(drop=True)
# 保存为新的 CSV 文件
df_repeated.to_csv('/home/jz08/LMUData/duplicated_rows.csv', index=False)


