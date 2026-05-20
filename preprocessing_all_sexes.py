import pandas as pd

EXCEL = "undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx"
OUT   = "data_all_sexes.parquet"
YEARS = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024]

print("Reading Excel...")
df_raw = pd.read_excel(EXCEL, sheet_name="Table 1", header=None)

data = df_raw.iloc[11:].copy().reset_index(drop=True)

SEX_START_COL = {
    "both_sexes": 7,
    "male":15,
    "female":23,
}

chunks = []
for sex, start in SEX_START_COL.items():
    for i, year in enumerate(YEARS):
        col = start + i
        chunk = pd.DataFrame({
            "destination":      data[1],
            "destination_code": pd.to_numeric(data[4], errors="coerce"),
            "origin":           data[5],
            "origin_code":      pd.to_numeric(data[6], errors="coerce"),
            "year":             year,
            "sex":              sex,
            "migrant_stock":    pd.to_numeric(data[col], errors="coerce"),
        })
        chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)

df["destination"] = df["destination"].str.replace("*", "", regex=False)
df["origin"]      = df["origin"].str.replace("*", "", regex=False)
df["destination"] = df["destination"].str.replace("Türkiye", "Turkey", regex=False)
df["origin"]      = df["origin"].str.replace("Türkiye", "Turkey", regex=False)

df = df[
    (df["destination_code"] < 900) &
    (df["origin_code"]      < 900)
].copy()

df["year"] = df["year"].astype(int)
df["sex"]  = df["sex"].astype(str)

print(f"Shape : {df.shape}")
print(f"Sex   : {df['sex'].unique().tolist()}")
print(f"Years : {sorted(df['year'].unique())}")
print(f"NaN   : {df['migrant_stock'].isna().sum()} rows with missing stock")

print(f"Saving to {OUT}...")
df.to_parquet(OUT, engine="fastparquet", index=False)
print("Done.")