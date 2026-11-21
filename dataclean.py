import pandas as pd
import numpy as np

# 1. Read the CSV file

df = pd.read_csv('ecommerce_customer_behavior_dataset_v2.csv')

# 2. Trim leading and trailing whitespace across all text columns
obj_cols = df.select_dtypes(include="object").columns
df[obj_cols] = df[obj_cols].apply(lambda col: col.str.strip())

# 3. Standardize inconsistent 'Gender' entries ('m', 'M' -> 'Male'; 'F' -> 'Female')
gender_map = {
    "m": "Male",
    "M": "Male",
    "F": "Female"
}
df["Gender"] = df["Gender"].replace(gender_map)

# 4. Impute missing 'Customer_ID' using the Order_ID prefix (e.g., ORD_000005-1 -> CUST_00005)
missing_cust_mask = df["Customer_ID"].isna()
extracted_id = df.loc[missing_cust_mask, "Order_ID"].str.extract(r"^ORD_0*(\d+)-")[0]
df.loc[missing_cust_mask, "Customer_ID"] = "CUST_" + extracted_id.str.zfill(5)

# 5. Clean 'Total_Amount': convert 'NAN' strings to NaN, cast to float, and recalculate
df["Total_Amount"] = pd.to_numeric(df["Total_Amount"], errors="coerce")
calculated_total = np.round(df["Unit_Price"] * df["Quantity"] - df["Discount_Amount"], 2)
df["Total_Amount"] = np.where(df["Total_Amount"].isna(), calculated_total, df["Total_Amount"])

# 6. Parse 'Date' into standard datetime format (YYYY-MM-DD)
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

# 7. Drop any duplicate rows and reset index
df = df.drop_duplicates().reset_index(drop=True)

# 8. Export the cleaned CSV in the same folder
df.to_csv('cleaned_data2.csv', index=False)

