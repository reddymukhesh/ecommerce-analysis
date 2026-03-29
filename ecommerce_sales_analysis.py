# ============================================================
#   E-COMMERCE SALES ANALYSIS PROJECT
#   Tools: NumPy, Pandas, Matplotlib, Seaborn
#   Dataset: Online Retail Dataset (UCI / Kaggle)
#   Level: Beginner-Friendly
# ============================================================

# ── STEP 1: Import Libraries ─────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")          # clean chart background
plt.rcParams["figure.figsize"] = (10, 5)  # default figure size


# ── STEP 2: Load Real Data ───────────────────────────────────
# We download the famous "Online Retail" dataset from UCI via a
# direct URL. It contains ~500k transactions from a UK retailer.

url = (
    "Online_Retail.xlsx"
)

print("📦 Loading dataset … (this may take ~30 seconds)")
df = pd.read_excel(url)
print(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns\n")


# ── STEP 3: First Look at the Data ──────────────────────────
print("=" * 55)
print("STEP 3 ── First Look")
print("=" * 55)
print(df.head())           # first 5 rows
print("\nColumn names:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nBasic stats:\n", df.describe())


# ── STEP 4: Data Cleaning ────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 4 ── Data Cleaning")
print("=" * 55)

# 4a. Check missing values
print("Missing values per column:\n", df.isnull().sum())

# 4b. Drop rows where CustomerID is missing (can't analyse per customer)
df.dropna(subset=["CustomerID"], inplace=True)

# 4c. Remove cancelled orders (InvoiceNo starts with 'C')
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# 4d. Remove negative or zero Quantity / UnitPrice
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# 4e. Create a Revenue column  (Quantity × UnitPrice)
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# 4f. Parse InvoiceDate properly
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Month"]       = df["InvoiceDate"].dt.to_period("M")   # e.g. 2011-01
df["DayOfWeek"]   = df["InvoiceDate"].dt.day_name()       # e.g. Monday
df["Hour"]        = df["InvoiceDate"].dt.hour

print(f"\n✅ Clean dataset: {df.shape[0]:,} rows")


# ── STEP 5: Key Business Questions ──────────────────────────
print("\n" + "=" * 55)
print("STEP 5 ── Answering Business Questions")
print("=" * 55)

# Q1. Total revenue
total_revenue = df["Revenue"].sum()
print(f"\n💰 Total Revenue        : £{total_revenue:,.2f}")

# Q2. Total orders and unique customers
total_orders    = df["InvoiceNo"].nunique()
total_customers = df["CustomerID"].nunique()
print(f"🛒 Total Orders         : {total_orders:,}")
print(f"👤 Unique Customers     : {total_customers:,}")

# Q3. Average order value
avg_order_value = df.groupby("InvoiceNo")["Revenue"].sum().mean()
print(f"📊 Avg Order Value      : £{avg_order_value:.2f}")

# Q4. Top 5 products by revenue
top_products = (
    df.groupby("Description")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)
print("\n🏆 Top 5 Products by Revenue:")
print(top_products.to_string())

# Q5. Top 5 countries by revenue
top_countries = (
    df.groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)
print("\n🌍 Top 5 Countries by Revenue:")
print(top_countries.to_string())


# ── STEP 6: Visualisations ───────────────────────────────────
print("\n" + "=" * 55)
print("STEP 6 ── Creating Charts")
print("=" * 55)

# ── Chart 1: Monthly Revenue Trend ──────────────────────────
monthly_rev = df.groupby("Month")["Revenue"].sum().reset_index()
monthly_rev["Month"] = monthly_rev["Month"].astype(str)

plt.figure(figsize=(12, 5))
sns.lineplot(data=monthly_rev, x="Month", y="Revenue",
             marker="o", color="steelblue", linewidth=2.5)
plt.title("📈 Monthly Revenue Trend", fontsize=15, fontweight="bold")
plt.xlabel("Month")
plt.ylabel("Revenue (£)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("chart1_monthly_revenue.png", dpi=150)
plt.show()
print("✅ Saved: chart1_monthly_revenue.png")

# ── Chart 2: Top 10 Products by Revenue (horizontal bar) ────
top10_products = (
    df.groupby("Description")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

plt.figure(figsize=(11, 6))
sns.barplot(data=top10_products, x="Revenue", y="Description",
            palette="Blues_r")
plt.title("🏆 Top 10 Products by Revenue", fontsize=15, fontweight="bold")
plt.xlabel("Revenue (£)")
plt.ylabel("")
plt.tight_layout()
plt.savefig("chart2_top_products.png", dpi=150)
plt.show()
print("✅ Saved: chart2_top_products.png")

# ── Chart 3: Revenue by Country (excluding UK) ──────────────
country_rev = (
    df[df["Country"] != "United Kingdom"]
    .groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

plt.figure(figsize=(11, 6))
sns.barplot(data=country_rev, x="Revenue", y="Country",
            palette="Greens_r")
plt.title("🌍 Top 10 Countries by Revenue (excl. UK)",
          fontsize=15, fontweight="bold")
plt.xlabel("Revenue (£)")
plt.ylabel("")
plt.tight_layout()
plt.savefig("chart3_top_countries.png", dpi=150)
plt.show()
print("✅ Saved: chart3_top_countries.png")

# ── Chart 4: Orders by Day of Week ──────────────────────────
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
orders_by_day = (
    df.groupby("DayOfWeek")["InvoiceNo"]
    .nunique()
    .reindex(day_order)
    .reset_index()
)
orders_by_day.columns = ["DayOfWeek", "Orders"]

plt.figure(figsize=(9, 5))
sns.barplot(data=orders_by_day, x="DayOfWeek", y="Orders",
            palette="OrRd")
plt.title("📅 Number of Orders by Day of Week",
          fontsize=15, fontweight="bold")
plt.xlabel("Day")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.savefig("chart4_orders_by_day.png", dpi=150)
plt.show()
print("✅ Saved: chart4_orders_by_day.png")

# ── Chart 5: Revenue Distribution (histogram) ───────────────
order_revenue = df.groupby("InvoiceNo")["Revenue"].sum()
order_revenue_clipped = order_revenue[order_revenue < 1000]  # focus on typical orders

plt.figure(figsize=(10, 5))
sns.histplot(order_revenue_clipped, bins=50, kde=True, color="mediumpurple")
plt.title("📊 Distribution of Order Revenue (< £1,000)",
          fontsize=15, fontweight="bold")
plt.xlabel("Order Revenue (£)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("chart5_revenue_distribution.png", dpi=150)
plt.show()
print("✅ Saved: chart5_revenue_distribution.png")

# ── Chart 6: Heatmap — Hour vs Day of Week (order count) ────
pivot = (
    df.groupby(["DayOfWeek", "Hour"])["InvoiceNo"]
    .nunique()
    .unstack(fill_value=0)
    .reindex(day_order)
)

plt.figure(figsize=(14, 6))
sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.3,
            cbar_kws={"label": "Number of Orders"})
plt.title("🕐 Order Heatmap: Hour of Day vs Day of Week",
          fontsize=15, fontweight="bold")
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")
plt.tight_layout()
plt.savefig("chart6_heatmap.png", dpi=150)
plt.show()
print("✅ Saved: chart6_heatmap.png")


# ── STEP 7: NumPy Insights ───────────────────────────────────
print("\n" + "=" * 55)
print("STEP 7 ── NumPy Statistical Insights")
print("=" * 55)

rev_array = order_revenue.values  # convert to NumPy array

print(f"Mean   order value : £{np.mean(rev_array):.2f}")
print(f"Median order value : £{np.median(rev_array):.2f}")
print(f"Std deviation      : £{np.std(rev_array):.2f}")
print(f"Max order value    : £{np.max(rev_array):.2f}")
print(f"Min order value    : £{np.min(rev_array):.2f}")

# Percentiles
p25, p75 = np.percentile(rev_array, [25, 75])
print(f"25th percentile    : £{p25:.2f}")
print(f"75th percentile    : £{p75:.2f}")


# ── STEP 8: Summary ──────────────────────────────────────────
print("\n" + "=" * 55)
print("🎉 PROJECT COMPLETE — Summary")
print("=" * 55)
print("""
Concepts covered:
  ✅ Loading real-world Excel data with Pandas
  ✅ Data cleaning (nulls, cancellations, negatives)
  ✅ Feature engineering (Revenue, Month, DayOfWeek, Hour)
  ✅ Groupby aggregations & sorting
  ✅ 6 professional charts with Matplotlib + Seaborn
  ✅ NumPy descriptive statistics

Charts saved:
  📊 chart1_monthly_revenue.png
  📊 chart2_top_products.png
  📊 chart3_top_countries.png
  📊 chart4_orders_by_day.png
  📊 chart5_revenue_distribution.png
  📊 chart6_heatmap.png
""")
