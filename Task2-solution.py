import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# PART 1: The Deep Learning Tensor Engine (NumPy)
# ==========================================

# Step 1: Imports & 3D Slicing
image_batch = np.array([
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],          # Image 1 (Index 0)
    [[10, 20, 30], [40, 50, 60], [70, 80, 90]]   # Image 2 (Index 1)
])

sub_region = image_batch[1, :, 1:]
print("Cropped Region:\n", sub_region)

# Step 2: Sequences, Math & Smart Filtering
range_arr = np.arange(0, 10, 2)
exp_arr = np.exp(range_arr)

cond = range_arr > 3
filtered_vals = range_arr[cond]

print("\nSequence:", range_arr)
print("Filtered (>3):", filtered_vals)

# Step 3: Structured Arrays & Sorting
log_dtype = [('model_name', 'S10'), ('epoch', int), ('loss', float)]

raw_logs = [('ResNet', 20, 0.45), ('VGG', 10, 0.60), ('ResNet', 10, 0.55), ('VGG', 20, 0.35)]
logs_array = np.array(raw_logs, dtype=log_dtype)

sorted_logs = np.sort(logs_array, order=['epoch', 'loss'])
print("\nSorted Logs:\n", sorted_logs)

# Step 4: The Speed Test
large_list = list(range(1000000))
large_arr = np.array(large_list)

# Pure Python
start_time = time.time()
_ = sum(large_list) 
py_time = time.time() - start_time

# NumPy
start_time = time.time()
_ = np.sum(large_arr)
np_time = time.time() - start_time

print(f"\nNumPy is {py_time / np_time:.2f}x faster than pure Python!")

# Step 5: Tensor Transformations
flat_tensor = np.arange(12)

matrix_3x4 = flat_tensor.reshape((3, 4))
flat_again = matrix_3x4.flatten()

print("\n3x4 Matrix:\n", matrix_3x4)

# Step 6: Matrix Multiplication & Aggregations
inputs = np.array([[1, 2], [3, 4]])
weights = np.array([[0.5, -0.5], [1.0, 2.0]])
bias = np.array([0.1, -0.1])

output_matrix = (inputs @ weights) + bias

max_val = np.max(output_matrix)
best_class_idx = np.argmax(output_matrix)
mean_val = np.mean(output_matrix)

print("\nOutput Matrix:\n", output_matrix)
print(f"Max: {max_val} | Argmax: {best_class_idx} | Mean: {mean_val}\n")


# ==========================================
# PART 2: Data Cleaning & Feature Engineering (Pandas)
# ==========================================

# Step 1: Creating & Exploring
raw_data = {
    'user_id': [101, 102, 103, 104, 105],
    'age': [22, 28, np.nan, 35, 40],
    'app_purchases': [0, 50, 10, 100, np.nan],
    'in_app_purchases': [5, 20, 0, 50, 10],
    'unnecessary_col': ['A', 'B', 'C', 'D', 'E']
}

df = pd.DataFrame(raw_data)

print("--- First 3 Rows ---")
print(df.head(3))

print("\n--- Statistical Summary ---")
print(df.describe())

# Step 2: Handling Missing Data & Dropping
df = df.drop('unnecessary_col', axis=1)
df = df.dropna(subset=['age'])
df = df.fillna(0)

print("\n--- Cleaned DataFrame ---")
print(df)

# Step 3: Combining Data (Concat & Merge)
new_users = pd.DataFrame({'user_id': [106], 'age': [30], 'app_purchases': [20], 'in_app_purchases': [5]})

# Stack them vertically
df = pd.concat([df, new_users], ignore_index=True)

# External data
sub_data = pd.DataFrame({
    'user_id': [101, 102, 104, 105, 106],
    'category': ['Free', 'Premium', 'Premium', 'Premium', 'Free']
})

# Merge based on matching IDs
df = pd.merge(df, sub_data, on='user_id')

print("\n--- Combined DataFrame ---")
print(df)

# Step 4: Filtering, Engineering & Grouping
adult_users = df[df['age'] > 25]

df['total_spent'] = df['app_purchases'] + df['in_app_purchases']

category_sales = df.groupby('category')['total_spent'].sum()

print("\n--- Sales by Category ---")
print(category_sales)

# Step 5: File I/O (Read & Write CSV)
df.to_csv("cleaned_data.csv", index=False)

loaded_df = pd.read_csv("cleaned_data.csv")

print("\n--- Final Loaded DataFrame from CSV ---")
print(loaded_df)


# ==========================================
# PART 3: The Advanced Visualization Dashboard (Matplotlib)
# ==========================================

# Create the 2x2 dashboard grid
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# 1. LINE CHART (Top-Left)
epochs = [1, 2, 3, 4, 5]
loss = [0.9, 0.6, 0.4, 0.2, 0.1]

axes[0, 0].plot(epochs, loss, marker='o', label="Training Loss")
axes[0, 0].set_title("1. Learning Curve")
axes[0, 0].set_xlabel("Epochs")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. BAR CHART (Top-Right)
classes = ['Dogs', 'Cats', 'Birds']
counts = [150, 120, 200]

axes[0, 1].bar(x=classes, height=counts, color='orange')
axes[0, 1].set_title("2. Image Counts")

# 3. HISTOGRAM (Bottom-Left)
pixel_values = np.random.randint(0, 255, 50)

axes[1, 0].hist(pixel_values, bins=10, color='steelblue')
axes[1, 0].set_title("3. Pixel Distribution")

# 4. SCATTER PLOT (Bottom-Right)
model_complexity = [10, 15, 20, 25, 30]
training_time = [12, 18, 25, 28, 35]

axes[1, 1].scatter(x=model_complexity, y=training_time, color='purple')
axes[1, 1].set_title("4. Complexity vs Time")

# Adjust spacing and export
plt.tight_layout()
fig.savefig("my_dashboard.png")
print("\nDashboard generated and saved successfully as 'my_dashboard.png'!")
