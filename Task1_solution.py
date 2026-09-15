# ==========================================
# 1. Data Cleaning
# ==========================================

# Step 1: Input, Output & Initialization
max_val = int(input("Enter maximum pixel value (e.g., 255): "))
has_error = False 
raw_pixels = [120, -15, 256, 100, 310, 45, 0, -5, 200]
processed_pixels = list()

print(f"Ready to process {len(raw_pixels)} pixels with a max limit of {max_val}.")

# Step 2: Iteration, Flow Control & List Methods
for pixel in raw_pixels:
    # Check for negative values (corrupted data)
    if pixel < 0:
        print(f"Warning: Negative value ({pixel}) detected! Skipping...")
        has_error = True
        continue  # Skip the rest of THIS loop iteration
        
    # Check if the pixel value exceeds the maximum limit
    elif pixel > max_val: 
        processed_pixels.append(max_val)
        
    # For all other normal values
    else:
        processed_pixels.append(pixel)

print("\nData processing complete!")

# Step 3: Final Output & Identity Operators
if has_error is True: 
    print("Process finished: Corrupted data was found and removed.")
else:
    print("Process finished: All data was originally clean.")

print("\nProcessed Data:", processed_pixels)


# ==========================================
# 2. Data Normalization Pipeline
# ==========================================

# Step 1: Default Arguments & Returning Values
transformation_count = 0

def min_max_scale(value, min_val=0, max_val=100):
    scaled = (value - min_val) / (max_val - min_val)
    return scaled

print("\nGlobal counter and scaling function defined successfully!")

# Step 2: Higher-Order Functions, Scope & *args
def process_data(data_list, transform_func, *args):
    global transformation_count
    result_list = []
    
    for item in data_list:
        new_item = transform_func(item, *args)
        result_list.append(new_item)
        
    transformation_count += 1
    return result_list

print("Manager function 'process_data' is ready!")

# Step 3: Execution (Passing by Reference)
raw_data = [20, 50, 80, 10]
data_min = 10
data_max = 80

normalized_data = process_data(raw_data, min_max_scale, data_min, data_max)

print("Original Data:", raw_data)
print("Normalized Data:", normalized_data)
print("Total Pipeline Executions:", transformation_count)


# ==========================================
# 3. Model Configuration & Metadata
# ==========================================

# Step 1: Tuples (Immutable Data)
image_shape = (224, 224, 3) 
print("\nModel Input Shape:", image_shape)

# Step 2: Dictionaries (Key-Value Pairs)
model_config = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "optimizer": "Adam"
}
model_config["batch_size"] = 64
print("Updated Config:", model_config)

# Step 3: Sets (Unique Elements)
raw_labels = ["cat", "dog", "cat", "bird", "dog", "cat", "bird"]
unique_classes = set(raw_labels)
print("Unique Classes:", unique_classes)

# Step 4: Strings & List Comprehension
image_files = ["image_01.jpg", "image_02.jpg", "image_03.jpg"]
clean_names = [file.replace(".jpg", "") for file in image_files]
print("Cleaned File Names:", clean_names)


# ==========================================
# 4. Building a Model Blueprint (OOP Basics)
# ==========================================

# Step 1: Defining the Class, Constructor, and Methods
class SimpleModel:
    def __init__(self, model_name, learning_rate):
        self.name = model_name
        self.lr = learning_rate
        self.is_trained = False
        print(f"\nModel '{self.name}' created with Learning Rate: {self.lr}")

    def train(self, epochs):
        print(f"Training '{self.name}' for {epochs} epochs...")
        self.is_trained = True
        print("Training complete!")

print("Blueprint (Class) successfully defined!")

# Step 2: Instantiating and Using the Object
my_ai_model = SimpleModel("VisionNet", 0.001)
my_ai_model.train(10)

print("Is the model trained?", my_ai_model.is_trained)


# ==========================================
# 5. Saving Training Logs (File Handling)
# ==========================================

# Step 1: Writing Logs to a File
training_logs = [
    "Epoch 1: Accuracy = 0.75, Loss = 0.50\n",
    "Epoch 2: Accuracy = 0.82, Loss = 0.40\n",
    "Epoch 3: Accuracy = 0.89, Loss = 0.30\n"
]

with open("metrics.txt", "w") as file:
    for log in training_logs:
        file.write(log)

print("\nLogs successfully saved to 'metrics.txt'!")

# Step 2: Reading from a File
with open("metrics.txt", "r") as file:
    saved_content = file.read()

print("--- Displaying Saved Logs ---")
print(saved_content)
