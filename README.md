# -mess-feedback-analyzer
# ==============================
# Hostel Feedback Analysis Project
# Simple & Beginner Friendly Code
# ==============================

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ------------------------------
# STEP 1: Load Data
# ------------------------------

# Sample data (you can replace with CSV later)
data = {
    "Hostel": ["A","A","A","A","A",
               "B","B","B","B","B",
               "C","C","C","C","C"],
    
    "Food": ["Biryani","Rice","Roti","Biryani","Rice",
             "Roti","Rice","Biryani","Roti","Rice",
             "Rice","Roti","Biryani","Rice","Roti"],
    
    "Rating": [5,4,3,4,5,
               3,4,5,3,4,
               2,3,3,2,4],
    
    "Cleanliness": [4,4,3,5,4,
                    3,3,4,3,3,
                    2,2,3,2,2],
    
    "Comment": [
        "Good food", "Nice", "Average", "Loved it", "Tasty",
        "Okay", "Good", "Best", "Not bad", "Fine",
        "Dirty", "Needs cleaning", "Okay food", "Bad hygiene", "Improve"
    ]
}

df = pd.DataFrame(data)

print("Data Loaded Successfully\n")
print(df.head())

# ------------------------------
# STEP 2: Data Analysis
# ------------------------------

print("\nAverage Rating by Food:")
food_rating = df.groupby("Food")["Rating"].mean()
print(food_rating)

print("\nAverage Cleanliness by Hostel:")
clean_rating = df.groupby("Hostel")["Cleanliness"].mean()
print(clean_rating)

# ------------------------------
# STEP 3: Visualization
# ------------------------------

# Food rating chart
food_rating.plot(kind="bar")
plt.title("Average Food Rating")
plt.xlabel("Food")
plt.ylabel("Rating")
plt.show()

# Cleanliness chart
clean_rating.plot(kind="bar", color='orange')
plt.title("Hostel Cleanliness Rating")
plt.xlabel("Hostel")
plt.ylabel("Cleanliness")
plt.show()

# ------------------------------
# STEP 4: Machine Learning Model
# ------------------------------

# Predict Rating based on Cleanliness

X = df[["Cleanliness"]]   # input
y = df["Rating"]          # output

model = LinearRegression()
model.fit(X, y)

print("\nML Model Trained!")

# Test prediction
test_value = [[3]]
prediction = model.predict(test_value)

print(f"Predicted Rating for Cleanliness=3: {prediction[0]:.2f}")

# ------------------------------
# STEP 5: Simple AI Summary (Beginner Version)
# ------------------------------

print("\n--- Feedback Summary ---")

# Most popular food
best_food = food_rating.idxmax()
print(f"Most popular food is {best_food}")

# Worst cleanliness hostel
worst_hostel = clean_rating.idxmin()
print(f"Hostel {worst_hostel} needs cleanliness improvement")

# Common words in comments
all_comments = " ".join(df["Comment"]).lower()

if "dirty" in all_comments or "bad" in all_comments:
    print("Students are unhappy with cleanliness")

print("\nSummary Generated Successfully!")

# ------------------------------
# END
# ------------------------------
