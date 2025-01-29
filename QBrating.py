import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# Load the CSV file
file_path = "NFLteam2024.csv"
df = pd.read_csv(file_path)

# Extract relevant columns
win_pct = df["Win %"]
qb_rating = df["Rate"]
teams = df["Team"]

# Add a constant for regression
X = sm.add_constant(qb_rating)
y = win_pct

# Fit the regression model
model = sm.OLS(y, X).fit()

# Generate predictions
df["Predicted Win %"] = model.predict(X)

# Directory and file path for saving the plot
output_dir = "/Users/joshuabrooks/NFLpublic"
output_path = os.path.join(output_dir, "WinPercentage_vs_QBRating.png")

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Plot with team names
plt.figure(figsize=(10, 8))
plt.scatter(qb_rating, win_pct, label="Actual Data", alpha=0.7)

# Add team name annotations
for i, team in enumerate(teams):
    plt.annotate(team, (qb_rating[i], win_pct[i]), textcoords="offset points", xytext=(5, 5), ha="center", fontsize=8)

# Add regression line
plt.plot(qb_rating, df["Predicted Win %"], color='red', label="Regression Line")

# Add labels, title, and legend
plt.xlabel("Quarterback Rating")
plt.ylabel("Win Percentage")
plt.title("Win Percentage vs. QB Rating (with Team Names)")
plt.legend()
plt.grid(True)

# Save the plot as a PNG file
plt.savefig(output_path, dpi=300)

# Show the plot
plt.show()

# Print the regression model summary
print(model.summary())

# Output the saved file path
print(f"Plot saved at: {output_path}")
