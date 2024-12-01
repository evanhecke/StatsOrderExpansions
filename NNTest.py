import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load from pickle
df = pd.read_pickle("dataset.pkl")
print(df.head())

# df = df.sample(n=1000, random_state=30)

# Step 3: Split the dataset into training and testing sets
df_train, df_test = train_test_split(df, test_size=0.2, random_state=30)

# Example DataFrame with longitude and latitude
data = pd.DataFrame({
    'Longitude': df_test['long'],
    'Latitude': df_test['lat'],
    #'Value': df_test['nclaims']  # Optional: Weight values
})

"""# Plot KDE heatmap
plt.figure(figsize=(8, 10))
sns.kdeplot(
    x=data['Longitude'], y=data['Latitude'],
    cmap='Reds', fill=True, bw_adjust=1
)
"""
# Hexbin plot with color based on the number of points in each hexagon
plt.hexbin(data['Longitude'], data['Latitude'], gridsize=50, cmap='coolwarm', mincnt=1)

# Load Belgium map from downloaded shapefile
belgium_map = gpd.read_file("C:/Users/Evert/PycharmProjects/CapstoneProject/ne_10m_admin_0_sovereignty/ne_10m_admin_0_sovereignty.shp")
belgium_map[belgium_map['NAME'] == 'Belgium'].plot(ax=plt.gca(), edgecolor='black', facecolor='none')

plt.title("Hexbin Plot of Points in Belgium")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Add colorbar to show density
plt.colorbar(label='Density')

plt.show()


