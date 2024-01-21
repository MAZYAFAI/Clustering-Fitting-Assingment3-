
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Read the data from the csv file 
df_countries=pd.read_csv("C:\\Users\\MAZ YAFAI\\Desktop\\Maz assingment\\22092593.csv",skiprows=4)


def age_percentage_model(year, a, b, c):
    return a * np.exp(b * (year - 1990)) + c

def plot_age_percentage_prediction(df_countries, country_name):
    # Extract years and age percentage data for the specified country
    country_data = df_countries[df_countries['Country Name'] == country_name]
    years = country_data.columns[4:]  # Assuming the years start from the 5th column
    age_percentage = country_data.iloc[:, 4:].values.flatten()

    # Convert years to numeric values
    years_numeric = pd.to_numeric(years, errors='coerce')
    age_percentage = pd.to_numeric(age_percentage, errors='coerce')

    # Remove rows with NaN or inf values
    valid_data_mask = np.isfinite(years_numeric) & np.isfinite(age_percentage)
    years_numeric = years_numeric[valid_data_mask]
    age_percentage = age_percentage[valid_data_mask]

    # Curve fitting
    popt, pcov = curve_fit(age_percentage_model, years_numeric, age_percentage, p0=[1, -0.1, 90])

    # Optimal parameters
    a_opt, b_opt, c_opt = popt

    # Generate model predictions for the year 2040
    year_2040 = 2040
    age_percentage_2040 = age_percentage_model(year_2040, a_opt, b_opt, c_opt)

    # Plot the original data and the fitted curve
    plt.scatter(years_numeric, age_percentage, label='Original Data')
    plt.plot(years_numeric, age_percentage_model(years_numeric, a_opt, b_opt, c_opt), label='Fitted Curve', color='red')

    # Highlight the prediction for 2040
    plt.scatter(year_2040, age_percentage_2040, color='green', marker='*', label='Prediction for 2040')

    # Add labels and legend
    plt.xlabel('Year')
    plt.ylabel('Age Percentage')
    plt.title(f'Age Percentage Prediction for {country_name}')
    plt.legend()

    # Show the plot
    plt.show()

# Example usage with a list of countries
countries = ['Argentina','Pakistan','Hong Kong SAR, China']
for country in countries:
    plot_age_percentage_prediction(df_countries, country)



#CLUSTER ANALYSIS
# Extract data for the years 1999 and 2022
years = ['1960', '2022']
age_dependency_data = df_countries[['Country Name'] + years]

# Drop rows with missing values
age_dependency_data = age_dependency_data.dropna()

# Set 'Country Name' as the index
age_dependency_data.set_index('Country Name', inplace=True)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(age_dependency_data)

# Perform KMeans clustering
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(normalized_data)

# Add cluster labels to the DataFrame
age_dependency_data['Cluster'] = labels

# Visualize the clusters
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Cluster for 1999
axs[0].scatter(age_dependency_data[years[0]], age_dependency_data.index, c=labels, cmap='viridis')
axs[0].set_title(f'Clustered Age Dependency in {years[0]}')
axs[0].set_xlabel('Age Dependency')
axs[0].set_ylabel('Countries')

# Cluster for 2022
axs[1].scatter(age_dependency_data[years[1]], age_dependency_data.index, c=labels, cmap='viridis')
axs[1].set_title(f'Clustered Age Dependency in {years[1]}')
axs[1].set_xlabel('Age Dependency')
axs[1].set_ylabel('Countries')

# Manually set y-axis label
for ax in axs:
    ax.set_yticks([])
    ax.set_yticklabels([])

plt.tight_layout()
plt.show()
