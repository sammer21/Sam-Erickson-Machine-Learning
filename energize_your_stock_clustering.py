#!/usr/bin/env python
# coding: utf-8

# # Energizing Stock Clustering

# In[22]:


# Import the required libraries and dependencies
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


# ### Step 1: Read in the `tsx-energy-2018.csv` file and create the DataFrame. Make sure to set the `Ticker` column as the DataFrame’s index. Then review the DataFrame.

# In[23]:


# Read the CSV file into a Pandas DataFrame
# Set the index using the Ticker column
df_stocks = pd.read_csv("https://static.bc-edx.com/mbc/ai/m2/datasets/tsx-energy-2018.csv", index_col="Ticker")

# Review the DataFrame
df_stocks.head()


# ### Step 2: Review the four code cells that are included in this step in the notebook. These cells contain the code that scales the `df_stocks` DataFrame and creates a new DataFrame that contains the scaled data. 

# In[24]:


# Scale price data, return, and variance values
stock_data_scaled = StandardScaler().fit_transform(
    df_stocks[["MeanOpen", "MeanHigh", "MeanLow", "MeanClose", "MeanVolume", "AnnualReturn", "AnnualVariance"]]
)


# In[25]:


# Create a DataFrame with the scaled data
df_stocks_scaled = pd.DataFrame(
    stock_data_scaled,
    columns=["MeanOpen", "MeanHigh", "MeanLow", "MeanClose", "MeanVolume", "AnnualReturn", "AnnualVariance"]
)

# Copy the tickers names from the original data
df_stocks_scaled["Ticker"] = df_stocks.index

# Set the Ticker column as index
df_stocks_scaled = df_stocks_scaled.set_index("Ticker")

# Display sample data
df_stocks_scaled.head()


# In[26]:


# Encode (convert to dummy variables) the `EnergyType` column, which categorizes oil versus non-oil firms
oil_dummies = pd.get_dummies(df_stocks["EnergyType"])
oil_dummies.head()


# In[27]:


# Concatenate the `EnergyType` encoded dummies with the scaled data DataFrame
df_stocks_scaled = pd.concat([df_stocks_scaled, oil_dummies], axis=1)

# Display the sample data
df_stocks_scaled.head()


# ### Step 3: Using the `df_stocks_scaled` DataFrame, cluster the data by using the K-means algorithm and a lowercase-k value of  3. Add the resulting list of company segment values as a new column in the `df_stocks_scaled` DataFrame. 
# 
# > **Rewind** You can use a lowercase-k value of 3 to start, or you can use the elbow method to find the optimal value for lowercase-k.

# In[28]:


# Initialize the K-Means model with n_clusters=3
model = KMeans(n_clusters=3, n_init='auto')


# In[29]:


# Fit the model for the df_stocks_scaled DataFrame
model.fit(df_stocks_scaled)


# In[30]:


# Predict the model segments (clusters)
stock_clusters = model.predict(df_stocks_scaled)

# View the stock segments
print(stock_clusters)


# In[31]:


# Create a new column in the DataFrame with the predicted clusters
df_stocks_scaled["StockCluster"] = stock_clusters

# Review the DataFrame
df_stocks_scaled.head()


# ### Step 4: Using Pandas plot, create a scatter plot to visualize the clusters setting `x="AnnualVariance"`,  `y="Annual Return"`, and `by="StockCluster"`. Be sure to style and format your plot.

# In[32]:


# Create a scatter plot with x="AnnualVariance:,  y="AnnualReturn"
df_stocks_scaled.plot.scatter(
    x="AnnualVariance",
    y="AnnualReturn",
    c="StockCluster",
    colormap='winter',
    title = "Scatter Plot by Stock Segment - k=3"
)


# ### Step 5: To get another perspective on the clusters, reduce the number of features to two principal components by using PCA. Make sure to do the following: 
# 
# ---
# - Use the `df_stocks_scaled` DataFrame to complete this analysis. 
# - Review the PCA data. 
# - Calculate the explained variance ratio that results from the PCA data. 
# 

# In[33]:


# Create the PCA model instance where n_components=2
pca = PCA(n_components=2)


# In[34]:


# Fit the df_stocks_scaled data to the PCA
stocks_pca_data = pca.fit_transform(df_stocks_scaled)

# Review the first five rose of the PCA data
# using bracket notation ([0:5])
stocks_pca_data[:5]


# In[35]:


# Calculate the explained variance
pca.explained_variance_ratio_


# ### Step 6: Using the PCA data calculated in the previous step, create a new DataFrame called `df_stocks_pca`. Make sure to do the following: 
# 
# * Add an additional column to the DataFrame that contains the tickers from the original `df_stocks` DataFrame. 
# 
# * Set the new Tickers column as the index. 
# 
# * Review the DataFrame.
# 

# In[36]:


# Creating a DataFrame with the PCA data
df_stocks_pca = pd.DataFrame(stocks_pca_data, columns=["PC1", "PC2"])

# Copy the tickers names from the original data
df_stocks_pca["Ticker"] = df_stocks.index

# Set the Ticker column as index
df_stocks_pca = df_stocks_pca.set_index("Ticker")

# Review the DataFrame
df_stocks_pca.head()


# ### Step 7: Rerun the K-means algorithm with the new principal-components data, and then create a scatter plot by using the two principal components for the x and y axes, and by using `StockCluster`. Be sure to style and format your plot.

# In[37]:


# Initialize the K-Means model with n_clusters=3
model = KMeans(n_clusters=3, n_init='auto')

# Fit the model for the df_stocks_pca DataFrame
model.fit(df_stocks_pca)

# Predict the model segments (clusters)
stock_clusters = model.predict(df_stocks_pca)

# View the stock segments
print(stock_clusters)


# In[38]:


# Create a copy of the df_stocks_pca DataFrame and name it as df_stocks_pca_predictions
df_stocks_pca_predictions = df_stocks_pca.copy()

# Create a new column in the DataFrame with the predicted clusters
df_stocks_pca_predictions["StockCluster"] = stock_clusters

# Review the DataFrame
df_stocks_pca_predictions.head()


# In[39]:


# Create the scatter plot with x="PC1" and y="PC2"
df_stocks_pca_predictions.plot.scatter(
    x="PC1",
    y="PC2",
    c="StockCluster",
    colormap='winter',
    title = "Scatter Plot by Stock Segment - PCA=2"
)


# In[40]:


component_df=pd.DataFrame(pca.components_,index=['PCA1',"PCA2"],columns=df_stocks_scaled.columns)
# Heat map
sns.heatmap(component_df)
plt.show()

