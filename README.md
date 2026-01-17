# ECommerce-Business
# Enhancing Customer Retention and Profitability in E-Commerce: A Data Driven Approach Using Behavioural and Predictive Analytics 

## Table of Content
- [Project Overview](#project-overview)
- [Introduction](#introduction)
- [Aim and Objective](#aim-and-objective)
- [Methodology and Data Collection Methods](#methodology-and-data-collection-methods)
- [Analysis and results](#analysis-and-results)
- [Discussion](#discussion)
- [Business Insights and Recommedations](#business-insights-and-recommendations)
- [Conclusion](#conclusion)
- [Appendix](#appendix)

  
## Project Overview
For e-commerce business to succeed, customer retention is important. Businesses are adopting technology by either going entirely online or using a hybrid business model. Therefore, it is important to comprehend how consumer satisfaction, service quality, and behaviour intentions relate to one another in e-commerce. According to this perspective, the study's objectives include analysing transactional data to offer a solution for how data-driven insights improve decision-making and customer satisfaction in eCommerce businesses, comprehending the purchasing patterns that have the biggest impact on customer retention in eCommerce businesses, figuring out how to segment customers based on their purchasing behaviour to improve personalisation, and identifying business products that have the biggest impact on overall profitability. To accomplish them, it collects data from UCI Machine Learning Repository, which offers business data from an ecommerce platform, examines the data using business intelligence tools and the Python programming language, and proposes an appropriate model for dealing with the problem. Furthermore, the project proposes that ecommerce online businesses include essential services that influence customer retention and revenue through behavioural and predictive analysis. 

## Introduction 
In contemporary marketing, customers are important and the mission of every organisation is to acquire, retain and expand their customers. Customer retention is an important strategy in any business that retains existing customers and generates revenue from them over time whilst acquiring new customers. Customer retention is classified as the mirror of customer defection or churn, high retention equivales to low defection (Galih Saputro et al., 2020). There are challenges in enhancing customer retention business today, many organisations struggle to comprehend factors that influence customers purchasing behaviour and repeat sales. Business still struggles to build their relationship with their customers.Understanding that customers are different and needs different services and products should be an important factor for any ecommerce business to consider. This prompt the need for customer segmentation, Umuhoza et al. (2020) explained customers segmentation as the basis of analysing the diverse needs of various customers and subdivided them with different attributes and features into specific categories. 

## Aim and Objective
This project aims to bridge these gaps by improving customer retention and profitability in E-Commerce using behavioural and predictive analytics in the online market. Providing deep insights into customer behaviour as well as effective approaches. Acquiring and retaining customers are pricy, however allocating resources to support them is crucial. To tackle these challenges, this project will be analysing transactional data to provide a solution to these research question: 
RQ1: How can data driven insights enhance decision making and customers satisfaction in eCommerce business? 
RQ2: What purchasing patterns most influence customer retention in eCommerce business? 
RQ3: How can customers be segmented based on their purchasing behaviour to enhance personalization? 
RQ4: Which product contribute most to overall profitability? 

## Methodology and Data collection Methods
This research uses a framework for customer behaviour and retention, combining logistic regression, segmentation, and clustering for customer retention with classification and descriptive methods for product and consumer shopping behaviour. Figure 1 illustrates the four processes of the methodology: dataset selection and extraction, data cleaning and preprocessing, exploratory data analysis (EDA), and analytical techniques and methods. 

<img width="212" height="482" alt="image" src="https://github.com/user-attachments/assets/42b6666d-5c23-4954-b8fa-ed82c944cecc" />

Figure 1: Methodology Framework

### Dataset Selection and Data Extraction
The selected dataset for this project is an online retail transactional data sourced from UCI Machine Learning Repository that relates to the business problem. The UCI Machine learning Repository is a collection of a reliable database for machine learning datasets. 

Description
The dataset is an excel file which contains all the transaction occurring between 2010 and 2011 for a United Kingdom based ad registered online retail store with five hundred and forty-one thousand, nine hundred and ten (541910) rows and eight (8) columns, which surpasses the minimum required for this project. The organisation mainly sells unique gifts to its customers, it’s a business to business (B2B) customers-based transaction because majority the transaction are sold to wholesalers (Chen, 2015). 

This dataset is suitable for the project because it has a variety of customer from around Europe and includes a typical variable of any business in ecommerce. It gives the edge to identify the problem with customer behavioural and retention of business and its suitable for the analysis.

### Data Extraction 
The dataset is publicly available at UCI Machine Learning Repository; the extraction process is listed below:
•	Downloaded the dataset in an xlsx format.
•	Loaded into my workspace using Power BI.
•	Store raw data in Microsoft Fabric Lakehouse.

The online retail dataset would be transformed to have different calculated table (Sales, Customer, Country Lookup, and Product) for data structure and organization of within the dataset and help with the analysis (Figure 2).
 <img width="751" height="584" alt="image" src="https://github.com/user-attachments/assets/81190eef-e56c-40e3-a2c5-896b791742ff" />
Figure 2: Data Schema

### Data Cleaning and Preprocessing
The process of data cleaning and preprocessing requires identifying and removing errors, inaccuracy and inconsistencies in dataset to ensure they are consistent. To achieve this, I have carefully outlined the steps I used to cleaning and preprocessing my dataset using Power Query.
•	Missing data: In carrying out data cleaning and preprocessing, the dataset was 541000 rows and 8 columns as explained in the data selection and extraction section. There was empty cell from the variables precisely 135080 from Customer ID and 1454 from Description and performed the exclusion strategy to remove empty cells by using the Remove Blank rows on the home bar (Figure 3)
<img width="1038" height="277" alt="image" src="https://github.com/user-attachments/assets/877009e5-857b-4f0c-a716-427acfed669b" />

Figure 3: Cleaning missing data.

•	Outliers: There were various outliers identified, such as negative and larger quantities in the quantity column, unit price with zero value, and null customer ID and Invoice date. To remove these outliers (Figure 4) using Power Query, I filtered Quantity and UnitPrice to remove negative values and larger numbers.
 <img width="427" height="515" alt="image" src="https://github.com/user-attachments/assets/c30db4fc-5104-4aeb-aea0-84a17c88858c" />

Figure 4: Outliers.

•	Data Types: This project ensured all columns had the correct data types by using power query. The dataset inserted into Power BI, transformed and loaded it. It used the table view to change its datatype in the Column tools menu as shown in Figure 5.
<img width="940" height="260" alt="image" src="https://github.com/user-attachments/assets/ca739f66-eff8-47b2-b5f2-90cd0d03198b" />
Figure 5: Assigning data types.

•	Normalization/Standardization: In alignment with the aim of this project, the model and methods (Logistic Regression and RFM features) that will be used would require standardization for data analysis. Computed Z-scores for Quantity and Unit price Column with formula (e.g.: UnitPrice * Average [Unit Price])/ S.D [Unit Price], and created measures for Revenue, Frequency, Monetary (RFM) and applied Normalization to the existing measures (Figure 6).
<img width="870" height="187" alt="image" src="https://github.com/user-attachments/assets/002a0d92-2bd7-4df1-94db-48577060afe3" />

Figure 6: Normalization.

•	Feature Engineering: To develop methods and models for an e-commerce company dataset (online retail), certain features were created. Sales, product, customer, and continent tables are among the calculated tables that were created to combine data and new variables together, as seen in figure 2. To better understand how a product or quantity affects a customer's perception towards a particular product, as well as what factors influence their purchasing behaviour, new metrics and criteria were created in Figure 7.
<img width="279" height="492" alt="image" src="https://github.com/user-attachments/assets/42b01f00-2609-43c4-9487-c1d4eab87d84" />

Figure 7: Feature Engineering

## Analysis and Results
To analyse customer behaviour and retention in an online business, models such as descriptive and diagnostic, prediction, segmentation, and classification models have been applied in this study.

Key Influencers
There are factors that influence revenue, analysis from Figure 9 reveals that seasonality and product play a role in revenue. The average Total revenue increases in November by €9,304.45 units greater than in all other months. This influencer accounts for around 9.20% of the data. On average, total revenue increases more in November than in other months. In November, over €19,000 in income was earned across all items, compared to an average of €10,1116.27.
 <img width="865" height="463" alt="image" src="https://github.com/user-attachments/assets/b4ae665d-9ff9-4089-b420-730fa645d7c5" />

Figure 9: Key Influencer.

When the product is Baking Equipment, the average total income increases by €4,590.43 above all other items. Total revenue is more likely to rise when the product is baking equipment than otherwise. The second highest product is bag, which generates an average total income of €4,172.15.

Churn Prediction
Identifying customers likely to stop patronizing a business, a customer churn prediction needs to be implemented. Figure 10 depicts the projected number of consumers that will churn (1) and remain active (0). The churn bar is lower than the active bar, indicating that there is no retention challenge. However, a churning segment should be considered.
 <img width="542" height="520" alt="image" src="https://github.com/user-attachments/assets/e1a050e1-e75d-45b7-9987-4f27b184201d" />

Figure 10: Predicted Churn vs Active.

The column churn probabilities indicate the possibility that a client will leave the store and not return. Customers with probability near 1.0 are at high risk of abandoning the store (figure 11). This provides interventions for discount distribution, loyalty programs, and personalised outreach. This procedure provides the business with a view of churn risk across its customer base.
 <img width="865" height="474" alt="image" src="https://github.com/user-attachments/assets/397b97db-5bf4-4fe2-824d-f5dbae4f6751" />

Figure 11: Top 20 Customer's Churn Prediction Probability

Segmentation using K-Means
Customer segmentation utilising K-means on RFM features resulted in the creation of four clusters that classified customers into one of four segments based on their RFM profiles. Figure 12 depicts each cluster by colour and groups them according to monetary and recency. 
 <img width="940" height="437" alt="image" src="https://github.com/user-attachments/assets/80ccafd6-50e7-4d6d-9695-dfa6265d67a2" />

Figure 12: Segmentation using K-Means

The business insight from figure 13 reveals that segment 2 customers are the most valuable, as they frequently purchase and spend heavily. Segment 3 is still viable, but less so than segment 2, as it consists of individual customers rather than bulk buyers. Segment 1 is likely to be new consumers, while Segment 0 consists of dominant clients who are about to leave the store. A possible option is to send personalised emails to segment 0 to help customers understand the drop off.
 <img width="940" height="321" alt="image" src="https://github.com/user-attachments/assets/b6a0bae0-f256-4e50-8f8d-893eaeb586b6" />

Figure 13: RFM values by segment

Classification using Correlation metric
To determine how products have a substantial impact on revenues, a correlation metric is created, which measures the quantity and total revenue of data. Figure 14 indicates a substantial positive corelation across all categories, all products are greater than 0.85, with decoration and household items (0.93) having the highest. Every unit sold generates more revenue.
 <img width="865" height="402" alt="image" src="https://github.com/user-attachments/assets/de2ed3be-5419-4f84-8843-1321f9d3930f" />

Figure 14: Correlation Metric

A feature importance for product classification was built using a bar chart to predict which products have an impact on revenue. Referring to figure 15, revenue is the most relevant factor for forecasting product type.
 
<img width="865" height="463" alt="image" src="https://github.com/user-attachments/assets/9ce46687-982c-4ac1-acaa-8ea1aa9735e3" />

## Discussion
In performing this analysis, the dataset had to be cleaned and pre-processed, calculated tables that distinguished between sales, products, customers, and regions had to be created. The description column in the dataset was cleaned and sectioned into various product categories. To find out how sales are for each region, the country's column was categorized into continents using Power BI. The data reveals that customer retention is favourable and the amount of product purchased have a significant impact on revenue. The limitations of the dataset, data preprocessing decisions, and modelling assumptions limited the machine learning models performance, even though they yielded insightful results. Future research must make improvements in the areas of richer data sources with demographic information (age, gender, etc) and methods to improve prediction accuracy.

## Business Insights and Recommendations
The business insight obtained from the data-driven research in ecommerce indicates that the business needs greater engagement tactics based on customer purchase behaviour. There are significant gaps between purchases, and only a few purchases are made per month. Despite growing revenues, the product performance demonstrates a small number of productivities. Product level optimisation is required to improved pricing strategies, which requires forecasting and inventory alignment for seasonal patterns. There should be different strategies for the international market, as the United Kingdom dominates sales compared to other regions. The ecommerce business needs to boost customer retention by offering discounts and recommending products based on previous purchases. Customers should be introduced to a loyalty program that offers points for each purchase and free gifts to loyal customers. Management needs to apply the RFM segment to identify valuable products and one-time purchasers to increase product recommendations. implementing these recommendations to practice, the business will be able to expand and increase operational effectiveness.

## Conclusion
Acquiring new potential customers for business growth is essential, but maintaining existing customers is more important. The project's primary outcome is focused on understanding customer purchase behaviour using predictive analytics, as well as keeping customers and increasing profitability applying ecommerce models. When implementing promotional techniques, an ecommerce business must first categorise its customers. With the advancement of artificial intelligence, there are models that can assist businesses understand their customers by evaluating their behaviour. Implementing a loyalty program and offering discounts to existing customers would help with retention and profitability. Further research can help to expand the analysis of customer retention and profitability in e-commerce utilising behavioural and predictive analytics.

## Appendix
Python code used in Py Visual
Predicted Churn vs Active
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Power BI dataset
df = dataset.copy().dropna()

# Features and label
X = df[['Recency', 'Frequency', 'Monetary']]
y = df['Churn EC']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
df['PredictedChurn'] = model.predict(X)
df['ChurnProbability'] = model.predict_proba(X)[:,1]

# Show output in visual
df[['CustomerID', 'PredictedChurn', 'ChurnProbability']]


df['PredictedChurn'].value_counts().plot(kind='bar')
#plt.title("Predicted Churn vs Active")
plt.xlabel("Churn (1) / Active (0)")
plt.ylabel("Customer Count")
plt.tight_layout()
plt.show()

Churn Prediction Results (Top 20 Customers)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Power BI dataset
df = dataset.copy().dropna()

# Features and label
X = df[['Recency', 'Frequency', 'Monetary']]
y = df['Churn EC']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
df['PredictedChurn'] = model.predict(X)
df['ChurnProbability'] = model.predict_proba(X)[:,1]

# Show output in visual
df[['CustomerID', 'PredictedChurn', 'ChurnProbability']]

 #Prepare table (limit to first 15 rows so it fits visually)
table_df = df[['CustomerID', 'PredictedChurn', 'ChurnProbability']].head(20)

# Plot table
plt.figure(figsize=(20, 6))
#plt.title("Churn Prediction Results (Top 20 Customers)", )
plt.axis('off')

tbl = plt.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    loc='center'
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(20)
tbl.scale(1, 1.3)
plt.show()

Average RFM Values by Segment

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# Load data from Power BI
df = dataset.copy()

# Select segmentation features
rfm = df[['Recency', 'Frequency', 'Monetary']].copy()

# Standardize the data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# Build K-Means model
kmeans = KMeans(n_clusters=4, random_state=42)
df['CustomerSegment'] = kmeans.fit_predict(rfm_scaled)

# Create cluster profile table
cluster_profiles = df.groupby('CustomerSegment')[['Recency','Frequency','Monetary']].mean()
cluster_profiles = cluster_profiles.round(2)

fig, ax = plt.subplots(figsize=(10,6))  # adjust width and height here
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=cluster_profiles.values,
                 colLabels=cluster_profiles.columns,
                 rowLabels=cluster_profiles.index,
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # scale up table size
plt.show()

# Return only the table to Power BI

plt.table(cluster_profiles)
plt.show()
Correlation Between Quantity and Revenue by Product

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Copy dataset
df = dataset.copy()

# Prepare list to store results
results = []

# Group by Product and compute correlation
for product, group in df.groupby('Product'):
    # Only compute if there is variation in both columns
    if group['Quantity'].nunique() > 1 and group['Total Revenue'].nunique() > 1:
        corr_value = group['Quantity'].corr(group['Total Revenue'])
        results.append([product, corr_value])

# Build new DataFrame with Product and Correlation
corr_df = pd.DataFrame(results, columns=['Product', 'Correlation'])

# Sort for readability
corr_df = corr_df.sort_values(by='Correlation', ascending=False)

# Plot bar chart instead of heatmap (simpler for categorical labels)
plt.figure(figsize=(10,6))
sns.heatmap(corr_df.set_index('Product').T, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
#plt.title("Correlation Between Quantity and Revenue by Product")
plt.tight_layout()
plt.show()


Feature Importance for Product Classification

# The following code to create a dataframe and remove duplicated rows is always executed and acts as a preamble for your script: 

# dataset = pandas.DataFrame(UnitPrice, Quantity, Revenue, Country, Frequency)
# dataset = dataset.drop_duplicates()

# Paste or type your script code here:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Data from Power BI
df = dataset.copy()

# Feature engineering
df['Revenue'] = df['Quantity'] * df['UnitPrice']

# Select predictors
X = df[['Quantity', 'UnitPrice', 'Revenue']].copy()

# Target (must exist in dataset, e.g., ProductCategory)
y = df['Product']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Print classification report
print(classification_report(y_test, y_pred))

# Feature importance plot
plt.figure(figsize=(7,5))
plt.bar(X.columns, model.feature_importances_)
#plt.title("Feature Importance for Product Classification")
plt.tight_layout()
plt.show()


Customer Segmentation Using K-Means

# The following code to create a dataframe and remove duplicated rows is always executed and acts as a preamble for your script: 

# dataset = pandas.DataFrame(Frequency, Monetary, Recency)
# dataset = dataset.drop_duplicates()

# Paste or type your script code here:
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data from Power BI
df = dataset.copy()

# Select segmentation features
rfm = df[['Recency', 'Frequency', 'Monetary']].copy()

# Standardize the data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# Build K-Means model
kmeans = KMeans(n_clusters=4, random_state=42)
df['CustomerSegment'] = kmeans.fit_predict(rfm_scaled)

# Plot Segments
plt.figure(figsize=(8,6))
plt.scatter(df['Recency'], df['Monetary'], c=df['CustomerSegment'])
cluster_profiles = df.groupby('CustomerSegment')[['Recency','Frequency','Monetary']].mean()
#plt.table(cluster_profiles)

#plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Recency")
plt.ylabel("Monetary")
plt.tight_layout()
plt.show()




