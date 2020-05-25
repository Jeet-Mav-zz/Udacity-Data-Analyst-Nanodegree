# Communicate Data Findings - FordGoBike
## By Jeet

Data visualization is the practice of translating information into a visual context, such as a map or graph, to make data easier for the human brain to understand and pull insights from. The main goal of data visualization is to make it easier to identify patterns, trends and outliers in large data sets.

## DataSet
BayWheels or FordGoBike is bike renting service in the areas of San Francisco through which we can rent/share a bike using their hourly as well as monthly and yearly subscription. I used dataset for the months of May, June, July, August, September and October which is available [here](https://s3.amazonaws.com/baywheels-data/index.html).

## Wrangling Process
* Fixed datatypes of start_time and end_time
* Converted bike_id and station_id to object type
* Converted bike_share and user_type columns to categorical
* Added new columns for Hour of the trip, Day of the trip and Month of the Trip
* Filtered out the null values

## Summary 
* **92.77%** users dont like to share a bike ride
* **79.76%** are Subscribers whereas only **20.24%** are customers
* **Berry St at 4th St** , **Market St at 10th St**, **San Francisco Ferry Building**, **San Francisco Caltrain** are stations with most number of riders and **Brannan St at 7th**, **19th St at William St** have the least riders i.e. 2
* **San Francisco Ferry Building** has an usual high number of customers with comparison to subscribers
* Majority of the rides happen betweeen the **7th - 9th hour** and **15th - 19th hour**
* **Wednesday** is the day with most number of rides in a day
* Weekends have the fall in number of riders especially **Sunday**

## Key Insights

Subscribers use the this service heavily on weekdays whereas customers dont use that much but their usage rises during the late hours. Many trips concentrated around 7-9am and 15-19pm which happends to be the work hours suggesting that subscribers use this service for their daily routine instead of leisure. Which means customer are taking advantage of this service in a form of leisure and general travel instead of work related and their usage generally rises during the later hours and also they tend to use this service on weekends also not like the subscribers which dont prefer it as much.