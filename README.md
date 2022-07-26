# Introduction to Dash (and Plotly)

Plotly is a python plotting library (compareable to matplotlib). Its biggest advantage is the interactivity features it provides, which is especially good due to its beginner-friendly high-level interface. Dash is built on top of Plotly, and is used to create interactive dashboards. 

## Three main parts of creating a Dash app:

Part| Description
---|----|
Layout| where you describe (using [Dash HTML Components](https://dash.plotly.com/dash-html-components) and [Dash Core Components](https://dash.plotly.com/dash-core-components))
Dash Core Components| like drop-down menues, buttons, and graphs; they provide the functionality that allows the user to interact with the different elements inside the layout
Callback| implements the logic governing the relationship between the input given from an Input Core Component (e.g., drop-down menue selection) and produced output seen through an Output Core Component (e.g., a graph)


## Files

File| Description
---|----|
dash-example.py| a simple dashboard showing the average data scientist's salary in different countries (built in class)
knn-dash-iris.py| an interactive dashboard allowing for analysis of KNN algorithm's predictions based on different parameters


## Learn More
- [:link: GitHub Repo containing 100s of nicely done dashboards](https://github.com/plotly/dash-sample-apps/)
- [:link: Dash official tutorial](https://dash.plotly.com/introduction)









