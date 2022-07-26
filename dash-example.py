
import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output  # pip install dash (version 2.0.0 or higher)
import country_converter as coco
import numpy as np



app = Dash(__name__)

df = pd.read_csv('ds_salaries_cleaned.csv')
df = df.drop(columns=['Unnamed: 0'])


print(df[:5])

'''
# convert country codes
locations = coco.convert(names=df['employee_residence'], to='ISO3')
locations = np.unique(locations)

avg_salary = df.groupby('employee_residence')['salary_in_usd'].mean()

fig = px.choropleth(df, locations=locations,
                    locationmode="ISO-3",
                    # color="salary_in_usd", 
                    color=avg_salary,
                    # hover_name="country", # column to add to hover information
                    # color_continuous_scale=px.colors.sequential.Plasma
                    )
'''

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.H1("Data Careers Salary", style={'text-align': 'center'}),

    dcc.Dropdown(id="selected-year",
                 options=[
                     {"label": "2020", "value": 2020},
                     {"label": "2021", "value": 2021},
                     {"label": "2022", "value": 2022}],
                 multi=False,
                 value=2022,
                 style={'width': "40%"}
                 ),

    html.Div(id='string_output', children=[]),
    html.Br(),

    dcc.Graph(id='salary_map', figure={})

])


# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='string_output', component_property='children'),
     Output(component_id='salary_map', component_property='figure')],
    [Input(component_id='selected-year', component_property='value')]
)
def update_graph(option_selected):

    container = "The year chosen by user was: {}".format(option_selected)

    dff = df.copy()
    dff = dff[dff["work_year"] == option_selected]

    
    # convert country codes
    locations = coco.convert(names=df['employee_residence'], to='ISO3')
    locations = np.unique(locations)

    # compute average salary for each country
    avg_salary = df.groupby('employee_residence')['salary_in_usd'].mean()

    fig = px.choropleth(df, locations=locations,
                        locationmode="ISO-3",
                        # color="salary_in_usd", #bad color scale
                        color=avg_salary,
                        )


    return container, fig


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)