from django.shortcuts import render
from django.http import HttpResponse
from pymongo import MongoClient
import plotly.graph_objects as go
import plotly.offline as opy
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import dash
from sklearn.cluster import KMeans
import plotly.figure_factory as ff
from sklearn import linear_model

client = MongoClient("localhost", 27017)
database = client.bigdata
collection1 = database.Country
collection2 = database.indicator
collection3 = database.goverment

app = dash.Dash()


def index(request):
    return HttpResponse("index")


def korea(request):
    pipline, pipline2, pipline3 = list(), list(), list()
    pipline.append({'$match': {'IndicatorName': "GDP growth (annual %)", 'CountryCode': "USA", 'Year': {'$lte': 2013}}})
    pipline.append({'$project': {'_id': 0, 'Value': 1}})
    pipline2.append(
        {'$match': {'IndicatorName': "GDP growth (annual %)", 'CountryCode': "KOR", 'Year': {'$lte': 2013}}})
    pipline2.append({'$project': {'_id': 0, 'Value': 1}})
    result = collection2.aggregate(pipline)
    result2 = collection2.aggregate(pipline2)
    USA, Korea = list(), list()
    for results in result:
        USA.append(round(results['Value'], 2))

    for i in result2:
        Korea.append(round(i['Value'], 2))

    model = LinearRegression()
    model = model.fit(X=pd.DataFrame(USA), y=Korea)
    prediction = model.predict(X=pd.DataFrame(USA))
    print(model.coef_, model.intercept_)
    residuals = Korea - prediction
    SSE = (residuals ** 2).sum()
    SST = ((pd.DataFrame(USA) - pd.DataFrame(USA).mean()) ** 2).sum()
    R_squared = 1 - (SSE / SST)
    print('R_squared = ', R_squared)

    trace1 = go.Scatter(x=USA, y=Korea,
                        mode="markers")
    trace2 = go.Scatter(x=USA, y=prediction, mode="lines")
    data = go.Data([trace1, trace2])
    layout = go.Layout(
        autosize=True,
    )
    fig = go.Figure(data=data, layout=layout)
    plot = opy.plot(fig, output_type='div')

    lst = [
        [2.75, 4.04, 2.72, 3.8, 4.49, 4.45, 4.69, 4.09, 0.98, 1.79, 2.81, 3.79, 3.35, 2.67, 1.78, -0.29, -2.78, 2.53,
         1.6, 2.32, 2.22, 2.39],
        [6.33, 8.77, 8.93, 7.19, 5.77, -5.71, 10.73, 8.83, 4.53, 7.43, 2.93, 4.9, 3.92, 5.18, 5.46, 2.83, 0.71, 6.5,
         3.68, 2.29, 2.9, 3.31],
        [13.94, 13.08, 10.99, 9.92, 9.23, 7.85, 7.62, 8.43, 8.3, 9.09, 10.02, 10.08, 11.35, 12.69, 14.19, 9.62, 9.23,
         10.63, 9.48, 7.75, 7.68, 7.27],
        [0.17, 0.86, 1.94, 2.61, 1.6, -2.0, -0.2, 2.26, 0.36, 0.29, 1.69, 2.36, 1.3, 1.69, 2.19, -1.04, -5.53, 4.65,
         -0.45, 1.75, 1.61, -0.1],
        [-0.96, 2.46, 1.74, 0.82, 1.85, 1.98, 1.99, 2.96, 1.7, 0.0, -0.71, 1.17, 0.71, 3.7, 3.26, 1.08, -5.62, 4.08,
         3.66, 0.41, 0.3, 1.6],
        [-0.61, 2.35, 2.09, 1.39, 2.34, 3.56, 3.41, 3.88, 1.95, 1.12, 0.82, 2.79, 1.61, 2.37, 2.36, 0.2, -2.94, 1.97,
         2.08, 0.18, 0.66, 0.18],
        [2.65, 4.02, 4.92, 2.67, 3.1, 3.38, 3.11, 3.8, 2.76, 2.49, 3.34, 2.49, 3.0, 2.66, 2.59, -0.47, -4.19, 1.54,
         1.97, 1.18, 2.16, 2.94],
        [-0.85, 2.15, 2.89, 1.29, 1.84, 1.62, 1.56, 3.71, 1.77, 0.25, 0.15, 1.58, 0.95, 2.01, 1.47, -1.05, -5.48, 1.71,
         0.59, -2.82, -1.75, -0.44], ]
    code = ['USA', 'KOR', 'CHN', 'JPN', 'DEU', 'FRA', 'GBR', 'ITA']
    df = pd.DataFrame(lst).T
    corr = df.corr(method='pearson')
    lst = corr.values.tolist()
    fig2 = go.Figure(data=go.Heatmap(
        z=lst,
        x=code,
        y=code))
    fig2.update_layout(
        autosize=True,
    )
    plot2 = opy.plot(fig2, output_type='div')
    pipline = []
    tax, export, gep = [], [], []
    pipline.append({'$match': {'IndicatorName': "Tax revenue (% of GDP)", "CountryCode": "USA",
                               'Year': {'$gte': 1995}}})
    pipline.append({'$project': {'_id': 0, 'Value': 1, 'Year': 1}})
    result = collection2.aggregate(pipline)
    for i in result:
        tax.append(round(i['Value'], 2))

    pipline = []
    pipline.append({'$match': {'IndicatorName': "Imports of goods and services (annual % growth)", "CountryCode": "USA",
                               'Year': {'$gte': 2001}}})
    pipline.append({'$project': {'_id': 0, 'Value': 1, 'Year': 1}})
    result = collection2.aggregate(pipline)
    for i in result:
        export.append(round(i['Value'], 2))

    pipline = []
    pipline.append({'$match': {'IndicatorName': "GDP growth (annual %)", "CountryCode": "USA",
                               'Year': {'$gte': 2001}}})
    pipline.append({'$project': {'_id': 0, 'Value': 1, 'Year': 1}})
    pipline.append({'$limit': 13})
    result = collection2.aggregate(pipline)
    for i in result:
        gep.append(round(i['Value'], 2))

    pipline, gni = [], []

    pipline.append({'$match': {'IndicatorName': "GNI growth (annual %)", "CountryCode": "USA",
                               'Year': {'$gte': 2001}}})
    pipline.append({'$project': {'_id': 0, 'Value': 1, 'Year': 1}})
    result = collection2.aggregate(pipline)
    for i in result:
        gni.append(round(i['Value'], 2))

    pipline, grs = [], []
    pipline.append({'$match': {'IndicatorName': "Gross savings (% of GDP)", "CountryCode": "USA",
                               'Year': {'$gte': 2001}}})
    pipline.append({'$project': {'_id': 0, 'Value': 1, 'Year': 1}})
    result = collection2.aggregate(pipline)
    for i in result:
        grs.append(round(i['Value'], 2))

    result = []
    result.append(tax)
    result.append(gep)
    result.append(export)
    result.append(gni)
    result.append(grs)
    print(result)

    gep_re = pd.DataFrame(result).T
    corr2 = gep_re.corr(method='pearson')
    lst2 = corr2.values.tolist()
    code2 = ['Tax', 'Import', 'GDP', 'GNI', 'Save']
    fig5 = go.Figure(data=go.Heatmap(
        z=lst2,
        x=code2,
        y=code2))
    fig5.update_layout(
        autosize=True,
    )
    plot5 = opy.plot(fig5, output_type='div')

    pipline, pipline2 = [], []
    pipline.append({'$match': {'IndicatorName': "Exports of goods and services (annual % growth)", 'CountryCode': "KOR",
                               'Year': {'$gte': 1995}}})
    pipline.append({'$project': {'_id': 0, 'Value': 1, 'Year': 1}})

    pipline2.append({'$match': {'IndicatorName': "Imports of goods and services (annual % growth)",
                                'CountryCode': "KOR", 'Year': {'$gte': 1995}}})
    pipline2.append({'$project': {'_id': 0, 'Value': 1, 'Year': 1}})

    pipline3.append(
        {'$match': {'IndicatorName': "GDP growth (annual %)", 'CountryCode': "KOR", 'Year': {'$gte': 1995}}})
    pipline3.append({'$project': {'_id': 0, 'Value': 1, 'Year': 1}})
    result = collection2.aggregate(pipline)
    result2 = collection2.aggregate(pipline2)
    result3 = collection2.aggregate(pipline3)
    income, export, gdp, year = list(), list(), list(), list()
    for i in result:
        income.append(round(i['Value'], 2))
        year.append(i['Year'])

    for i in result2:
        export.append(round(i['Value'], 2))

    for i in result3:
        gdp.append(round(i['Value'], 2))
    data = {'x1': income,
            'x2': export,
            'y': gdp}
    data = pd.DataFrame(data)
    X = data[['x1', 'x2']]
    y = data['y']

    model = LinearRegression()
    model = model.fit(X=pd.DataFrame(X), y=y)
    prediction = model.predict(X=pd.DataFrame(X))

    trace3 = go.Scatter(x=year, y=income,
                        mode="lines", name='Imports')
    trace4 = go.Scatter(x=year, y=export, mode="lines", name='Exports')
    trace5 = go.Scatter(x=year, y=gdp, mode="lines", name='GDP')
    data2 = go.Data([trace3, trace4, trace5])
    layout2 = go.Layout(
        autosize=True,
    )
    fig3 = go.Figure(data=data2, layout=layout2)
    plot3 = opy.plot(fig3, output_type='div')

    pipline, pipline2, pipline3 = [], [], []
    pipline.append({'$match': {'IndicatorName': "Broad money growth (annual %)", "CountryCode": "KOR",
                               'Year': {'$gte': 1995}}})
    pipline.append({'$project': {'_id': 0, 'Value': 1, 'Year': 1}})

    pipline3.append(
        {'$match': {'IndicatorName': "GDP growth (annual %)", 'CountryCode': "KOR", 'Year': {'$gte': 2002}}})
    pipline3.append({'$project': {'_id': 0, 'Value': 1, 'Year': 1}})
    result = collection2.aggregate(pipline)
    result3 = collection2.aggregate(pipline3)
    broad, gdp = list(), list()
    for i in result:
        broad.append(round(i['Value'], 2))
    for i in result3:
        gdp.append(round(i['Value'], 2))

    chart4 = {'x': broad,
              'y': gdp}
    chart4 = pd.DataFrame(chart4)
    print("chart4")
    print(chart4)

    linear_regression = LinearRegression()
    linear_regression.fit(X=pd.DataFrame(chart4["x"]), y=chart4["y"])
    prediction = linear_regression.predict(X=pd.DataFrame(chart4["x"]))
    trace3 = go.Scatter(x=broad, y=gdp,
                        mode="markers")
    trace4 = go.Scatter(x=broad, y=prediction, mode="lines")
    chart4 = go.Data([trace3, trace4])
    layout = go.Layout(
        autosize=True,
    )
    fig4 = go.Figure(data=chart4, layout=layout)
    plot4 = opy.plot(fig4, output_type='div')

    gep_emp = []
    pipline = []
    pipline.append({'$match': {'IndicatorName': "GDP growth (annual %)", 'CountryCode': "KOR", 'Year': {'$gte': 2001}}})
    pipline.append({'$project': {'_id': 0, 'Value': 1, 'Year': 1}})
    pipline.append({'$limit': 13})
    result = collection2.aggregate(pipline)
    for i in result:
        gep_emp.append(round(i['Value'], 2))
    pipline = []
    pipline.append({'$match': {'IndicatorName': "GDP growth (annual %)", 'CountryCode': "USA", 'Year': {'$gte': 2001}}})
    pipline.append({'$project': {'_id': 0, 'Value': 1, 'Year': 1}})
    pipline.append({'$limit': 13})
    result = collection2.aggregate(pipline)
    for i in result:
        gep_emp.append(round(i['Value'], 2))
    pipline = []
    pipline.append({'$match': {'IndicatorName': "GDP growth (annual %)", 'CountryCode': "JPN", 'Year': {'$gte': 2001}}})
    pipline.append({'$project': {'_id': 0, 'Value': 1, 'Year': 1}})
    pipline.append({'$limit': 13})
    result = collection2.aggregate(pipline)
    for i in result:
        gep_emp.append(round(i['Value'], 2))
    pipline = []
    pipline.append({'$match': {'IndicatorName': "GDP growth (annual %)", 'CountryCode': "CHN", 'Year': {'$gte': 2001}}})
    pipline.append({'$project': {'_id': 0, 'Value': 1, 'Year': 1}})
    pipline.append({'$limit': 13})
    result = collection2.aggregate(pipline)
    for i in result:
        gep_emp.append(round(i['Value'], 2))

    df = pd.read_csv("./csv/unemployment.csv", index_col="Country Code")
    df = df.drop('Country Name', axis=1)
    kor = df.loc['KOR'].values
    kor = np.round(kor, 2)
    usa = df.loc['USA'].values
    usa = np.round(usa, 2)
    jpn = df.loc['JPN'].values
    jpn = np.round(jpn, 2)
    chn = df.loc['CHN'].values
    chn = np.round(chn, 2)
    emp = np.concatenate((kor, usa, jpn, chn), axis=0)

    model = LinearRegression()
    model = model.fit(X=pd.DataFrame(gep_emp), y=emp)
    prediction2 = model.predict(X=pd.DataFrame(gep_emp))
    trace5 = go.Scatter(x=gep_emp, y=emp,
                        mode="markers")
    trace6 = go.Scatter(x=gep_emp, y=prediction2, mode="lines")
    chart5 = go.Data([trace5, trace6])
    layout = go.Layout(
        autosize=True,
    )
    fig5 = go.Figure(data=chart5, layout=layout)
    plot6 = opy.plot(fig5, output_type='div')

    return render(request, 'korea.html', {'plot_div': plot,
                                          'plot_div2': plot2,
                                          'plot_div3': plot3,
                                          'plot_div4': plot4,
                                          'plot_div5': plot5,
                                          'plot_div6': plot6,
                                          })


def GDP_growth(request):
    pipline, pipline2, pipline3, pipline4 = list(), list(), list(), list()
    pipline.append({'$match': {'IndicatorName': "GDP growth (annual %)", 'Year': 2014}})
    pipline.append({'$project': {'_id': 0, 'CountryName': 1, 'Value': 1, 'CountryCode': 1}})
    result = collection2.aggregate(pipline)

    results = collection2.find({'IndicatorName': 'GDP growth (annual %)'}, {'_id': 0})
    value, CountryName, CountryCode = [], [], []
    for i in result:
        value.append(i['Value'])
        CountryName.append(i['CountryName'])
        CountryCode.append(i['CountryCode'])

    fig = go.Figure(data=go.Choropleth(
        locations=CountryCode,
        z=value,
        text=CountryName,
        colorscale='Blues',
        autocolorscale=False,
        reversescale=True,

        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_tickprefix='',
        colorbar_title='GDP<br>Growth',
    ))

    fig.update_layout(
        title_text='GDP Growth',
        width=1300,
        height=700,
        autosize=False,
        margin=dict(l=300),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        ),
    )
    plot = opy.plot(fig, output_type='div')

    pipline2.append({'$match': {'IndicatorName': "GDP growth (annual %)", 'Year': 2014}})
    pipline2.append({'$project': {'_id': 0, 'CountryName': 1, 'Value': 1, 'CountryCode': 1}})
    pipline2.append({'$limit': 10})
    pipline2.append({'$sort': {'Value': -1}})
    result2 = collection2.aggregate(pipline2)
    c, v = [], []
    for i in result2:
        c.append(i['CountryCode'])
        v.append(i['Value'])
    data = [go.Scatter(x=c, y=v)]
    layout = go.Layout(
        autosize=True,

    )
    fig2 = go.Figure(data=data, layout=layout)
    plot2 = opy.plot(fig2, output_type='div')

    pipline3.append({'$match': {'IndicatorName': "Exports of goods and services (annual % growth)", 'Year': 2014}})
    pipline3.append({'$project': {'_id': 0, 'CountryName': 1, 'Value': 1, 'CountryCode': 1}})
    pipline3.append({'$limit': 10})
    pipline3.append({'$sort': {'Value': -1}})
    result3 = collection2.aggregate(pipline3)
    p3_c, p3_v = [], []
    for i in result3:
        p3_c.append(i['CountryCode'])
        p3_v.append(i['Value'])

    data2 = [go.Bar(x=p3_c, y=p3_v)]
    layout2 = go.Layout(
        autosize=True,

    )
    fig3 = go.Figure(data=data2, layout=layout2)
    plot3 = opy.plot(fig3, output_type='div')

    pipline4.append({'$match': {'IndicatorName': "Imports of goods and services (annual % growth)", 'Year': 2014}})
    pipline4.append({'$project': {'_id': 0, 'CountryName': 1, 'Value': 1, 'CountryCode': 1}})
    pipline4.append({'$limit': 10})
    pipline4.append({'$sort': {'Value': -1}})
    result4 = collection2.aggregate(pipline4)
    p4_c, p4_v = [], []
    for i in result4:
        p4_c.append(i['CountryCode'])
        p4_v.append(i['Value'])

    trace1 = go.Scatter(x=p4_c, y=p4_v, marker={'color': 'red', 'symbol': 104, 'size': 10},
                        mode="markers+lines")
    data3 = go.Data([trace1])
    layout3 = go.Layout(
        autosize=True
    )
    fig4 = go.Figure(data=data3, layout=layout3)
    plot4 = opy.plot(fig4, output_type='div')

    pipline, pipline2, pipline3, pipline4 = list(), list(), list(), list()
    pipline.append({'$match': {'IndicatorName': "GDP growth (annual %)", 'CountryCode': "KOR", 'Year': {'$gte': 1994}}})
    pipline.append({'$project': {'Value': 1}})
    pipline2.append(
        {'$match': {'IndicatorName': "GDP growth (annual %)", 'CountryCode': "USA", 'Year': {'$gte': 1994}}})
    pipline.append({'$project': {'Value': 1}})
    pipline3.append(
        {'$match': {'IndicatorName': "GDP growth (annual %)", 'CountryCode': "CHN", 'Year': {'$gte': 1994}}})
    pipline.append({'$project': {'Value': 1}})
    pipline4.append(
        {'$match': {'IndicatorName': "GDP growth (annual %)", 'CountryCode': "JPN", 'Year': {'$gte': 1994}}})
    pipline.append({'$project': {'Value': 1}})

    result = collection2.aggregate(pipline)
    result2 = collection2.aggregate(pipline2)
    result3 = collection2.aggregate(pipline3)
    result4 = collection2.aggregate(pipline4)
    kor, chn, usa, jpn = list(), list(), list(), list()
    for i in result:
        kor.append(round(i['Value'], 2))

    for i in result2:
        usa.append(round(i['Value'], 2))
    for i in result3:
        chn.append(round(i['Value'], 2))
    for i in result4:
        jpn.append(round(i['Value'], 2))
    hist_data = [kor, usa, chn, jpn]
    labels = ['kor', 'usa', 'chn', 'jpn']
    print(kor)

    fig5 = ff.create_distplot(hist_data, labels, bin_size=[.1, .25, .5, 1])
    fig5.update_layout(title_text='5Country GDP Growth',
                       autosize=True)

    plot5 = opy.plot(fig5, output_type='div')

    df2 = pd.read_csv("./csv/GDP_Growth.csv")
    df1 = pd.read_csv("./csv/GEPData.csv")
    Country_Code1 = df1['Country Code']
    Country_Code2 = df2['Country Code']
    df2 = pd.read_csv("./csv/GDP_Growth.csv", index_col="Country Code")
    df1 = pd.read_csv("./csv/GEPData.csv", index_col="Country Code")
    data2 = pd.DataFrame(columns=('GEP', 'GDP'))
    Country_Code3 = []
    print(Country_Code2)
    print(Country_Code1)
    for i in Country_Code2:
        for j in Country_Code1:
            if i == j:
                Country_Code3.append(i)
    for i in Country_Code3:
        data2.loc[i] = [df1.loc[i, '2016'], df2.loc[i, '2016']]
    data2 = data2.reset_index(drop=True)

    data2 = data2.dropna(axis=0)
    data_point = data2.values
    kmeans = KMeans(n_clusters=3).fit(data_point)
    data2['cluster_id'] = kmeans.labels_
    trace8 = go.Scatter(x=data2['GEP'], y=data2['GDP'], mode="markers",
                        marker=dict(
                            color=(data2['cluster_id'] == 1).astype('int'),

                        ))
    data5 = go.Data([trace8])
    fig6 = go.Figure(data=data5, layout=layout)
    plot6 = opy.plot(fig6, output_type='div')

    df2 = pd.read_csv("./csv/capital.csv")
    df1 = pd.read_csv("./csv/GEPData.csv")
    Country_Code1 = df1['Country Code']
    Country_Code2 = df2['Country Code']
    df2 = pd.read_csv("./csv/capital.csv", index_col="Country Code")
    df1 = pd.read_csv("./csv/GEPData.csv", index_col="Country Code")
    data2 = pd.DataFrame(columns=('capital', 'GDP'))
    Country_Code3 = []
    for i in Country_Code2:
        for j in Country_Code1:
            if i == j:
                Country_Code3.append(i)
    for i in Country_Code3:
        data2.loc[i] = [df1.loc[i, '2016'], df2.loc[i, '2016']]
    data2 = data2.reset_index(drop=True)
    data2 = data2.dropna(axis=0)

    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(X=pd.DataFrame(data2["capital"]), y=data2["GDP"])
    prediction = linear_regression.predict(X=pd.DataFrame(data2["capital"]))
    trace9 = go.Scatter(x=data2['capital'], y=data2['GDP'], mode="markers",name="capital&GDP")
    trace10 = go.Scatter(x=data2['capital'], y=prediction, mode="lines", name="Linear")
    data7 = go.Data([trace9, trace10])
    layout = go.Layout(
        autosize=True,
    )
    fig7 = go.Figure(data=data7, layout=layout)
    plot7= opy.plot(fig7, output_type='div')

    df2 = pd.read_csv("./csv/CO2_emissions.csv")
    df1 = pd.read_csv("./csv/greenhouse.csv")
    Country_Code1 = df1['Country Code']
    Country_Code2 = df2['Country Code']
    df2 = pd.read_csv("./csv/CO2_emissions.csv", index_col="Country Code")
    df1 = pd.read_csv("./csv/greenhouse.csv", index_col="Country Code")
    data2 = pd.DataFrame(columns=('greenhouse', 'co2'))
    Country_Code3 = []
    for i in Country_Code2:
        for j in Country_Code1:
            if i == j:
                Country_Code3.append(i)
    for i in Country_Code3:
        data2.loc[i] = [df1.loc[i, '2012'], df2.loc[i, '2012']]
    data2 = data2.reset_index(drop=True)
    data2 = data2.dropna(axis=0)
    data_point = data2.values
    kmeans = KMeans(n_clusters=3).fit(data_point)
    data2['cluster_id'] = kmeans.labels_
    trace11 = go.Scatter(x=data2['greenhouse'], y=data2['co2'], mode="markers",
                        marker=dict(
                            color=(data2['cluster_id'] == 1).astype('int')
                        ))

    data8 = go.Data([trace11])
    fig8 = go.Figure(data=data8,layout=layout)
    plot8 = opy.plot(fig8,output_type='div')

    return render(request, 'dashboard.html', {'plot_div': plot,
                                              'plot_div2': plot2,
                                              'plot_div3': plot3,
                                              'plot_div4': plot4,
                                              'plot_div5': plot5,
                                              'plot_div6': plot6,
                                              'plot_div7': plot7,
                                              'plot_div8':plot8})


def unemploy(request):
    client = MongoClient("localhost", 27017)
    database = client.bigdata
    collection2 = database.indicator
    gep, pipline, dep,kor_gep,pop,age,tax,kor_pop = [], [], [],[],[],[],[],[]
    pipline.append(
        {'$match': {'IndicatorName': "Population growth (annual %)", 'CountryCode': "KOR", 'Year': {'$gte': 2001}}})
    pipline.append({'$project': {'_id': 0, 'Value': 1, 'Year': 1}})
    pipline.append({'$limit': 13})
    result = collection2.aggregate(pipline)
    for i in result:
        pop.append(round(i['Value'], 2))
        kor_gep.append(round(i['Value'], 2))

    pipline = []
    pipline.append(
        {'$match': {'IndicatorName': "Population growth (annual %)", 'CountryCode': "USA", 'Year': {'$gte': 2001}}})
    pipline.append({'$project': {'_id': 0, 'Value': 1, 'Year': 1}})
    pipline.append({'$limit': 13})
    result = collection2.aggregate(pipline)
    for i in result:
        kor_pop.append(round(i['Value'], 2))

    pipline= []
    pipline.append(
        {'$match': {'IndicatorName': "Population growth (annual %)", 'CountryCode': "USA", 'Year': {'$gte': 2001}}})
    pipline.append({'$project': {'_id': 0, 'Value': 1, 'Year': 1}})
    pipline.append({'$limit': 13})
    result = collection2.aggregate(pipline)
    for i in result:
        pop.append(round(i['Value'], 2))
    pipline = []
    pipline.append(
        {'$match': {'IndicatorName': "Population growth (annual %)", 'CountryCode': "JPN", 'Year': {'$gte': 2001}}})
    pipline.append({'$project': {'_id': 0, 'Value': 1, 'Year': 1}})
    pipline.append({'$limit': 13})
    result = collection2.aggregate(pipline)
    for i in result:
        pop.append(round(i['Value'], 2))
    pipline = []
    pipline.append(
        {'$match': {'IndicatorName': "Population growth (annual %)", 'CountryCode': "CHN", 'Year': {'$gte': 2001}}})
    pipline.append({'$project': {'_id': 0, 'Value': 1, 'Year': 1}})
    pipline.append({'$limit': 13})
    result = collection2.aggregate(pipline)
    for i in result:
        pop.append(round(i['Value'], 2))

    pipline,year = [],[]
    pipline.append({'$match': {'IndicatorName': "Age dependency ratio, young (% of working-age population)",
                                   'CountryCode': "KOR", 'Year': {'$gte': 2001}}})
    pipline.append({'$project': {'_id': 0, 'Value': 1, 'Year': 1}})
    pipline.append({'$limit': 13})
    result = collection2.aggregate(pipline)
    for i in result:
         dep.append(round(i['Value'], 2))
         year.append(i['Year'])
    pipline = []
    pipline.append({'$match': {'IndicatorName': "Population ages 65 and above (% of total)",
                               'CountryCode': "KOR", 'Year': {'$gte': 2001}}})
    pipline.append({'$project': {'_id': 0, 'Value': 1, 'Year': 1}})
    pipline.append({'$limit': 13})
    result = collection2.aggregate(pipline)
    for i in result:
        age.append(round(i['Value'], 2))
    pipline = []
    pipline.append({'$match': {'IndicatorName': "Taxes on goods and services (% of revenue)",
                               'CountryCode': "KOR", 'Year': {'$gte': 2001}}})
    pipline.append({'$project': {'_id': 0, 'Value': 1, 'Year': 1}})
    pipline.append({'$limit': 13})
    result = collection2.aggregate(pipline)
    for i in result:
        tax.append(round(i['Value'], 2))


    df = pd.read_csv("./csv/unemployment.csv", index_col="Country Code")
    df = df.drop('Country Name', axis=1)
    kor = df.loc['KOR'].values
    kor = np.round(kor, 2)
    usa = df.loc['USA'].values
    usa = np.round(usa, 2)
    jpn = df.loc['JPN'].values
    jpn = np.round(jpn, 2)
    chn = df.loc['CHN'].values
    chn = np.round(chn, 2)
    unemp = np.concatenate((kor, usa, jpn, chn), axis=0)

    model = LinearRegression()
    model = model.fit(X=pd.DataFrame(pop), y=unemp)
    prediction2 = model.predict(X=pd.DataFrame(pop))
    print(model.coef_, model.intercept_)

    residuals = unemp - prediction2
    SSE = (residuals ** 2).sum()
    SST = ((pd.DataFrame(pop) - pd.DataFrame(pop).mean()) ** 2).sum()
    R_squared = 1 - (SSE / SST)
    print('R_squared = ', R_squared)
    print('score = ', model.fit(X=pd.DataFrame(pop), y=unemp))
    print('Mean_Squared_Error = ', mean_squared_error(prediction2, unemp))
    print('RMSE = ', mean_squared_error(prediction2, unemp))
    print('RMSE = ', mean_squared_error(prediction2, unemp) ** 0.5)
    trace1 = go.Scatter(x=pop, y=unemp,
                        mode="markers")
    trace2 = go.Scatter(x=pop, y=prediction2, mode="lines")
    data = go.Data([trace1, trace2])
    layout = go.Layout(
        autosize=True,
    )
    trace3 = go.Scatter(x=year, y=kor,
                        mode="lines", name='Unemployment')
    trace4 = go.Scatter(x=year, y=dep, mode="lines", name='deployment')
    trace5 = go.Scatter(x=year, y=kor_gep, mode="lines", name='GEP_growth')
    data2 = go.Data([trace3, trace4,trace5])
    layout2 = go.Layout(
        autosize=True,
    )

    fig = go.Figure(data=data, layout=layout)
    plot = opy.plot(fig, output_type='div')
    fig2 = go.Figure(data=data2, layout=layout2)
    plot2 = opy.plot(fig2, output_type='div')

    pierson=[]
    pierson.append(kor_pop)
    pierson.append(kor)
    pierson.append(age)
    pierson.append(tax)
    pierson.append(dep)

    emp_re = pd.DataFrame(pierson).T
    corr2 = emp_re.corr(method='pearson')
    print(corr2)
    corr2 = corr2.values.tolist()
    indicator = ['pop_growth', 'Unemployment', 'Age', 'TAX', 'dep']
    fig3 = go.Figure(data=go.Heatmap(
        z=corr2,
        x=indicator,
        y=indicator))
    fig3.update_layout(
        autosize=True,
    )
    plot3 = opy.plot(fig3, output_type='div')

    gev_debt, pipline = [], []
    pipline.append({'$match': {'Country Code': "USA"}})
    pipline.append(
        {'$project': {'_id': 0, 'Indicator Name': 0, 'Country Code': 0, 'Indicator Code': 0, 'Country Name': 0}})
    result = collection3.aggregate(pipline)
    for i in result:
        gev_debt = (list(i.values()))
    pipline = []
    pipline.append({'$match': {'Country Code': "JPN"}})
    pipline.append(
        {'$project': {'_id': 0, 'Indicator Name': 0, 'Country Code': 0, 'Indicator Code': 0, 'Country Name': 0}})
    result = collection3.aggregate(pipline)
    for i in result:
        gev_debt.extend(list(i.values()))
    pipline = []
    pipline.append({'$match': {'Country Code': "GBR"}})
    pipline.append(
        {'$project': {'_id': 0, 'Indicator Name': 0, 'Country Code': 0, 'Indicator Code': 0, 'Country Name': 0}})
    result = collection3.aggregate(pipline)
    for i in result:
        gev_debt.extend(list(i.values()))


    usa = df.loc['USA'].values
    jpn = df.loc['JPN'].values
    gbr = df.loc['GBR'].values
    unemp2 = []
    unemp2.extend(usa)
    unemp2.extend(jpn)
    unemp2.extend(gbr)
    print(gev_debt)
    print(unemp)
    print(len(gev_debt), len(unemp2))


    model = LinearRegression()
    model = model.fit(X=pd.DataFrame(gev_debt), y=unemp2)
    prediction2 = model.predict(X=pd.DataFrame(gev_debt))
    print(model.coef_, model.intercept_)

    residuals = unemp2 - prediction2
    SSE = (residuals ** 2).sum()
    SST = ((pd.DataFrame(gev_debt) - pd.DataFrame(gev_debt).mean()) ** 2).sum()
    R_squared = 1 - (SSE / SST)
    print('R_squared = ', R_squared)
    print('score = ', model.fit(X=pd.DataFrame(gev_debt), y=unemp2))
    print('Mean_Squared_Error = ', mean_squared_error(prediction2, unemp2))
    print('RMSE = ', mean_squared_error(prediction2, unemp2))
    print('RMSE = ', mean_squared_error(prediction2, unemp2) ** 0.5)
    trace6 = go.Scatter(x=gev_debt, y=unemp2,
                        mode="markers")
    trace7 = go.Scatter(x=gev_debt, y=prediction2, mode="lines")
    data4 = go.Data([trace6, trace7])
    layout3 = go.Layout(
        autosize=True,
    )
    fig4 = go.Figure(data=data4, layout=layout3)
    plot4 = opy.plot(fig4, output_type='div')

    #Unemployment &GEP Growth k-means
    df2 = pd.read_csv("./csv/unemployment_rate.csv")
    df1 = pd.read_csv("./csv/GEPData.csv")
    Country_Code1 = df1['Country Code']
    Country_Code2 = df2['Country Code']
    df2 = pd.read_csv("./csv/unemployment_rate.csv", index_col="Country Code")
    df1 = pd.read_csv("./csv/GEPData.csv", index_col="Country Code")
    df3 = pd.read_csv("./csv/inflation.csv", index_col="Country Code")
    df4 = pd.read_csv("./csv/domestic_company.csv", index_col="Country Code")
    df5 = pd.read_csv("./csv/capital.csv", index_col="Country Code")
    df6 = pd.read_csv("./csv/External_debt.csv", index_col="Country Code")
    data = pd.DataFrame(columns=('x', 'y'))
    data2 = pd.DataFrame(columns=('gep', 'unemp','inflation','domestic','capital','debt'))
    count = 0
    Country_Code3 = []
    print(Country_Code2)
    print(Country_Code1)
    for i in Country_Code2:
        for j in Country_Code1:
            if i == j:
                Country_Code3.append(i)
    for i in Country_Code3:
        data2.loc[i] = [df1.loc[i, '2016'], df2.loc[i, '2016'], df3.loc[i, '2016'], df4.loc[i, '2016'],
                       df5.loc[i, '2016'], df6.loc[i, '2016']]
    data2 = data2.reset_index(drop=True)
    corr2 = data2.corr(method='pearson')

    for i in Country_Code3:
        data.loc[i] = [df1.loc[i, '2016'], round(df2.loc[i, '2016'], 2)]
    data = data.dropna(axis=0)
    data_point = data.values
    kmeans = KMeans(n_clusters=3).fit(data_point)
    data['cluster_id'] = kmeans.labels_
    trace8 = go.Scatter(x=data['x'],y=data['y'],mode="markers",
                        marker=dict(
                            color=(data['cluster_id']==1).astype('int'),

                        ))
    data5 = go.Data([trace8])
    layout3 = go.Layout(
        autosize=True,
    )
    fig5 = go.Figure(data=data5, layout=layout3)
    plot5 = opy.plot(fig5, output_type='div')
    corr2 = corr2.values.tolist()
    indicator2 = ['GEP', 'Unemp', 'inflation', 'domestic', 'captial','Debt']
    fig6 = go.Figure(data=go.Heatmap(
        z=corr2,
        x=indicator2,
        y=indicator2))
    fig6.update_layout(
        autosize=True,
    )
    plot6 = opy.plot(fig6, output_type='div')

    pipline2 = []
    pipline2.append({'$match': {'IndicatorName': "Population growth (annual %)", 'Year': 2014}})
    pipline2.append({'$project': {'_id': 0, 'CountryName': 1, 'Value': 1, 'CountryCode': 1}})
    pipline2.append({'$limit': 10})
    pipline2.append({'$sort': {'Value': -1}})
    result2 = collection2.aggregate(pipline2)
    c, v = [], []
    for i in result2:
        c.append(i['CountryCode'])
        v.append(i['Value'])
    data7 = [go.Scatter(x=c, y=v)]
    layout = go.Layout(
        autosize=True,

    )
    fig7 = go.Figure(data=data7, layout=layout)
    plot7 = opy.plot(fig7, output_type='div')

    pipline2 = []
    pipline2.append({'$match': {'IndicatorName': "Unemployment, total (% of total labor force)", 'Year': 2014}})
    pipline2.append({'$project': {'_id': 0, 'CountryName': 1, 'Value': 1, 'CountryCode': 1}})
    pipline2.append({'$limit': 10})
    pipline2.append({'$sort': {'Value': -1}})
    result2 = collection2.aggregate(pipline2)
    c, v = [], []
    for i in result2:
        c.append(i['CountryCode'])
        v.append(i['Value'])
    data8 = [go.Bar(x=c, y=v)]
    fig8 = go.Figure(data=data8, layout=layout)
    plot8 = opy.plot(fig8, output_type='div')
    pipline2 = []
    pipline2.append({'$match': {'IndicatorName': "Population ages 65 and above (% of total)", 'Year': 2014}})
    pipline2.append({'$project': {'_id': 0, 'CountryName': 1, 'Value': 1, 'CountryCode': 1}})
    pipline2.append({'$limit': 10})
    pipline2.append({'$sort': {'Value': -1}})
    result2 = collection2.aggregate(pipline2)
    c, v = [], []
    for i in result2:
        c.append(i['CountryCode'])
        v.append(i['Value'])
    data9 = [go.Bar(x=c, y=v)]
    fig9 = go.Figure(data=data9, layout=layout)
    plot9 = opy.plot(fig9, output_type='div')


    return render(request, 'unemployment.html', {'plot_div': plot,
                                                 'plot_div2': plot2,
                                                 'plot_div3':plot3,
                                                 'plot_div4':plot4,
                                                 'plot_div5':plot5,
                                                 'plot_div6':plot6,
                                                 'plot_div7':plot7,
                                                 'plot_div8':plot8,
                                                 'plot_div9':plot9})


def nearestCountry(request):
    df1 = pd.read_csv("./csv/Methane_emission.csv", index_col="Country Code")
    df1 = df1.drop(['Country Name', 'Indicator Name'], axis=1)
    df2 = pd.read_csv("./csv/CO2_emissions.csv", index_col="Country Code")
    df2 = df2.drop(['Country Name', 'Indicator Name'], axis=1)
    df3 = pd.read_csv("./csv/Electronic.csv", index_col="Country Code")
    df2 = df3.drop(['Country Name', 'Indicator Name'], axis=1)
    year = []
    for i in range(1990, 2016):
        year.append(i)
    print(len(year))
    co2 = {'kor': df1.loc['KOR', '1990':'2015'].values,
           'jpn': df1.loc['JPN', '1990':'2015'].values,
           'chn': df1.loc['CHN', '1990':'2015'].values,
           'year': year}
    co2_data = pd.DataFrame(co2, columns=['kor', 'jpn', 'chn', 'year'])
    co2_data = co2_data.dropna(axis=0)
    methan = {'kor': df2.loc['KOR', '1990':'2015'].values,
              'jpn': df2.loc['JPN', '1990':'2015'].values,
              'chn': df2.loc['CHN', '1990':'2015'].values,
              'year': year}
    methan_data = pd.DataFrame(methan, columns=['kor', 'jpn', 'chn', 'year'])
    methan_data = methan_data.dropna(axis=0)
    elec = {'kor': df3.loc['KOR', '1990':'2015'].values,
            'jpn': df3.loc['JPN', '1990':'2015'].values,
            'chn': df3.loc['CHN', '1990':'2015'].values,
            'year': year}
    elec_data = pd.DataFrame(elec, columns=['kor', 'jpn', 'chn', 'year'])
    elec_data = elec_data.dropna(axis=0)

    df1 = pd.read_csv("./csv/Stock_trade.csv", index_col="Country Code")
    df1 = df1.drop(['Country Name', 'Indicator Name'], axis=1)
    df2 = pd.read_csv("./csv/credit.csv", index_col="Country Code")
    df2 = df2.drop(['Country Name', 'Indicator Name'], axis=1)
    df3 = pd.read_csv("./csv/capital.csv", index_col="Country Code")
    stock = {'kor': df1.loc['KOR', '1990':'2015'].values,
             'jpn': df1.loc['JPN', '1990':'2015'].values,
             'chn': df1.loc['CHN', '1990':'2015'].values,
             'year': year}
    stock_data = pd.DataFrame(stock, columns=['kor', 'jpn', 'chn', 'year'])
    stock_data = stock_data.dropna(axis=0)
    credit = {'kor': df2.loc['KOR', '1990':'2015'].values,
              'jpn': df2.loc['JPN', '1990':'2015'].values,
              'chn': df2.loc['CHN', '1990':'2015'].values,
              'year': year}
    credit_data = pd.DataFrame(credit, columns=['kor', 'jpn', 'chn', 'year'])
    credit_data = credit_data.dropna(axis=0)
    cap = {'kor': df3.loc['KOR', '1990':'2015'].values,
           'jpn': df3.loc['JPN', '1990':'2015'].values,
           'chn': df3.loc['CHN', '1990':'2015'].values,
           'year': year}
    cap_data = pd.DataFrame(cap, columns=['kor', 'jpn', 'chn', 'year'])
    cap_data = cap_data.dropna(axis=0)

    df1 = pd.read_csv("./csv/Fertility_rate.csv", index_col="Country Code")
    df1 = df1.drop(['Country Name', 'Indicator Name'], axis=1)
    df2 = pd.read_csv("./csv/Life_expectancy.csv", index_col="Country Code")
    df2 = df2.drop(['Country Name', 'Indicator Name'], axis=1)
    df3 = pd.read_csv("./csv/Mortality.csv", index_col="Country Code")
    df3 = df3.drop(['Country Name', 'Indicator Name'], axis=1)
    fertility = {'kor': df1.loc['KOR', '1990':'2015'].values,
                'jpn': df1.loc['JPN', '1990':'2015'].values,
                'chn': df1.loc['CHN', '1990':'2015'].values,
                'year': year}
    fertility_data = pd.DataFrame(fertility, columns=['kor', 'jpn', 'chn', 'year'])
    fertility_data = fertility_data.dropna(axis=0)
    life = {'kor': df2.loc['KOR', '1990':'2015'].values,
            'jpn': df2.loc['JPN', '1990':'2015'].values,
            'chn': df2.loc['CHN', '1990':'2015'].values,
            'year': year}
    life_data = pd.DataFrame(life, columns=['kor', 'jpn', 'chn', 'year'])
    life_data = life_data.dropna(axis=0)
    mortality = {'kor': df3.loc['KOR', '1990':'2015'].values,
                 'jpn': df3.loc['JPN', '1990':'2015'].values,
                 'chn': df3.loc['CHN', '1990':'2015'].values,
                 'year': year}
    mor_data = pd.DataFrame(mortality, columns=['kor', 'jpn', 'chn', 'year'])
    mor_data = mor_data.dropna(axis=0)

    trace1 = go.Scatter(x=co2_data['year'], y=co2_data['kor'], mode="lines", name='KOR')
    trace2 = go.Scatter(x=co2_data['year'], y=co2_data['jpn'], mode="lines", name='JPN')
    trace3 = go.Scatter(x=co2_data['year'], y=co2_data['chn'], mode="lines", name='CHN')
    trace4 = go.Bar(x=methan_data['year'], y=methan_data['kor'],name='KOR')
    trace5 = go.Bar(x=methan_data['year'], y=methan_data['jpn'],  name='JPN')
    trace6 = go.Bar(x=methan_data['year'], y=methan_data['chn'], name='CHN')
    trace7 = go.Scatter(x=elec_data['year'], y=elec_data['kor'], name='KOR', mode='markers')
    trace8 = go.Scatter(x=elec_data['year'], y=elec_data['jpn'], name='JPN', mode='markers')
    trace9 = go.Scatter(x=elec_data['year'], y=elec_data['chn'], name='CHN', mode='markers')
    trace10 = go.Scatter(x=stock_data['year'], y=stock_data['kor'], mode="lines", name='KOR')
    trace11 = go.Scatter(x=stock_data['year'], y=stock_data['jpn'], mode="lines", name='JPN')
    trace12 = go.Scatter(x=stock_data['year'], y=stock_data['chn'], mode="lines", name='CHN')
    trace13 = go.Bar(x=credit_data['year'], y=credit_data['kor'], name='KOR')
    trace14 = go.Bar(x=credit_data['year'], y=credit_data['jpn'], name='JPN')
    trace15 = go.Bar(x=credit_data['year'], y=credit_data['chn'], name='CHN')
    trace16 = go.Scatter(x=cap_data['year'], y=cap_data['kor'], name='KOR', mode='markers')
    trace17 = go.Scatter(x=cap_data['year'], y=cap_data['jpn'], name='JPN', mode='markers')
    trace18 = go.Scatter(x=cap_data['year'], y=cap_data['chn'], name='CHN', mode='markers')
    trace19 = go.Scatter(x=fertility_data['year'], y=fertility_data['kor'], mode="lines", name='KOR')
    trace20 = go.Scatter(x=fertility_data['year'], y=fertility_data['jpn'], mode="lines", name='JPN')
    trace21 = go.Scatter(x=fertility_data['year'], y=fertility_data['chn'], mode="lines", name='CHN')
    trace22 = go.Bar(x=life_data['year'], y=life_data['kor'], name='KOR')
    trace23 = go.Bar(x=life_data['year'], y=life_data['jpn'], name='JPN')
    trace24 = go.Bar(x=life_data['year'], y=life_data['chn'], name='CHN')
    trace25 = go.Scatter(x=mor_data['year'], y=mor_data['kor'], name='KOR', mode='markers')
    trace26 = go.Scatter(x=mor_data['year'], y=mor_data['jpn'], name='JPN', mode='markers')
    trace27 = go.Scatter(x=mor_data['year'], y=mor_data['chn'], name='CHN', mode='markers')

    data1 = go.Data([trace1, trace2, trace3])
    data2 = go.Data([trace4, trace5, trace6])
    data3 = go.Data([trace7, trace8, trace9])
    data4 = go.Data([trace10, trace11, trace12])
    data5 = go.Data([trace13, trace14, trace15])
    data6 = go.Data([trace16, trace17, trace18])
    data7 = go.Data([trace19, trace20, trace21])
    data8 = go.Data([trace22, trace23, trace24])
    data9 = go.Data([trace25, trace26, trace27])


    layout = go.Layout(
        autosize=True,
    )
    fig1 = go.Figure(data=data1, layout=layout)
    fig2 = go.Figure(data=data2, layout=layout)
    fig3 = go.Figure(data=data3, layout=layout)
    fig4 = go.Figure(data=data4, layout=layout)
    fig5 = go.Figure(data=data5, layout=layout)
    fig6 = go.Figure(data=data6, layout=layout)
    fig7 = go.Figure(data=data7, layout=layout)
    fig8 = go.Figure(data=data8, layout=layout)
    fig9 = go.Figure(data=data9, layout=layout)
    plot = opy.plot(fig1, output_type='div')
    plot2 = opy.plot(fig2, output_type='div')
    plot3 = opy.plot(fig3, output_type='div')
    plot4 = opy.plot(fig4, output_type='div')
    plot5 = opy.plot(fig5, output_type='div')
    plot6 = opy.plot(fig6, output_type='div')
    plot7 = opy.plot(fig7, output_type='div')
    plot8 = opy.plot(fig8, output_type='div')
    plot9 = opy.plot(fig9, output_type='div')

    return render(request,'country3.html',{'plot_div':plot,
                                           'plot_div2':plot2,
                                           'plot_div3':plot3,
                                           'plot_div4':plot4,
                                           'plot_div5':plot5,
                                           'plot_div6':plot6,
                                           'plot_div7':plot7,
                                           'plot_div8':plot8,
                                           'plot_div9':plot9,
                                           })
