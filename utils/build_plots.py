import pandas as pd
import plotly.express as px
import json
from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler



def build_small_multiples(path_data, metric):

    geo_json_norte = json.load(open(path_data + 'geojson/regiao_norte.json'))
    geo_json_nordeste = json.load(open(path_data + 'geojson/regiao_nordeste.json'))
    geo_json_sudeste = json.load(open(path_data + 'geojson/regiao_sudeste.json'))
    geo_json_sul = json.load(open(path_data + 'geojson/regiao_sul.json'))
    geo_json_centrooeste = json.load(open(path_data + 'geojson/regiao_centro_oeste.json'))

    mygeojsons = {1: geo_json_norte, 2: geo_json_nordeste, 3: geo_json_sudeste, 4: geo_json_sul, 5: geo_json_centrooeste}

    df = pd.read_csv(path_data+'datasets/aggregations/df_new_month_uf.csv', encoding='utf-8', sep=';')

    months = list(df[['Mês', 'Mês_Nome']].drop_duplicates()['Mês_Nome'])  # Mantendo a ordem dos meses

    rows, cols = 5, len(months) # 5 regiões ao longo de x meses

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=list(months),
                        specs=[[{'type': 'choropleth'} for c in np.arange(cols)] for r in np.arange(rows)], )

    # Usuário seleciona a métrica que deseja acompanhar
    mycol = 'Novos_Casos' if metric == 'n_casos' else 'Novas_Mortes'

    for ro,month in enumerate(months):

        result = df[['UF',mycol]][df['Mês_Nome'] == month]

        for co,reg in enumerate(mygeojsons):

            fig.add_trace(
                go.Choropleth(geojson=mygeojsons[co + 1], locations=result['UF'], z=result[mycol],
                              marker_line_color='white', hoverinfo='location+z',
                              zmin=0, zmax=max(df[mycol]), colorbar_title='Nº '+mycol.replace('_', ' '),
                              colorscale=px.colors.diverging.Geyser),
                row=co+1, col=ro+1)

    fig.update_geos(fitbounds='locations', resolution=50, visible=False, showframe=False,
                    projection={'type': 'mercator'}, )

    for annotation in fig['layout']['annotations']: annotation['textangle'] = 0

    return fig


def build_heatmap(path_data, metric):

    df = pd.read_csv(path_data+'datasets/aggregations/df_new_day_uf.csv', encoding='utf-8', sep=';')
    # print(df.head())

    # Usuário seleciona a métrica que deseja acompanhar
    mycol = 'Novos_Casos' if metric == 'n_casos' else 'Novas_Mortes'

    df2 = df.sort_values(by=['Data', 'UF']).fillna(0) \
        .groupby('UF')[mycol].apply(list) \
        .reset_index(name=mycol)

    df2 = df2.set_index('UF')
    df2.index.name = None

    z = df2[mycol].tolist()
    myindex = df2.index
    mydates = pd.to_datetime(df['Data'].unique())
    fig = go.Figure(data=go.Heatmap(z=z, x=mydates, y=myindex, colorscale='Viridis'))
    fig.update_layout(title='Evolução diária por estado')

    return fig


def build_distribuicao_cidades(path_data, grain):

    df = pd.read_csv(path_data + 'datasets/aggregations/df_class_munic.csv', encoding='utf-8', sep=';')
    # df.head()

    # Usuário seleciona o grão do eixo x
    mygrain = 'UF' if grain == 'by_uf' else 'Região'

    df = df.dropna()
    df = df.sort_values(by=mygrain)

    color_discrete_map = {'Impacto_Baixo': px.colors.diverging.Geyser[0],
                          'Impacto_Médio': px.colors.diverging.Geyser[3],
                          'Impacto_Alto': px.colors.diverging.Geyser[6]}

    fig = px.strip(df, y='Taxa_Letalidade', stripmode='overlay', x=mygrain, color='Impacto',
                   color_discrete_map=color_discrete_map, )
    fig.add_scatter(x=df[mygrain], y=df['Taxa_Letalidade_Br'], mode='lines', marker={'color':px.colors.diverging.Geyser[3]},
                    name='Letalidade Brasil')

    fig.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
                      legend_title_text='', xaxis_title='', yaxis_title='', title='Letalidade por município')

    return fig


def build_rug_cidades(path_data):

    df = pd.read_csv(path_data + 'datasets/aggregations/df_class_munic.csv', encoding='utf-8', sep=';')
    df = df.dropna()

    df2 = df.groupby('Região')['Taxa_Letalidade','Município'].agg(list).reset_index()
    df2 = df2.set_index('Região')
    df2.index.name = None
    # print(df2.head())

    mydata = df2['Taxa_Letalidade'].tolist()
    mytext = df2['Município'].tolist()
    mylabels = df2.index
    mycolors = ['#333F44', '#37AA9C', '#94F3E4', '#eda147', '#dc3e00']

    fig = ff.create_distplot(mydata, mylabels, show_hist=False, colors=mycolors, rug_text=mytext)
    fig.update_layout(title_text='Distribuição das taxas de letalidade por região')

    # Adiciona média BR como referência
    k = df['Taxa_Letalidade_Br'].unique()[0]
    fig.add_shape(type="line", x0=k, x1=k, y0=0, y1=30, line=dict(dash="dot", color="rgb(171,171,170)", width=1))
    fig.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))

    return fig


def build_perc_mortes(path_data):

    df = pd.read_csv(path_data+'datasets/aggregations/df_let_uf.csv', encoding='utf-8', sep=';')

    df['%_Mortes'] = round(df['Mortes_Cumul'] / df['Casos_Cumul'], 4)
    df['%_Sem_Mortes'] = round((df['Casos_Cumul'] - df['Mortes_Cumul']) / df['Casos_Cumul'], 4)
    df = df.sort_values(by='%_Mortes')

    df2 = df.melt(id_vars='UF', value_vars=['%_Mortes', '%_Sem_Mortes'])

    fig = px.bar(df2, 'value', 'UF', color='variable', barmode='stack', orientation='h', template='simple_white',
                 color_discrete_map={'%_Mortes': '#ef553b', '%_Sem_Mortes': 'rgb(224,224,223)'},
                 title='Taxa de letalidade')

    # fig.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1), legend_title_text='')
    fig.update_layout(showlegend=False, xaxis_title='', yaxis_title='', )

    return fig


def build_matrix_corr(path_data, list_indicadores):

    # Base com a classif das cidades
    class_munic = pd.read_csv(path_data + 'datasets/aggregations/df_class_munic.csv', sep=';') \
                    .drop(columns=['Taxa_Letalidade', 'Taxa_Letalidade_Br', 'Desvio_Br'])


    df_1 = pd.read_csv(path_data + 'datasets/prepared/pd_df_ind_ibge_complete.csv', encoding='utf-8')
    df_1 = df_1[df_1['id_indicador'].isin(list_indicadores)]
    df_1.rename(columns={'id_cidade': 'Município_Cod', 'resultado': 'Valor_Indicador'}, inplace=True)

    # Amostragem estratificada de 40% das cidades de cada Impacto
    sample_cities = class_munic.groupby('Impacto', group_keys=False).apply(lambda x: x.sample(frac=0.4))

    # Join
    df_2 = sample_cities.merge(df_1[['nome_completo_indicador','id_indicador','Município_Cod','Valor_Indicador']],
                               on='Município_Cod', how='inner')

    df_3 = df_2.pivot(index='Município_Cod', columns='nome_completo_indicador', values='Valor_Indicador').reset_index() \
               .merge(sample_cities[['UF', 'Município_Cod', 'Município', 'Impacto', 'Casos_Cumul', 'Mortes_Cumul']],
                      on='Município_Cod', how='inner') \
               .drop(columns=['Município_Cod', 'UF', 'Município', 'Impacto'])

    for c in df_3:
        df_3[c] = df_3[c].astype('float')

    # Calcula correlação
    corr = df_3.corr()

    # Torna a matriz diagonal (Plotly não plota NaN)
    for i in range(len(corr.values)):
        for j in range(len(corr.values[0])):
            if i <= j:
                corr.values[i,j] = np.nan

    layout = go.Layout(template='simple_white', title='Matriz de correlação')
    data = [go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale=px.colors.diverging.Geyser)]
    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(type='category')
    fig.update_yaxes(type='category')

    return fig


def build_top10_indicadores(path_data, metric, ascending):

    corr_values = pd.read_csv(path_data + 'datasets/correlation/correlations_full.csv')
    corr_values.rename(columns={'Casos': 'Corr_Casos', 'Mortes': 'Corr_Mortes', 'Letalidade': 'Corr_Letalidade'},
                       inplace=True)

    details_ind = pd.read_csv(path_data + 'datasets/details_indicadores.csv', sep=';', encoding='utf-8')

    top10 = corr_values.merge(details_ind[['id_indicador', 'nome_completo_indicador']], on='id_indicador', how='inner')

    # Top 10 maior valor absoluto
    top10['abs_Corr_Casos'] = abs(top10['Corr_Casos'])
    top10['abs_Corr_Mortes'] = abs(top10['Corr_Mortes'])
    top10['abs_Corr_Letalidade'] = abs(top10['Corr_Letalidade'])

    # Usuário seleciona a métrica que deseja acompanhar
    if metric == 'n_casos':
        mycol,mycol2 = 'abs_Corr_Casos','Corr_Casos'
    else:
        if metric == 'n_mortes':
            mycol, mycol2 = 'abs_Corr_Mortes', 'Corr_Mortes'
        else:
            mycol, mycol2 = 'abs_Corr_Letalidade', 'Corr_Letalidade'

    if ascending == False:
        mytitle = 'Top 10 indicadores com maior correlação (em módulo)'
    else:
        mytitle = 'Top 10 indicadores com menor correlação (em módulo)'

    top10 = top10.sort_values(by=mycol, ascending=ascending)[0:10]

    fig = px.bar(top10, mycol2, 'nome_completo_indicador', orientation='h', title=mytitle)

    fig.update_layout(showlegend=False, xaxis_title='', yaxis_title='', )

    return fig


def build_film_strips(path_data, metric, indicador, impacto):

    # Base com a classif das cidades
    class_munic = pd.read_csv(path_data + 'datasets/aggregations/df_class_munic.csv', sep=';') \
        .drop(columns=['Taxa_Letalidade', 'Taxa_Letalidade_Br', 'Desvio_Br'])

    df_1 = pd.read_csv(path_data + 'datasets/prepared/pd_df_ind_ibge_complete.csv', encoding='utf-8')
    df_1 = df_1[df_1['id_indicador'] == indicador]
    df_1.rename(columns={'id_cidade': 'Município_Cod', 'resultado': 'Valor_Indicador'}, inplace=True)

    # Amostragem estratificada de 10% das cidades de cada Impacto
    if impacto == 'geral':
        sample_cities = class_munic.groupby('Impacto', group_keys=False).apply(lambda x: x.sample(frac=0.1))
    else:
        if impacto == 'baixo':
            sample_cities = class_munic[class_munic['Impacto']=='Impacto_Baixo']\
                                .groupby('Impacto', group_keys=False).apply(lambda x: x.sample(n=400))
        else:
            if impacto == 'medio':
                sample_cities = class_munic[class_munic['Impacto'] == 'Impacto_Médio'] \
                    .groupby('Impacto', group_keys=False).apply(lambda x: x.sample(n=400))
            else:
                sample_cities = class_munic[class_munic['Impacto'] == 'Impacto_Alto'] \
                    .groupby('Impacto', group_keys=False).apply(lambda x: x.sample(n=400))


    # Join
    df_2 = sample_cities.merge(df_1[['nome_completo_indicador', 'id_indicador', 'Município_Cod', 'Valor_Indicador']],
                               on='Município_Cod', how='inner')

    df_3 = df_2.pivot(index='Município_Cod', columns='nome_completo_indicador', values='Valor_Indicador').reset_index() \
        .merge(sample_cities[['UF', 'Município_Cod', 'Município', 'Impacto', 'Casos_Cumul', 'Mortes_Cumul']],
               on='Município_Cod', how='inner')

    name_col = df_2['nome_completo_indicador'].unique()[0]

    df_3[name_col] = df_3[name_col].astype('float')

    # Usuário seleciona a métrica que deseja acompanhar
    mycol = 'Casos_Cumul' if metric == 'n_casos' else 'Mortes_Cumul'

    # Min Max Scaler para passar ambas as variáveis para a mesma escala
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(df_3[[mycol, name_col]])
    df_scaled = pd.DataFrame(data=data, columns=[(mycol + '_scaled'), (name_col + '_scaled')])

    df_scaled['Neg'] = - df_scaled[name_col + '_scaled']
    df_scaled['UF'] = df_3['UF']
    df_scaled['Município'] = df_3['Município']
    df_scaled['Impacto'] = df_3['Impacto']
    df_scaled[mycol] = df_3[mycol]
    df_scaled[name_col] = df_3[name_col]

    # Exclui linhas fora do range [0,1]
    df_scaled = df_scaled[(df_scaled[mycol + '_scaled'] >= 0) & (df_scaled[mycol + '_scaled'] <= 1)]
    df_scaled = df_scaled[(df_scaled[name_col + '_scaled'] >= 0) & (df_scaled[name_col + '_scaled'] <= 1)]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_scaled['Município'].values, y=df_scaled['Neg'].values, name=mycol,
                         text=df_scaled[mycol], hovertemplate="%{x}: %{text}", ))
    fig.add_trace(go.Bar(x=df_scaled['Município'].values, y=df_scaled[name_col].values, name=name_col,
                         text=df_scaled[name_col], hovertemplate="%{x}: %{text}", ))

    fig.update_xaxes(type='category', showticklabels=False)
    fig.update_layout(barmode='relative', title='Correlação',)
    fig.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))

    return fig



def build_moving_average(path_data, metric, grain):

    # Usuário seleciona a métrica que deseja acompanhar
    mycol = 'Média_Móvel_Casos' if metric == 'n_casos' else 'Média_Móvel_Mortes'

    if grain == 'br':
        df = pd.read_csv(path_data + 'datasets/aggregations/df_mvg_avg_br.csv', encoding='utf-8', sep=';')
        fig = px.line(df, 'Data', mycol, title='Média Móvel 14 dias')
    else:
        df = pd.read_csv(path_data + 'datasets/aggregations/df_mvg_avg_uf.csv', encoding='utf-8', sep=';')
        fig = px.line(df, 'Data', mycol, color='UF', title='Média Móvel 14 dias')

    fig.update_layout(xaxis_title='', yaxis_title='')

    return fig