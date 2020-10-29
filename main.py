import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output, State, MATCH, ALL
from utils import build_plots as bp

# ------------------------------
# Initial setup
# ------------------------------

path_data = 'D:/data_dash_covid/'
df = pd.read_csv(path_data+'datasets/prepared/pd_df_covid_complete.csv', encoding='utf-8', sep=';')
details_ind = pd.read_csv(path_data+'datasets/details_indicadores.csv', sep=';', encoding='utf-8')
list_indicadores = [30255,77881,28163,47001,29171,28120]


fig1 = bp.build_perc_mortes(path_data)
fig2 = bp.build_small_multiples(path_data, 'n_mortes')
fig3 = bp.build_heatmap(path_data, 'n_casos')
fig4 = bp.build_distribuicao_cidades(path_data, 'by_uf')
fig5 = bp.build_rug_cidades(path_data)
fig6 = bp.build_matrix_corr(path_data, list_indicadores)
fig7 = bp.build_top10_indicadores(path_data, 'n_casos', False)
fig8 = bp.build_film_strips(path_data, 'n_casos', 30255, 'geral', )
fig9 = bp.build_moving_average(path_data, 'n_casos', 'br')

# ------------------------------
# Dash app
# ------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], )

app.title = 'COVID Cidades'

app.layout = html.Div(
    className='container',
    children=[
        html.Div(
            className='divided',
            children=[
                html.Div(
                    className='mysection color_even', id='capa',
                    children=[
                        html.Div(
                            id='mytitleheader', className='item',
                            children=[
                                html.H1('Há algo em comum entre as cidades mais afetadas pela COVID-19?')
                            ]
                        ),
                        html.Div(
                            id='myimgheader', className='item',
                            children=[
                                html.Img(src='/assets/img/pessoas-bg.jpg')
                            ]
                        ),
                    ]
                ),
                html.Div(
                    className='mysection color_even', id='section_9_moving_avg',
                    children=[
                        html.Div(
                            className='item mysectiontext',
                            children=[
                                html.Div(
                                    className='explicacao',
                                    children=[
                                        html.H4(className='item', children=['Situação atual']),
                                        html.P(className='item', children=['O número de novos casos e novos óbitos devido à COVID-19 tem apresentado queda nas últimas semanas.']),
                                        html.P(className='item', children=['Avaliar a média do número de casos (ou mortes) dos últimos 14 dias é uma forma de identificar indicar se há tendência de alta, estabilidade ou queda.']),
                                        html.P(className='item', children=['Os dados são referentes ao período entre 27/03/2020 e 24/10/2020 inclusive.']),
                                    ]
                                ),
                                html.Div(
                                    className='myfilters',
                                    children=[
                                        dcc.RadioItems(
                                            id='metric_mv_avg', className='item myfilters_radio',
                                            options=[
                                                {'label': ' Nº Casos', 'value': 'n_casos'},
                                                {'label': ' Nº Mortes', 'value': 'n_mortes'},
                                            ],
                                            value='n_casos'
                                        ),
                                        dcc.RadioItems(
                                            id='grain_mv_avg', className='item myfilters_radio',
                                            options=[
                                                {'label': ' Brasil', 'value': 'br'},
                                                {'label': ' Por UF', 'value': 'by_uf'},
                                            ],
                                            value='br'
                                        )
                                    ]
                                ),
                            ]
                        ),
                        html.Div(
                            className='item mysectionchart',
                            children=[
                                dcc.Graph(
                                    className='item mygraphs', id='mygraph9',
                                    figure=fig9
                                )
                            ]
                        ),
                    ]
                ),
                html.Div(
                    className='mysection color_odd', id='section_1_def_letalidade',
                    children=[
                        html.Div(
                            className='item mysectiontext',
                            children=[
                                html.Div(
                                    className='explicacao',
                                    children=[
                                        html.H4(className='item', children=['Contaminação e letalidade']),
                                        html.P(className='item', children=['Os efeitos da COVID-19 podem ser quantificados tanto pelo número de casos quanto pela proporção de mortes dentre os casos confirmados.']),
                                        html.P(className='item', children=['Ao longo da nossa análise, serão considerados o nº de novos casos, o nº de novas mortes e a letalidade. A letalidade é a razão entre o total (cumulativo) de mortes sobre o total (cumulativo) de casos no último dia disponível.']),
                                    ]
                                ),
                            ]
                        ),
                        html.Div(
                            className='item mysectionchart',
                            children=[
                                dcc.Graph(
                                    className='item mygraphs', id='mygraph1',
                                    figure=fig1
                                )
                            ]
                        ),
                    ]
                ),
                html.Div(
                    className='mysection color_even', id='section_2_panorama_br',
                    children=[
                        html.Div(
                            className='item mysectiontext',
                            children=[
                                html.Div(
                                    className='explicacao',
                                    children=[
                                        html.H4(className='item', children=['Panorama Brasil']),
                                        html.P(className='item', children=[
                                            'É interessante observar a dinâmica de cada região com relação à doença desde o primeiro caso confirmado, na cidade de São Paulo, em fevereiro deste ano.']),
                                    ]
                                ),
                                html.Div(
                                    className='myfilters',
                                    children=[
                                        dcc.RadioItems(
                                            id='metric_small_multiples', className='myfilters_radio',
                                            options=[
                                                {'label': ' Nº Novos Casos', 'value': 'n_casos'},
                                                {'label': ' Nº Novas Mortes', 'value': 'n_mortes'},
                                            ],
                                            value='n_mortes'
                                        )
                                    ]
                                ),
                            ]
                        ),
                        html.Div(
                            className='item mysectionchart',
                            children=[
                                dcc.Graph(
                                    className='item mygraphs', id='mygraph2',
                                    figure=fig2
                                )
                            ]
                        ),
                    ]
                ),
                html.Div(
                    className='mysection color_odd', id='section_3_heatmap',
                    children=[
                        html.Div(
                            className='item mysectiontext',
                            children=[
                                html.Div(
                                    className='explicacao',
                                    children=[
                                        html.H4(className='item', children=['Evolução diária']),
                                        html.P(className='item', children=['Uma análise ainda mais detalhada pode ser realizada dia-a-dia.']),
                                        html.P(className='item', children=['Observe no gráfico pequenos blocos mais claros, evidenciando um padrão de subnotificação aos finais de semana.']),
                                    ]
                                ),
                                html.Div(
                                    className='myfilters',
                                    children=[
                                        dcc.RadioItems(
                                            id='metric_heatmap', className='myfilters_radio',
                                            options=[
                                                {'label': ' Nº Novos Casos', 'value': 'n_casos'},
                                                {'label': ' Nº Novas Mortes', 'value': 'n_mortes'},
                                            ],
                                            value='n_casos'
                                        )
                                    ]
                                ),
                            ]
                        ),
                        html.Div(
                            className='item mysectionchart',
                            children=[
                                dcc.Graph(
                                    className='item mygraphs', id='mygraph3',
                                    figure=fig3
                                )
                            ]
                        ),
                    ]
                ),
                html.Div(
                    className='mysection color_even', id='section_4_cidades',
                    children=[
                        html.Div(
                            className='item mysectiontext',
                            children=[
                                html.Div(
                                    className='explicacao',
                                    children=[
                                        html.H4(className='item', children=['E as cidades?']),
                                        html.P('Para entender melhor o comportamento das cidades, elas foram divididas em três grupos, de acordo com o quanto desviam da taxa de letalidade geral do Brasil, que é 0.0292.'),
                                        html.P('As cidades com letalidade variando menos de 10% em relação a 0.0292 são consideradas de Impacto Médio. As cidades com letalidade 10% abaixo de 0.0292 são de Impato Baixo, enquanto as acima de 10% são Impacto Alto.')
                                    ],
                                ),
                                html.Div(
                                    className='myfilters',
                                    children=[
                                        dcc.RadioItems(
                                            id='grain_distribution', className='myfilters_radio',
                                            options=[
                                                {'label': ' Por UF', 'value': 'by_uf'},
                                                {'label': ' Por Região', 'value': 'by_reg'},
                                            ],
                                            value='by_uf'
                                        )
                                    ]
                                ),
                            ]
                        ),
                        html.Div(
                            className='item mysectionchart',
                            children=[
                                dcc.Graph(
                                    className='item mygraphs', id='mygraph4',
                                    figure=fig4
                                )
                            ]
                        ),
                    ]
                ),
                html.Div(
                    className='mysection color_odd', id='section_5_cidades',
                    children=[
                        html.Div(
                            className='item mysectiontext',
                            children=[
                                html.Div(
                                    className='explicacao',
                                    children=[
                                        html.H4(className='item', children=['Distribuição das cidades']),
                                        html.P(className='item', children=['O Brasil tem 5.570 municípios. Como elas estão em relação ao todo?']),
                                        html.P(className='item', children=['Pelo gráfico de distribuição é possível notar que a maioria das cidades, em todas as regiões, está com letalidade abaixo da média geral.']),
                                    ]
                                ),
                            ]
                        ),
                        html.Div(
                            className='item mysectionchart',
                            children=[
                                dcc.Graph(
                                    className='item mygraphs', id='mygraph5',
                                    figure=fig5
                                )
                            ]
                        ),
                    ]
                ),
                html.Div(
                    className='mysection color_even', id='section_6_corr',
                    children=[
                        html.Div(
                            className='item mysectiontext',
                            children=[
                                html.Div(
                                    className='explicacao',
                                    children=[
                                        html.H4(className='item', children=['Há semelhanças?']),
                                        html.P(className='item', children=['Suspeitamos que as condição socioeconômicas, de saúde e de educação de cada município estão de alguma forma relacionadas aos impactos da COVID.']),
                                        html.P(className='item', children=['Dentre as centenas de indicadores das pesquisas do IBGE Cidades, selecionamos alguns para analisar primeiro.']),
                                    ]
                                ),
                                html.Div(
                                    className='myfilters',
                                    children=[
                                        dcc.Dropdown(
                                            id='indicadores_corr',
                                            options=[{'label': row.nome_completo_indicador, 'value': str(row.id_indicador)}
                                                      for _,row in details_ind.iterrows()],
                                            value=['30255','77881','28163','47001','29171','28120'],
                                            multi=True, optionHeight=100,
                                        )
                                    ]
                                ),
                            ]
                        ),
                        html.Div(
                            className='item mysectionchart',
                            children=[
                                dcc.Graph(
                                    className='item mygraphs', id='mygraph6',
                                    figure=fig6
                                )
                            ]
                        ),
                    ]
                ),
                html.Div(
                    className='mysection color_odd', id='section_7_top10',
                    children=[
                        html.Div(
                            className='item mysectiontext',
                            children=[
                                html.Div(
                                    className='explicacao',
                                    children=[
                                        html.H4(className='item',
                                                children=['Quais as mais decisivas?']),
                                        html.P(className='item', children=['Aparentemente a economia é o fator que mais interfere na capacidade de resposta do município à doença.']),
                                    ]
                                ),
                                html.Div(
                                    className='myfilters',
                                    children=[
                                        dcc.RadioItems(
                                            id='metric_top10', className='item myfilters_radio',
                                            options=[
                                                {'label': ' Nº Casos', 'value': 'n_casos'},
                                                {'label': ' Nº Mortes', 'value': 'n_mortes'},
                                                {'label': ' Taxa Letalidade', 'value': 'tx_letal'},
                                            ],
                                            value='n_casos'
                                        ),
                                        dcc.RadioItems(
                                            id='ascending_top10', className='item myfilters_radio',
                                            options=[
                                                {'label': ' Maior correlação', 'value': False},
                                                {'label': ' Menor correlação', 'value': True},
                                            ],
                                            value=False
                                        )
                                    ]
                                ),
                            ]
                        ),
                        html.Div(
                            className='item mysectionchart',
                            children=[
                                dcc.Graph(
                                    className='item mygraphs', id='mygraph7',
                                    figure=fig7
                                )
                            ]
                        ),
                    ]
                ),
                html.Div(
                    className='mysection color_even', id='section_8_strips',
                    children=[
                        html.Div(
                            className='item mysectiontext',
                            children=[
                                html.Div(
                                    className='explicacao',
                                    children=[
                                        html.H4(className='item',
                                                children=['Correlação individual']),
                                        html.P(className='item', children=['Ainda assim, é interessante verificar o perfil dos municípios através de outros indicadores.']),
                                        html.P(className='item', children=['Se determinado indicador é tanto maior quanto maiores as taxas de contaminação e letalidade do município, pode-se estimar uma certa correlação.']),
                                    ]
                                ),
                                html.Div(
                                    className='myfilters container',
                                    children=[
                                        dcc.RadioItems(
                                            id='metric_strips', className='item myfilters_radio',
                                            options=[
                                                {'label': ' Nº Casos', 'value': 'n_casos'},
                                                {'label': ' Nº Mortes', 'value': 'n_mortes'},
                                            ],
                                            value='n_casos'
                                        ),
                                        dcc.Dropdown(
                                            id='indicador_strips', className='item',
                                            options=[
                                                {'label': row.nome_completo_indicador, 'value': row.id_indicador}
                                                for _, row in details_ind.iterrows()],
                                            value=30255, optionHeight=100,
                                        ),
                                        dcc.RadioItems(
                                            id='impacto_strips', className='item myfilters_radio',
                                            options=[
                                                {'label': ' Geral', 'value': 'geral'},
                                                {'label': ' Cidades de Impacto Baixo', 'value': 'baixo'},
                                                {'label': ' Cidades de Impacto Médio', 'value': 'medio'},
                                                {'label': ' Cidades de Impacto Alto', 'value': 'alto'},
                                            ],
                                            value='geral'
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        html.Div(
                            className='item mysectionchart',
                            children=[
                                dcc.Graph(
                                    className='item mygraphs', id='mygraph8',
                                    figure=fig8
                                )
                            ]
                        ),
                    ]
                ),
                html.Div(
                    className='container', id='footer',
                    children=[
                        html.Div(
                            className='item', id='footer_1',
                            children=[
                                html.H4('Fontes dos dados'),
                                html.P(['Dados da COVID-19 no Brasil: ', html.A('Kaggle',href='https://www.kaggle.com/unanimad/corona-virus-brazil')]),
                                html.P(['Dados de indicadores dos municípios do Brasil: ', html.A('IBGE Cidades', href='https://servicodados.ibge.gov.br/api/docs/pesquisas?versao=1')]),
                            ]
                        ),
                        html.Div(
                            className='item', id='footer_2',
                            children=[
                                html.P('Departamento de Ciência da Computação - UFMG'),
                                html.P('Visualização de dados - Prof. Dra. Raquel Minardi'),
                                html.P('Carolinne Magalhães e Gestefane Magalhães | Outubro 2020'),
                            ]
                        ),
                    ]
                ),
            ]
        ),
    ]
)


@app.callback(Output('mygraph2', 'figure'),
              [Input('metric_small_multiples','value')])
def load_plot_small_multiples(metric):
    return bp.build_small_multiples(path_data, metric)

@app.callback(Output('mygraph3', 'figure'),
              [Input('metric_heatmap','value')])
def load_plot_heatmap(metric):
    return bp.build_heatmap(path_data, metric)

@app.callback(Output('mygraph4', 'figure'),
              [Input('grain_distribution','value')])
def load_plot_distribution(grain):
    return bp.build_distribuicao_cidades(path_data, grain)

@app.callback(Output('mygraph6', 'figure'),
              [Input('indicadores_corr','value')])
def load_plot_corr_matrix(list_indicadores):
    return bp.build_matrix_corr(path_data, list_indicadores)

@app.callback(Output('mygraph7', 'figure'),
              [Input('metric_top10','value'),
               Input('ascending_top10','value')])
def load_plot_top10(metric, ascending):
    return bp.build_top10_indicadores(path_data, metric, ascending)

@app.callback(Output('mygraph8', 'figure'),
              [Input('metric_strips','value'),
               Input('indicador_strips','value'),
               Input('impacto_strips','value'),])
def load_plot_film_strips(metric, indicador, impacto):
    return bp.build_film_strips(path_data, metric, indicador, impacto)


@app.callback(Output('mygraph9', 'figure'),
              [Input('metric_mv_avg','value'),
               Input('grain_mv_avg','value')])
def load_plot_moving_avg(metric, grain):
    return bp.build_moving_average(path_data, metric, grain)






if __name__ == '__main__':
    app.run_server(debug=False)
