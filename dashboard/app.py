#!/usr/bin/env python3
import dash
import dash_auth
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import os


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, csrf_protect=False)



drive_path = 'data'
odds_path = os.path.join(drive_path, 'odds.xlsx')
betin_results_path = os.path.join(drive_path, 'all_leagues_results_20190525.xlsx')
odds_df = pd.read_excel(odds_path, index_col=0)
results_betin = pd.read_excel(betin_results_path ,index_col=0)

odds_new_cols = ['home_odds', 'draw_odds', 'away_odds', 'gg_odds', 'nogoal_odds',
       'over15_odds', 'under15_odds', '1/x_odds', '1/2_odds', 'x/2_odds',
       'over25_odds', 'under25_odds', 'over35_odds', 'under35_odds',
       'over45_odds', 'under45_odds', 'homeover05_odds',
       'homeunder05_odds', 'homeover15_odds', 'homeunder15_odds',
       'homeover25_odds', 'homeunder25_odds', 'homeover35_odds',
       'homeunder35_odds', 'awayover05_odds', 'awayunder05_odds',
       'awayover15_odds', 'awayunder15_odds', 'awayover25_odds',
       'awayunder25_odds', 'awayover35_odds', 'awayunder35_odds',
       'away_and_over_15_odds', 'draw_and_over_15_odds', 'home_and_over_15_odds',
       'away_and_over_25_odds', 'draw_and_over_25_odds', 'home_and_over_25_odds',
       'away_and_under_15_odds', 'draw_and_under_15_odds', 'home_and_under_15_odds',
       'away_and_under_25_odds', 'draw_and_under_25_odds', 'home_and_under_25_odds']

odds_df.columns = odds_new_cols

columns = ['MP', 'W', 'D','L', 'GF', 'GA', 'GD', 'Pts', 'Last 14']
teams = sorted(results_betin.team1.unique())

leagues_no = sorted(set([int(x.split('_')[1]) for x in results_betin.index.values]))

features = ['match_history_both', 'match_history_individual', 'over25']

features_1 = ['teams', 'team1', 'team2', 'match_result', 'result1', 'result2']
features_dt2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]

stats_cols = ['gg', 'nogoal', 'over15', 'over25', 'over35', 'over45', 'homeover05', 'homeover15', 'homeover25', 'homeover35', 'awayover05', 'awayover15', 'awayover25', 'awayover35','home_and_over_15', 'draw_and_over_15', 'away_and_over_15', 'home_and_over_25', 'draw_and_over_25', 'away_and_over_25']
stats_under_cols = ['under15', 'under25', 'under35', 'under45', 'homeunder05', 'homeunder15', 'homeunder25', 'homeunder35', 'awayunder05', 'awayunder15', 'awayunder25', 'awayunder35' ,'home_and_under_15', 'draw_and_under_15', 'away_and_under_15', 'home_and_under_25', 'draw_and_under_25', 'away_and_under_25']
stats_cols.extend(stats_under_cols)

features_1.extend(stats_cols)
features_1.extend(['league', 'week'])

def dropdown_week_selector():

    return [{'label' : f'week_{x+1}', 'value' : x+1} for x in range(38)]

def dropdown_league_selector():

    return [{'label' : f'league_{x}', 'value' : x} for x in leagues_no]

def dropdown_team_selector():

    return [{'label' : f'{x}', 'value' : f'{x}'} for x in teams]

def dropdown_feature_selector():

    return [{'label' : f'{x}', 'value' : f'{x}'} for x in features]

def dropdown_games_no_selector():

    games_no_dict = [{'label' : f'{x} games', 'value' : x} for x in range(38)]

    games_no_dict[0]['label'] = 'all games'

    return games_no_dict

def dropdown_investment_selector():

    return [{'label' : f'{x}', 'value' : f'{x}'} for x in np.arange(100, 2000, 200)]

def dropdown_home_away_selector():

    return [{'label' : f'{x}', 'value' : f'{x}'} for x in ['Home', 'Away', 'Both']]

def dropdown_target_history_selector():

    return [{'label' : f'{x}', 'value' : f'{x}'} for x in stats_cols]

dropdown_list_dicts_week = dropdown_week_selector()
dropdown_list_dicts_league = dropdown_league_selector()
dropdown_list_dicts_team = dropdown_team_selector()
dropdown_list_dicts_feature = dropdown_feature_selector()
dropdown_list_dicts_games_no = dropdown_games_no_selector()
dropdown_list_dicts_investment = dropdown_investment_selector()
dropdown_list_dicts_home_away = dropdown_home_away_selector()
dropdown_list_dicts_target_history = dropdown_target_history_selector()


def generate_league_table(league_no, week_no):
  
  league_df = results_betin.loc[results_betin.league == league_no]
  
  table = pd.DataFrame(np.zeros((len(teams), len(columns))), index=teams, columns=columns)

  table = table.astype(int)

  table['Last 14'] = table['Last 14'].map(str)

  for team in teams:

    GF = 0
    GA = 0

    league_df_temp = league_df.reset_index().set_index('week').loc[:week_no].reset_index().set_index('index')
    team_df_temp = league_df_temp.loc[league_df_temp.teams.str.contains(team)]

    try:
    
      wins = (team_df_temp.winner == team).sum()

      table.at[team, 'W'] = wins
    
    except:

      pass


    try:

=      draws = (team_df_temp.winner == 'D').sum()
      table.at[team, 'D'] = draws

    except:

      pass


    try:

      losses = team_df_temp.loc[~((team_df_temp.winner == team) | (team_df_temp.winner == 'D'))].shape[0]
      table.at[team, 'L'] = losses

    except:

      pass


    GF = league_df_temp.loc[league_df_temp.team1 == team, 'result1'].sum()
    GF += league_df_temp.loc[league_df_temp.team2 == team, 'result2'].sum()

    table.at[team, 'GF'] = GF

    GA = league_df_temp.loc[league_df_temp.team1 == team, 'result2'].sum()
    GA += league_df_temp.loc[league_df_temp.team2 == team, 'result1'].sum()

    table.at[team, 'GA'] = GA

    table.at[team, 'GD'] = np.abs(GF - GA)

    table.at[team, 'Pts'] = table.at[team, 'W'] * 3 + table.at[team, 'D'] * 1

    last_14 = league_df_temp.loc[(league_df_temp.teams.str.contains(team)), ['match_result','winner']][-5:]

    last_14['team_result'] = ''

    for index in last_14.index.values:

        if team == last_14.at[index, 'winner']:
            last_14.at[index, 'team_result'] = 'W'

        elif 'D' == last_14.at[index, 'winner']:
            last_14.at[index, 'team_result'] = 'D'

        else:
            last_14.at[index, 'team_result'] = 'L'
    
    last_14 = list(last_14.team_result.values)
    last_14.reverse()

    table.at[team, 'Last 14'] = '-'.join(last_14)

    table.at[team, 'MP'] = league_df_temp.loc[league_df_temp.sort_values(by='week').teams.str.contains(team), 'week'][-1]
    
  table.sort_values(by='Pts', ascending=False, inplace=True)

  table.index.name = 'TEAM'

  table.reset_index(inplace=True)
    
  return table
  
  
  

app.layout = html.Div(children=[

	html.Div([html.H6('Select League'),
            dcc.Dropdown(id='league-multi-dropdown',
                         options=dropdown_list_dicts_league,
                         # default='week_38',
                         multi=False),
            html.H6('Select Week'),
            dcc.Dropdown(id='week-multi-dropdown',
                         options=dropdown_list_dicts_week,
                         # default='week_38',
                         multi=False)]),

# 	dash_table.DataTable(


  html.Div(id='table-wrapper',
               style={'width': 'auto', 'overflow-y': 'scroll','max-width': '90vw', 'margin':'0 auto 50px'},
               children=dash_table.DataTable(id='datatable',
              columns=[{"name": i, "id": i} for i in ['TEAM'] + columns],
               # style_as_list_view=True,
                style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                style_cell={
                    'backgroundColor': 'rgb(50, 50, 50)',
                    'color': 'white'
                },
                # sorting=True,
                # sorting_type='multi',
                # filtering=True
                                                 )
                             ),

  html.Div([html.H6('Select Team 1'),
            dcc.Dropdown(id='team1-multi-dropdown',
                         options=dropdown_list_dicts_team,
                         # default='week_38',
                         multi=False),
            html.H6('Select Team 2'),
            dcc.Dropdown(id='team2-multi-dropdown',
                         options=dropdown_list_dicts_team,
                         # default='week_38',
                         multi=False),
            html.H6('Select No of games'),
            dcc.Dropdown(id='games-no-multi-dropdown',
                         options=dropdown_list_dicts_games_no,
                         # default='week_38',
                         multi=False),

            html.H6('Select Investment'),
            dcc.Dropdown(id='investment-multi-dropdown',
                         options=dropdown_list_dicts_investment,
                         # default='week_38',
                         multi=False),

            html.H6('Select Home/Away'),
            dcc.Dropdown(id='home-away-multi-dropdown',
                         options=dropdown_list_dicts_home_away,
                         # default='week_38',
                         multi=False)

            ]),
  html.H6('Stats'),

  html.Div(id='table-wrapper2',
               style={'width': 'auto', 'overflow-y': 'scroll','max-width': '90vw', 'margin':'0 auto 50px'},
               children=dash_table.DataTable(id='datatable2',
              columns=[{"name": i, "id": i} for i in features_dt2],
               # style_as_list_view=True,
                style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                style_cell={
                    'backgroundColor': 'rgb(50, 50, 50)',
                    'color': 'white'
                },
                # sorting=True,
                # sorting_type='multi',
                # filtering=True
                 )
                             ),

    

  html.H6('Select Target to view history'),
            dcc.Dropdown(id='target-history-multi-dropdown',
                         options=dropdown_list_dicts_target_history,
                         # default='week_38',
                         multi=True),

  html.H6('Match History'),

  html.Div(id='table-wrapper3',
               style={'width': 'auto', 'overflow-y': 'scroll','max-width': '90vw', 'margin':'0 auto 50px'},
               children=dash_table.DataTable(id='live-table',
              columns=[{"name": i, "id": i} for i in ['teams', 'team1', 'team2', 'match_result', 'goals', 'league', 'week']],
               style_as_list_view=True,
                style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                style_cell={
                    'backgroundColor': 'rgb(50, 50, 50)',
                    'color': 'white'
                },
                sorting=True,
                sorting_type='multi',
                filtering=True
                                                 )
                             ),

    
    ])

@app.callback(
  Output(component_id='datatable', component_property='data'),
  [Input(component_id='league-multi-dropdown', component_property='value'),
  Input(component_id='week-multi-dropdown', component_property='value'),]
  )

def update_graph(league_no, league_week):


  table = generate_league_table(league_no=league_no, week_no=league_week)

  return table.to_dict('records')

# Live Update of tables columns
@app.callback(
    Output('live-table', 'columns'),
    [Input('target-history-multi-dropdown', 'value')])

def update_graph(columns):

  features_hist = ['teams', 'team1', 'team2', 'match_result', 'goals', 'league', 'week']
  features_hist.extend(columns)

  return [{"name": i, "id": i} for i in features_hist]

@app.callback(
  Output(component_id='datatable2', component_property='data'),
  [Input(component_id='team1-multi-dropdown', component_property='value'),
  Input(component_id='team2-multi-dropdown', component_property='value'),
  Input(component_id='games-no-multi-dropdown', component_property='value'),
  Input(component_id='investment-multi-dropdown', component_property='value'),
  Input(component_id='home-away-multi-dropdown', component_property='value'),]
  )

def update_graph(team1, team2, no_games, investment, home_away):

  investment = int(investment)

  stats_df = pd.DataFrame(index=['stats', 'investment', 'profit', 'profit_%'], columns=stats_cols)

  teams_sep = [team1, team2]

  teams = '-'.join(teams_sep)

  teams_sep.reverse()

  teams_rev = '-'.join(teams_sep)

  temp_df = results_betin.loc[results_betin.teams.str.contains(teams) | results_betin.teams.str.contains(teams_rev), features_1][-no_games:]

  temp_df = temp_df.reset_index().set_index('teams').join(odds_df).reset_index().set_index('index')

  temp_df['investment'] = investment

  for target in stats_cols:
  
    if home_away == 'Home':

      stats_df.loc['investment'] = temp_df.loc[temp_df.team1 == team1, 'investment'].sum()

    
      vc_team1 = temp_df.loc[temp_df.team1 == team1, target].value_counts()

      teams_games_no = vc_team1.sum()

      try:

        team1_profit = vc_team1.loc[1] * odds_df.at[f'{team1}-{team2}', f'{target}_odds'] * investment
        rate_observed = (vc_team1.loc[1] / teams_games_no) * 100

      except:

        team1_profit = 0
        rate_observed = 0

      
      
      profit = (team1_profit) - (teams_games_no * investment)
      profit_perc = (profit / (teams_games_no * investment)) * 100

    elif home_away == 'Away':

      stats_df.loc['investment'] = temp_df.loc[temp_df.team1 == team2, 'investment'].sum()
    
      vc_team2 = temp_df.loc[temp_df.team1 == team2, target].value_counts()

      teams_games_no = vc_team2.sum()

      try:

        team2_profit = vc_team2.loc[1] * odds_df.at[f'{team2}-{team1}', f'{target}_odds'] * investment
        

        rate_observed = (vc_team2.loc[1] / teams_games_no) * 100

      except:

        rate_observed = 0
      
      profit = (team2_profit) - (teams_games_no * investment)
      profit_perc = (profit / (teams_games_no * investment)) * 100

    else:

      stats_df.loc['investment'] = temp_df['investment'].sum()
    
      vc_team1 = temp_df.loc[temp_df.team1 == team1, target].value_counts()
      vc_team2 = temp_df.loc[temp_df.team1 == team2, target].value_counts()

      try:

        teams_games_no = vc_team1.sum() + vc_team2.sum()

        team1_profit = vc_team1.loc[1] * odds_df.at[f'{team1}-{team2}', f'{target}_odds'] * investment
        team2_profit = vc_team2.loc[1] * odds_df.at[f'{team2}-{team1}', f'{target}_odds'] * investment
        

        rate_observed = (((vc_team1.loc[1] / vc_team1.sum()) * 100) + ((vc_team2.loc[1] / vc_team2.sum()) * 100))  / 2

      except:

        team1_profit = 0
        team2_profit = 0

        rate_observed = 0

      
      profit = (team1_profit + team2_profit) - (teams_games_no * investment)
      profit_perc = (profit / (teams_games_no * investment)) * 100

    stats_df.at['profit', target] = int(profit)
    stats_df.at['profit_%', target] = np.round(profit_perc, 2)
    stats_df.at['stats', target] = np.round(rate_observed, 2)


  home_goals = temp_df.result1.sum()
  away_goals = temp_df.result2.sum()

  team1_goals = temp_df.loc[temp_df.team1 == team1, 'result1'].sum() + temp_df.loc[temp_df.team2 == team1, 'result2'].sum()
  team2_goals = temp_df.loc[temp_df.team1 == team2, 'result1'].sum() + temp_df.loc[temp_df.team2 == team2, 'result2'].sum()

  team1_winrate = np.round(((temp_df.loc[temp_df.team1 == team1, 'match_result'] == 0).sum() + (temp_df.loc[temp_df.team2 == team1, 'match_result'] == 1).sum()) / temp_df.shape[0] * 100, 2)
  team2_winrate = np.round(((temp_df.loc[temp_df.team1 == team2, 'match_result'] == 0).sum() + (temp_df.loc[temp_df.team2 == team2, 'match_result'] == 1).sum()) / temp_df.shape[0] * 100, 2)
  draw_rate = np.round((temp_df.match_result == 2).sum() / temp_df.shape[0] * 100, 2)

  stats_df.index.name = 'rates'

  stats_df.sort_values(by='profit_%', axis=1, ascending=False, inplace=True)

  stats_df = stats_df.reset_index()

  stats_df = stats_df.T.reset_index().T  

  return stats_df.to_dict('records')

@app.callback(
  Output(component_id='live-table', component_property='data'),
  [Input(component_id='team1-multi-dropdown', component_property='value'),
  Input(component_id='team2-multi-dropdown', component_property='value'),
  Input(component_id='games-no-multi-dropdown', component_property='value'),
  Input(component_id='home-away-multi-dropdown', component_property='value'),
  Input('target-history-multi-dropdown', 'value'),]
  )

def update_graph(team1, team2, no_games, home_away, columns):

  features_hist = ['teams', 'team1', 'team2', 'match_result', 'result1', 'result2', 'league', 'week']
  features_hist.extend(columns)

  teams_sep = [team1, team2]

  teams = '-'.join(teams_sep)

  teams_sep.reverse()

  teams_rev = '-'.join(teams_sep)

  if home_away == 'Home':

    temp_df = results_betin.loc[results_betin.teams.str.contains(teams), features_hist][-no_games:]

  elif home_away == 'Away':

    temp_df = results_betin.loc[results_betin.teams.str.contains(teams_rev), features_hist][-no_games:]

  else:

    temp_df = results_betin.loc[results_betin.teams.str.contains(teams) | results_betin.teams.str.contains(teams_rev), features_hist][-no_games:]


  temp_df['winner'] = ''
  temp_df['goals'] = temp_df.result1.astype(str) + '-' + temp_df.result2.astype(str)

  for index in temp_df.index.values:
  
    if temp_df.at[index, 'match_result'] == 0:
      temp_df.at[index, 'winner'] = temp_df.at[index, 'team1']
      
    elif temp_df.at[index, 'match_result'] == 1:
      temp_df.at[index, 'winner'] = 'D'
      
    else:
      temp_df.at[index, 'winner'] = temp_df.at[index, 'team2']

  temp_df.drop(['match_result', 'result1', 'result2'], axis=1, inplace=True)

  temp_df.rename(columns={'winner':'match_result'}, inplace=True)

  temp_df['league'] = temp_df.league.astype(int) 
  temp_df['week'] = temp_df.week.astype(int)

  cols = ['team1', 'team2', 'match_result', 'goals']

  temp_df = pd.concat([temp_df[cols], temp_df.drop(cols, axis=1)], axis=1)

  return temp_df.to_dict('records')

if __name__ == '__main__':
    app.run_server(debug=True, port=8092)