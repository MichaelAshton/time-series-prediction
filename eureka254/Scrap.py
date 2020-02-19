import sys
sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os

import time as t

from slackclient import SlackClient

import psycopg2
import sqlalchemy

options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

db_username = os.environ["db_username"]
db_password = os.environ["db_password"]
db_ip = os.environ["db_ip"]
db_database = os.environ["db_database"]

scrappy_username = os.environ["scrappy_username"]
scrappy_password = os.environ["scrappy_password"]


engine = sqlalchemy.create_engine("postgresql+psycopg2://{}:{}@{}/{}".format(db_username, db_password, db_ip, db_database))

from eureka254.DB import DB

DB = DB()


class Scrap:
  
  def __init__(self, username=scrappy_username, password=scrappy_password, wd=webdriver.Chrome('chromedriver',options=options)):
    
    self.username=username
    self.password=password
    self.wd = wd
    self.weekly_cols = DB.get_table_header('weekly')

  # create folder helper function
  def _create_dir(self, path):
    if not os.path.exists(path):
      os.makedirs(path)
    
    
  def login(self):
    
    print('Loading login page')
    
    self.wd.get("https://sports.betin.co.ke/mobile#/login");

    # find element by css selector instead of by class as spaces are not yet supported in find_by_class_name
    username = self.wd.find_element_by_css_selector(".credentials__input.mt20")
    password = self.wd.find_element_by_css_selector(".credentials__input.mt15")

    print('Attempting to login')

    username.send_keys(f"{self.username}")
    password.send_keys(f"{self.password}")

    login_attempt = self.wd.find_element_by_xpath("//*[@type='submit']")
    login_attempt.submit()

    print(f"current url : {self.wd.current_url}")

    # test if login is successful
    # self.wd.find_elements_by_xpath("//*[contains(text(), 'My Account')]")
    
    
  def navigate_and_scrap(self):
    
    try:
    
      self.login()

      t.sleep(2)

      # print all elements that have an HTML href link and look for anything with betin league
      elems = self.wd.find_elements_by_xpath("//a[@href]")
      for elem in elems:
          print(elem.get_attribute("id"))

      # sleep to wait for page load
      t.sleep(2)
      # navigate to the virtual games page
      self.wd.execute_script("arguments[0].click()",self.wd.find_element_by_css_selector("a[id='pushmenu_quicklinks_link_1811_betin_league']"))
      t.sleep(2)
      # navigate to the premier league
      self.wd.find_element_by_xpath("//div[@onclick=\"openSelection('premier')\"]").click()

      betin_folder = '/home/ashton/betin_league'

      self._create_dir(betin_folder)

      league_count_old = 0
      league_week_old = 0


      while True:
        # wait for the next league to load by waiting for a specific element (league_number)
        WebDriverWait(driver=self.wd, timeout=40, poll_frequency=20).until(EC.visibility_of_element_located((By.XPATH, './/span[@id = "idleague"]')))

        try:

          league_count = int(self.wd.find_element_by_xpath('.//span[@id = "idleague"]').text)

        except:

          t.sleep(10)

          league_count = int(self.wd.find_element_by_xpath('.//span[@id = "idleague"]').text)

        premier_folder =  os.path.join(betin_folder, 'premier_league_{}'.format(league_count))
        odds_folder = os.path.join(premier_folder, 'odds')
        correct_score_folder = os.path.join(premier_folder, 'correct_score')
        results_folder = os.path.join(premier_folder, 'results')

        self._create_dir(premier_folder)
        self._create_dir(odds_folder)
        self._create_dir(correct_score_folder)
        self._create_dir(results_folder)

        print('League : {}'.format(league_count))

        print('Waiting for current week')

        WebDriverWait(driver=self.wd, timeout=100, poll_frequency=20).until(EC.visibility_of_element_located((By.XPATH, './/span[@id = "leagueWeekNumber"]')))


        league_week = int(self.wd.find_element_by_xpath('//span[@id = "leagueWeekNumber"]').text)

        if league_week_old == league_week:

            if league_week != 38:

              WebDriverWait(driver=self.wd, timeout=130, poll_frequency=30).until(EC.text_to_be_present_in_element((By.XPATH, './/span[@id = "leagueWeekNumber"]'), "{}".format(league_week+1)))

              league_week = int(self.wd.find_element_by_xpath('//span[@id = "leagueWeekNumber"]').text)

            else:

              WebDriverWait(driver=self.wd, timeout=130, poll_frequency=5).until(EC.text_to_be_present_in_element((By.XPATH, './/span[@id = "leagueWeekNumber"]'), "{}".format(1)))

              league_count+=1

              league_week = int(self.wd.find_element_by_xpath('//span[@id = "leagueWeekNumber"]').text)




        t.sleep(2)

        print('done waiting : \nWeek : {}'.format(league_week))

        final_df_temp = self.scrap_betin_odds(league_week = league_week, league_count = league_count)

        t.sleep(2)
        self.wd.execute_script("arguments[0].click()",self.wd.find_element_by_css_selector("a[id='a_bet_results']"))
        t.sleep(2)
        self.scrap_betin_results(league_count, league_week)
        self.wd.execute_script("arguments[0].click()",self.wd.find_element_by_css_selector("a[id='yellow_color_letter']"))
        #       wd.get(premier_url)
        t.sleep(2)

        league_week_old = league_week


    # send a slack message incase of failure
    except Exception as e:

      print(f'Exception : {e}')

      # slack_token = os.environ["SLACK_API_TOKEN"]
      slack_token = 'xoxb-589741945461-590957910966-AydeEpC4EXabA6dpp7yqzRQD'
      sc = SlackClient(slack_token)

      sc.api_call(
                "chat.postMessage",
                channel="modelling",
                text="Scrap failed : {}".format(e)
            )

      self.wd.quit()
    
   
    
    return
  
  # print all elements that have an HTML href link and look for anything with betin league
  def print_all_hrefs(self):
    
    elems = self.wd.find_elements_by_xpath("//a[@href]")
    for elem in elems:
        print(elem.get_attribute("id"))

  # scrap the odds, clean them and store them in a table
  def scrap_betin_odds(self, league_week, league_count):
    
    
    soup = BeautifulSoup(self.wd.page_source, 'lxml')

    tables = soup.select('table')

    comb_df = pd.DataFrame()

    start = 1
    stop = 12

    final_df = pd.DataFrame()
    final_correct_score_df = pd.DataFrame()

    for i in range(21):

      comb_df = pd.DataFrame()
      correct_score_df = pd.DataFrame()



      for table in tables[start:stop]:

        tab_data = table
        for items in tab_data.select('tr'):
            item = [elem.text for elem in items.select('th,td')]

            if len(item) == 1:
              teams = item[0].strip('\n')[-10:]

            # correct score row has 5 items
            elif len(item) == 5:
              cell1 = item[0]
              cell2 = item[1]
              cell3 = item[2]
              cell4 = item[3]
              cell5 = item[4]

              df = pd.DataFrame([cell1, cell2, cell3, cell4, cell5], index=['cell1', 'cell2', 'cell3', 'cell4', 'cell5'], columns=[teams]).T

              correct_score_df = pd.concat([correct_score_df, df])

            elif item[0] == '':
    #           index = ['time'] + item[1:]
              index = item[1:]

            # deal with the last rows which we don't require
            elif 'G/C' in item[0] or 'Bet type' in item[0] or '\n\n' in item[0] or 'CANCEL' in item[0] or 'Language' in item[0]:

                pass

            else:

              part1 = item[0].split('\n')
              time = part1[1]
              teams = part1[2]

              if len(index) == 3:

                cell3 = item[3]
                cell1 = item[1]
                cell2 = item[2]

                df = pd.DataFrame([cell1, cell2, cell3], columns=[teams], index=index)

              else:

                cell1 = item[1]
                cell2 = item[2]

                df = pd.DataFrame([cell1, cell2], columns=[teams], index=index)

              comb_df = pd.concat([comb_df, df], axis=1)

      start = stop
      stop += 11

      final_df = pd.concat([final_df, comb_df])
      final_correct_score_df = pd.concat([final_correct_score_df, correct_score_df])


    if (final_df.loc['1/X'].iloc[0].isnull() == False).sum() == 1:

      column_drop = final_df.columns[final_df.loc['1/X'].iloc[0].isnull() == False].values[0]  

    else:

      column_drop = final_df.columns[final_df.loc['1/X'].iloc[1].isnull() == False].values[0]  

    column_drop_df = final_df[column_drop]
    final_df_temp = final_df.drop(column_drop, axis=1)
    final_df_temp = pd.concat([column_drop_df.dropna(), final_df_temp.dropna()], axis=1, sort=False)

    week_teams = [x.replace(' ', '') for x in final_df_temp.columns.values]

    week_teams_df = pd.DataFrame(columns=self.weekly_cols[1:])

    week_teams_df['teams_id'] = teams_df.reset_index().set_index('teams_name').loc[week_teams, 'teams_id'].values

    week_teams_df['week_position'] = np.arange(1, week_teams_df.shape[0] + 1)
    week_teams_df['league_no'] = league_count
    week_teams_df['league_week'] = league_week

    
    week_teams_df.to_sql(con=engine, name='weekly', if_exists='append', index=False, method='multi')


    odds_path = os.path.join(odds_folder, 'odds_league_{}_week_{}.xlsx'.format(league_count, league_week))
    correct_score_path = os.path.join(correct_score_folder, 'correct_score_league_{}_week_{}.xlsx'.format(league_count, league_week))

    return final_df_temp

  # scrap the results, clean them and store them in a table
  def scrap_betin_results(self, league_count, league_week):
    # internal method to clean the results
    def structure_results(result_df, league_no):
      
      comb_df = pd.DataFrame()

      for j in np.arange(0, result_df.shape[1]):

        indices=[]
        results=[]
        
        week_no = int(result_df.columns[j][6:])



        for i in np.arange(0, result_df.shape[0], 3):

          team_1 = result_df.iat[i, j]
          result = result_df.iat[i+1, j]
          team_2 = result_df.iat[i+2, j]



          index = team_1 + '-' + team_2

          indices.append(index)
          results.append(result)

        df = pd.Series(results, index=indices, name='week_{}'.format(week_no))

        df = df.reset_index()

        df = df.rename(columns={'index':'week_{}_teams'.format(week_no)})

        comb_df = pd.concat([comb_df, df], axis=1, sort=False)

        
      if comb_df.empty:
        
        return 0
      
      else:
      
        test_df = comb_df.copy(deep=True)

        cols = test_df.columns

        for i in np.arange(0, len(cols), 2):

          col=cols[i]

          for j in test_df[col].index:


             test_df.iat[j, i] = cols[i+1] + '_' + test_df.iat[j, i]

          i+=2

        all_temp_df = pd.DataFrame()

        for i in np.arange(0, len(cols), 2):

          temp_df = pd.concat([test_df[cols[i]], test_df[cols[i+1]]], axis=1)

          temp_df = temp_df.T.reset_index(drop=True).T

          all_temp_df = pd.concat([all_temp_df, temp_df], sort=False)

        all_temp_df.columns = ['team_week', 'result']

        all_temp_df.set_index('team_week', inplace=True)

        all_temp_df.index = ['league_{}_'.format(league_no)] + all_temp_df.index 
      
        return all_temp_df
    
    soup = BeautifulSoup(self.wd.page_source, 'lxml')

    tables = soup.select('table')
    
    all_weeks_df = pd.DataFrame()


    for table in tables[265:]:

      week_df = pd.DataFrame()

      tab_data = table
      for items in tab_data.select('tr'):
          item = [elem.text for elem in items.select('th,td')]

          if len(item) == 1:

            week_no = item[0].split('\n')[1]


          elif len(item) == 3:
            team1 = item[0]
            result = item[1]
            team2 = item[2]

            df = pd.DataFrame([team1, result, team2], index=['team1', 'result', 'team2'], columns=[week_no])

            week_df = pd.concat([week_df, df])

      all_weeks_df = pd.concat([all_weeks_df, week_df], axis=1)

    if league_week==1 or league_week==2:
      
       all_weeks_df_path = os.path.join(results_folder, 'results_whole_league_{}_v{}.xlsx'.format(league_count-1, league_week))

    else:
      
      all_weeks_df_path = os.path.join(results_folder, 'results_league_{}_week_{}.xlsx'.format(league_count, league_week))

    # handle the fact that we only receive results of a match 2 virtual weeks later. Thus an edge case
    # appears on the first 2 weeks
    if league_week > 2:
      league_week_temp = league_week -2
      league_count_temp = league_count
    elif league_week == 1:
      league_week_temp = 38
      league_count_temp = league_count - 1
    elif league_week == 2:
      league_week_temp = 37
      league_count_temp = league_count - 1

    print(f'results :  \n {all_weeks_df.head()} \n')

    try:

      result_week_df = structure_results(all_weeks_df[f'WEEK  {league_week_temp}'].to_frame(), league_count)

      teams = [x.split('_')[-1] for x in result_week_df.index.values]

      result_week_df['teams_id'] = teams_df.reset_index().set_index('teams_name').loc[teams, 'teams_id'].values
      result_week_df['team1_result'] = [int(x.split('-')[0]) for x in result_week_df.result.values]
      result_week_df['team2_result'] = [int(x.split('-')[0]) for x in result_week_df.result.values]
      result_week_df['total_goals'] = result_week_df['team1_result'] + result_week_df['team2_result']
      result_week_df['league_no'] = league_count_temp # [x.split('_')[1] for x in result_week_df.index.values]
      result_week_df['league_week'] = league_week_temp # [x.split('_')[3] for x in result_week_df.index.values]

      result_week_df = result_week_df.reset_index(drop=True).drop('result', axis=1)      

      result_week_df.to_sql(con=engine, name='result', if_exists='append', index=False, method='multi')

    except Exception as e:

      print(f'Exception : {e}')
      
    

if __name__ == "__main__":

  Scrap = Scrap()

  Scrap.navigate_and_scrap()