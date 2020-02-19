import psycopg2
import sqlalchemy
import pandas as pd
import numpy as np
import os

db_username = os.environ["db_username"]
db_password = os.environ["db_password"]
db_ip = os.environ["db_ip"]
db_database = os.environ["db_database"]

class DB:
  
  def __init__(self, host=db_ip, database=db_database, user=db_username, password=db_password):
    
    self.host = host
    self.database = database
    self.user = user
    self.password = password
    
  # connect to db and return a cursor object
  def get_db_cursor(self):
    
    conn = psycopg2.connect(host=self.host,database=self.database, user=self.user, password=self.password)
    cur = conn.cursor()
    
    return conn, cur

  # alternate db connection via sqlalchemy
  def get_alchemy_engine(self):

    engine = sqlalchemy.create_engine(f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}/{self.database}")

    return engine
  
  # get the column header of the passed tables i.e. the fields
  def get_table_header(self, table: str):
    
    conn = None
    
    try:
      
      conn, cur = self.get_db_cursor()
    
      cur.execute("""Select * FROM "%s" LIMIT 0""", (table,))
      header = [desc[0] for desc in cur.description]
      
      print(f'{table} headers : {header}')
      
      return header
    
    except psycopg2.Error as e:
      print(f'Exception : {e}')

    finally:
    
      if conn is not None:
        conn.close()
  
  # helper function 
  def show_query(title: str, qry: str, cur: object):
    
    print('%s' % (title))
    cur.execute(qry)
    for row in cur.fetchall():
        print(row)
    print('')
    
    return cur
  
  # display the active database
  def show_current_db(self):
    
    conn = None
    
    try:
      
      conn, cur = self.get_db_cursor()
    
      cur = show_query('current database', 'SELECT current_database()', cur)
      
      cur.close()
    
    except psycopg2.Error as e:
      print(f'Exception : {e}')

    finally:
    
      if conn is not None:
        conn.close()
    
    
  # print all tables in the DB to screen  
  def show_all_tables(self):
    
    conn = None
    
    try:
      
      conn, cur = self.get_db_cursor()
    
      cur = show_query('all tables', """SELECT table_name FROM information_schema.tables
         WHERE table_schema = 'public'""", cur)
      
      cur.close()
      
    except psycopg2.Error as e:
      print(f'Exception : {e}')

    finally:
    
      if conn is not None:
        conn.close()
   
  
  # WARNING - THIS WILL PERMANENTLY DELETE THE PASSED TABLE WITH NO ARE YOU SURE MESSAGE
  # TO DO - Make this generic for any/all tables 
  # drop the passed tables 
  def drop_tables(self, tables: list):
    
    conn = None
    
    try:
      
      conn, cur = self.get_db_cursor()
      
      for table in tables:
    
        cur.execute('DROP TABLE "%s" CASCADE;', (table,))
      
      conn.commit()

      cur.close()
      
    except psycopg2.Error as e:
      print(f'Exception : {e}')

    finally:
    
      if conn is not None:
        conn.close()
  
  # TO DO - pass the table to create as a parameter and have another function this one to create all tables
  # create the tables. Should only be run for a new DB or when the tables have been dropped
  def create_tables(self):
    
    conn = None
    try:
      
      """ create tables in the PostgreSQL database"""
      commands = (
          """
          CREATE TABLE team (
              team_id SERIAL PRIMARY KEY,
              team_name VARCHAR(255) NOT NULL
          )
          """,
          """
          CREATE TABLE teams (
              teams_id SERIAL PRIMARY KEY,
              team1_id INTEGER,
              team2_id INTEGER,
              teams_name VARCHAR(255) NOT NULL,
              FOREIGN KEY (team1_id)
              REFERENCES team (team_id),
              FOREIGN KEY (team2_id)
              REFERENCES team (team_id)
              ON UPDATE CASCADE ON DELETE CASCADE
          )
          """,

          """
          CREATE TABLE odds (
              teams_id SERIAL PRIMARY KEY,
              home INTEGER NOT NULL,
              draw INTEGER NOT NULL,
              away INTEGER NOT NULL,
              gg_odds INTEGER NOT NULL, 
              nogoal_odds INTEGER NOT NULL,
              over15_odds INTEGER NOT NULL,
              under15_odds INTEGER NOT NULL,
              home_or_draw_odds INTEGER NOT NULL,
              home_or_away_odds INTEGER NOT NULL,
              draw_or_away_odds INTEGER NOT NULL,
              over25_odds INTEGER NOT NULL,
              under25_odds INTEGER NOT NULL,
              over35_odds INTEGER NOT NULL,
              under35_odds INTEGER NOT NULL,
              over45_odds INTEGER NOT NULL,
              under45_odds INTEGER NOT NULL,
              homeover05_odds INTEGER NOT NULL,
              homeunder05_odds INTEGER NOT NULL,
              homeover15_odds INTEGER NOT NULL,
              homeunder15_odds INTEGER NOT NULL,
              homeover25_odds INTEGER NOT NULL,
              homeunder25_odds INTEGER NOT NULL,
              homeover35_odds INTEGER NOT NULL,
              homeunder35_odds INTEGER NOT NULL,
              awayover05_odds INTEGER NOT NULL,
              awayunder05_odds INTEGER NOT NULL,
              awayover15_odds INTEGER NOT NULL,
              awayunder15_odds INTEGER NOT NULL,
              awayover25_odds INTEGER NOT NULL,
              awayunder25_odds INTEGER NOT NULL,
              awayover35_odds INTEGER NOT NULL,
              awayunder35_odds INTEGER NOT NULL,
              awayandover15_odds INTEGER NOT NULL,
              drawandover15_odds INTEGER NOT NULL,
              homeandover15_odds INTEGER NOT NULL,
              awayandover25_odds INTEGER NOT NULL,
              drawandover25_odds INTEGER NOT NULL,
              homeandover25_odds INTEGER NOT NULL,
              awayandunder15_odds INTEGER NOT NULL,
              drawandunder15_odds INTEGER NOT NULL,
              homeandunder15_odds INTEGER NOT NULL,
              awayandunder25_odds INTEGER NOT NULL,
              drawandunder25_odds INTEGER NOT NULL,
              homeandunder25_odds INTEGER NOT NULL,
              FOREIGN KEY (teams_id)
              REFERENCES teams (teams_id)
          )
          """,
          """
          CREATE TABLE weekly (
              weekly_id SERIAL PRIMARY KEY,
              teams_id INTEGER,
              league_no INTEGER,
              league_week INTEGER,
              week_position INTEGER,
              FOREIGN KEY (teams_id)
              REFERENCES teams (teams_id)
              )
          """,
          """
          CREATE TABLE result (
              result_id SERIAL PRIMARY KEY,
              teams_id INTEGER NOT NULL,
              team1_result INTEGER,
              team2_result INTEGER,
              total_goals INTEGER,
              league_no INTEGER NOT NULL,
              league_week INTEGER NOT NULL,
              week_position INTEGER NOT NULL,
              FOREIGN KEY (teams_id)
              REFERENCES teams (teams_id)
              ON UPDATE CASCADE ON DELETE CASCADE
          )
          """
      )

      
    
      conn, cur = self.get_db_cursor()

      # create table one by one
      for command in commands:
          cur.execute(command)
      
      cur.close()
      # commit the changes
      conn.commit()
      
      
    except psycopg2.Error as e:
      print(f'Exception : {e}')

    finally:
    
      if conn is not None:
        conn.close()
 
  # get database size in KB
  def get_db_size(self):
    
    conn = None
    
    try:
    
      conn, cur = self.get_db_cursor()

      q = """SELECT pg_database.datname as "%s", pg_size_pretty(pg_database_size(pg_database.datname)) AS size_in_mb FROM pg_database ORDER by size_in_mb DESC;"""
      # cur.execute(q, (dbname, ))
      cur.execute(q, (self.database,))
      row = cur.fetchone()
      print(str(row[1]))
      
      cur.close()
      
   
    except psycopg2.Error as e:
      print(f'Exception : {e}')

    finally:
    
      if conn is not None:
        conn.close()
    
  # TO DO - DO this with a generic function that works for any table
  # retrieve all records from results table 
  def get_all_results(self):
    
    conn = None
    
    try:
    
      conn, cur = self.get_db_cursor()
      
      df = pd.read_sql("""SELECT * from result""", con=conn)
      
      cur.close()
      
      return df.dropna(axis=1, how='all')
      
   
    except psycopg2.Error as e:
      print(f'Exception : {e}')

    finally:
    
      if conn is not None:
        conn.close()

  # TO DO - DO this with a generic function that works for any table
  # retrieve all records from teams table 
  def get_all_teams(self):
    
    conn = None
    
    try:
    
      conn, cur = self.get_db_cursor()
      
      df = pd.read_sql("""SELECT * from teams""", con=conn)
      
      cur.close()
      
      return df.dropna(axis=1, how='all')
      
   
    except psycopg2.Error as e:
      print(f'Exception : {e}')

    finally:
    
      if conn is not None:
        conn.close()

  # TO DO - DO this with a generic function that works for any table
  # retrieve all records from odds table 
  def get_all_odds(self):
    
    conn = None
    
    try:
    
      conn, cur = self.get_db_cursor()
      
      df = pd.read_sql("""SELECT * from odds""", con=conn, index_col='teams_names')
      
      cur.close()
      
      return df
      
   
    except psycopg2.Error as e:
      print(f'Exception : {e}')

    finally:
    
      if conn is not None:
        conn.close()

  # TO DO - make this more generic
  # upload results from xls into DB
  def upload_results(self, results_path):
  
    def calc_week_positions(df_week):
      
      df_week['week_position'] = np.nan

      df_week.set_index(['league_no', 'league_week'], inplace=True)

      for week in df_week.reset_index().league_week.unique():

        for league in df_week.reset_index().league_no.unique():

          try:

            df_week.loc[(league, week), 'week_position'] = np.arange(1, df_week.loc[(league, week)].shape[0]+1)

          except Exception as e:

            print(f'League : {league}, Week : {week} \n Exception : {e}')
        
      return df_week
    
    def upload_in_parts(df, engine):
      
      # TO DO - make this fucntion generic
      no_parts = 89
      part_size = 5660

      for start in np.arange(0, df.shape[0], part_size):
        
        stop = start + part_size
        print(f'start : {start}, stop : {stop}')

        df.iloc[start:stop].to_sql(con=engine, name='result', if_exists='append', index=False, method='multi')
        
      return
    
    
    engine = self.get_alchemy_engine()      
        
    teams_df = self.get_all_teams()
    
    df = pd.read_excel(results_path, index_col=0)
    
    teams = [x.split('_')[-1] for x in df.index.values]

    df['teams_id'] = teams_df.reset_index().set_index('teams_name').loc[teams, 'teams_id'].values
    df['team1_result'] = [int(x.split('-')[0]) for x in df.result.values]
    df['team2_result'] = [int(x.split('-')[0]) for x in df.result.values]
    df['total_goals'] = df['team1_result'] + df['team2_result']
    df['league_no'] = [x.split('_')[1] for x in df.index.values]
    df['league_week'] = [x.split('_')[3] for x in df.index.values]

    df = df.reset_index(drop=True).drop('result', axis=1)
    
    df = calc_week_positions(df)
    
    df.week_position = df.week_position.map(int)
    df.reset_index(inplace=True)
    
    if df.shape[0] > 1000:
      
      # TO DO - Not ready yet - Make this function generic
      upload_in_parts(df, engine)
      
    else:

      df.to_sql(con=engine, name='result', if_exists='append', index=False, method='multi')

    return 

  def create_golden_dataset(self, regenerate=False, golden_start=21886, golden_stop=21943):

    # get longest running leagues range with no missing league
    def longest_leagues_range(df):
      longest_running = []
      leagues = df.league_no.unique()
      longest_running_df = pd.DataFrame(columns=['count'])
      start = df.league_no.unique()[0]
      count = 0
      for idx, league in zip(np.arange(0, len(leagues)-1), leagues):
        
        if league+1 == leagues[idx+1]:
          longest_running.append(leagues)
          count+=1
        else:
          stop = league
          key = f'{start}-{stop}'
          # longest_running_dict[key] = count
          longest_running_df.at[key, 'count'] = count
          count = 0
          start = leagues[idx+1]

      return longest_running_df

    # get longest running leagues with no missing weeks
    def longest_weeks_range(df, leagues_df):
      longest_weeks_df = pd.DataFrame(columns=['count', 'percentage'])
      for x in longest_running_dict.keys():
        count = 0
        start, stop = x.split('-')
        start = int(start)
        stop = int(stop)
        expected_total = (stop - start) * 38

        for league in np.arange(start, stop):

          no_weeks = df.loc[df.league_no == league].league_week.unique().shape[0]
          count += no_weeks

        try:
          percentage = (count / expected_total) * 100
        except:
          percentage=0

        percentage = np.round(percentage, 2)
        

        longest_weeks_df.at[f'{start}-{stop}', 'count'] = leagues_df.at[f'{start}-{stop}', 'count']
        longest_weeks_df.at[f'{start}-{stop}', 'percentage'] = percentage


      return longest_weeks_df


    df = self.get_all_results()

    if regenerate:

      leagues_df = longest_leagues_range(df)
      weeks_df = longest_weeks_range(df, leagues_df)

      weeks_df['count'] = weeks_df['count'].map(int)
      weeks_df = weeks_df.loc[weeks_df.percentage == 100].copy(deep=True) 
      golden_start, golden_stop = weeks_df['count'].idxmax().split('-') 

      golden_start = int(golden_start)
      golden_stop = int(golden_stop)

    golden_df = df.reset_index().set_index('league_no').loc[golden_start:golden_stop].copy(deep=True)

    golden_df = golden_df.reset_index().set_index('index').copy(deep=True)

    return golden_df


# Example Usage    
if __name__ == "__main__":

  # initialize
  db = DB()

  # show database size in KB
  db.get_db_size()

  # fetch all results
  df = db.get_all_results()   

  # fetch golden dataset
  df = db.create_golden_dataset()  

  print(f'first 5 results : \n {df.head()} \n')

  # fetch all teams
  df = db.get_all_teams()   

  print(f'first 5 teams : \n {df.head()} \n')
    
    
    
    
    
    