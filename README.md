# time-series-prediction

### Description
This project is an attempt at developing a generic package for frequent time series modelling applications e.g. soil moisture prediction, stock prediction, footbal predicion. The first application supported is virtual football modelling. Virtual football is a simulation of football games where a match is 2 minute long. People then place bets on teams of their choice. The algorithm that runs the game is supposedly random but the house doesn't leave winnings to chance. Thus this application is an attempt at modelling the time series representing match results over time.

The project is structured as a python package. It should work for other time series applications as well. There are a number of components:

1. Scrapping component - uses selenium to programmatically access the betting website, navigate to the relevant pages and download the data into a database
2. Database - a postgresql database to store the data
3. Modeller - a bunch of scripts to preprocess the scrapped data, train and evaluate model
4. Dashboard - a visualization tool that enables data exploration in form of league tables and potential profitability of certain odds

The python package is named eureka254 and is split into the following scripts:
1. Scrap.py - Access the betting website, scrap the relevant data and store it into the database
2. DB.py - Contains database creation, population and helper functions.
3. ModelKwargs.py - Contains model initializations, hyperparameters and search spaces
4. Eureka Regression.py - is the main API a user interacts with. Contains the fit, predict and evaluation methods
5. TrainingHarness.py - contains the abstracted training functions
6. Evaluation.py - contains methods for profit evaluation and the more common metrics such as rmse etc

The `demo_run.ipynb` notebook contains a demo of running a training and evaluation run. (Tested on colab)
Alternatively, one can use `demo.py`

### Install (tested on colab)

#### Replace `master` with your preferred installable git branch
`pip install git+https://github.com/MichaelAshton/time-series-prediction.git@master`

You can now import the package:

`import eureka254`

The `demo_run.ipynb` notebook contains a demo of running a training and evaluation run. (Tested on colab)

### TO DO
- Test on windows/ubuntu
- Separate packages for GPU/Non-GPU
- Quick prototype
- Automated spin up of droplets every league or every X leagues
- Ensure the scrapping agent makes as little requests as possible
