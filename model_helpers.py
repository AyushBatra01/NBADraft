import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import plotly.graph_objects as go

pd.set_option('mode.chained_assignment', None)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


MAIN_COLOR = '#E4DBD7'
MY_COLOR = '#274ada'
GRAY = '#C0C0C0'

def cleanFigure(fig, size = 700, grid = False):
    """
    Make a plotly figure look nicer
    """
    fig.update_layout(plot_bgcolor = MAIN_COLOR, paper_bgcolor = MAIN_COLOR,
                      height = size, width = size,
                      title_x = 0.5, title_xanchor = 'center', title_font_size = 20,
                      xaxis_gridcolor = GRAY, yaxis_gridcolor = GRAY,
                      xaxis_showgrid = grid, yaxis_showgrid = grid,
                      xaxis_linecolor = 'black', xaxis_linewidth = 1,
                      yaxis_linecolor = 'black', yaxis_linewidth = 1)
    fig.update_yaxes(ticks = 'outside', tickcolor = MAIN_COLOR)
    fig.update_xaxes(ticks = 'outside', tickcolor = MAIN_COLOR)


def toDF(x, cols):
    """
    Converts a 2d array to a pandas dataframe
    
    Parameters
    ----------
    x : 2d numpy.array
        2D numpy array to be converted into a pandas dataframe
    cols : list of str
        column names for new pandas dataframe
        length should be equal to number of columns of numpy array 
        
    Returns
    -------
    pd.DataFrame
        numpy array converted into a pandas dataframe
    """
    df = pd.DataFrame(x)
    df.columns = cols
    return df



def assignPosition(pos):
    """
    Converts a NBA position into a one-hot encoded vector
    Formatting of vector is [PG, SG, SF, PF, C]
    If a player has two positions listed, this will assign 0.5 to both positions
    Helper function for model_helpers.getPosCols
    
    Parameters
    ----------
    pos : str
        position
        must be one of 'PG', 'SG', 'SF', 'PF', or 'C'
    
    Returns
    -------
    list 
        list of length 5 containing the one-hot encoded position vector
    """
    p_dct = {'PG' : 0, 'SG' : 1, 'SF' : 2, 'PF' : 3, 'C' : 4}
    pos_dummy = [0 for _ in range(5)]
    # put 0.5 for both positions if have two positions
    if "/" in pos:
        p1, p2 = pos.split("/")
        pos_dummy[p_dct[p1]] = 0.5
        pos_dummy[p_dct[p2]] = 0.5
    # put 1 for position if only have one position listed
    else:
        pos_dummy[p_dct[pos]] = 1
    return pos_dummy



def getPosCols(df):
    """
    Creates a dataframe with columns for one-hot position encodings
    Helper function for model_helpers.clean
    
    Parameters
    ----------
    df : pd.DataFrame
        dataframe that contains a column called 'Position' that should be converted into a one-hot vector format
    
    Returns
    -------
    pd.DataFrame
        dataframe of length n x 5 that gives one-hot encoded position vector for each observation
    """
    # create column for each position variable
    pos_cols = ['PG', 'SG', 'SF', 'PF', 'C']
    pos_series = [df['Position'].apply(assignPosition).apply(lambda x : x[i]) for i in range(5)]
    # combine them together and give col names
    pos_df = pd.concat(pos_series, axis = 1)
    pos_df.columns = pos_cols
    return pos_df




def clean(df, min_minutes = 200):
    """
    Do some initial data cleaning on draft data
    This creates columns for 2pt shooting, removes columns for overall shooting (since this can be captured using 2pt and 3pt shooting),
        renames the PF column, drops irrelevant columns, filters out players without enough playing time, and adds the one-hot encoded
        position vectors
    
    Parameters
    ----------
    df : pd.DataFrame
        dataframe to be cleaned
    min_minutes : int, default = 200
        Minimum number of total minutes played to be included in the sample
    
    Returns
    -------
    pd.DataFrame
        cleaned dataframe implementing all the changes stated above
    """
    # use 2pt field goals instead of total field goals (to separate completely from 3's)
    df['2PM'] = df['FGM'] - df['3PM']
    df['2PA'] = df['FGA'] - df['3PA']
    df['2P%'] = df['2PM'] / df['2PA']
    # and drop original FGA cols
    df = df.drop(columns = ['FGM', 'FGA', 'FG%'])
    
    # rename col to avoid conflict with PF positional dummy column
    df = df.rename(columns = {'PF' : 'Fouls'})
    # remove irrelevant information
    df = df.drop(columns = ['Birthdate', 'Nation'])
    # remove players without any data
    df = df[df['G'].notnull()]
    
    # remove players without enough playing time
    df['Total_MP'] = df['G'] * df['MP']
    df = df[df['Total_MP'] > min_minutes]
    df = df.drop(columns = ['Total_MP'])
    
    # add dummy variables for position
    pos_df = getPosCols(df)
    df = pd.concat([df, pos_df], axis = 1)
    df = df.reset_index(drop = True)
    return df




def add_general_features(df):
    """
    Implements initial feature engineering for a draft dataframe
    Features include: BMI, indicators for not having advanced data or box plus minus (BPM) data, 
        an adjustment from 3-point percentage to 3-point proficiency, the create of log terms for heavily
        skewed predictors, the creation of a log term for draft pick, and handling of incorrect PER data
    
    Parameters
    ----------
    df : pd.DataFrame
        draft dataframe to implement feature engineering for
    
    Returns
    -------
    pd.DataFrame
        dataframe with implemented features (as specified above)
    """
    # New Features
    df['BMI'] = 703 * df['Weight'] / (df['Height']**2)
    df['WS/Ht Ratio'] = df['Wingspan'] / df['Height']
    df['Wingspan_diff'] = df['Wingspan'] - df['Height']
    df['TOV_Rate'] = df['TO'] / (df['TO'] + df['2PA'] + df['3PA'] + 0.44 * df['FTA'])

    # Add identifiers telling if missing advanced stats or BPM stats
    df['na_advanced'] = (df['USG%'].isnull()).astype(int)
    df['na_bpm'] = (df['BPM'].isnull()).astype(int)
    
    # Adjust 3P% to account for low volume shooters (using 3-point proficiency)
    df['3P%'] = df['3P%'] * (2 / (1 + np.exp(-df['3PA'])) - 1)
    
    # add some log terms for skewed variables
    log_cols = ['BLK', 'AST', 'AST/USG', 'AST/TO']
    for col in log_cols:
        df['log_' + col] = np.log(df[col] + 1)
    df['log_MP'] = np.log(df['MP'])
    # add log term for pick since it makes more sense on a log scale
    df['log_pick'] = np.log(df['Pick'])
    
    # set PER to NA if PER = 0
    df['PER'] = np.where(df['PER'] == 0, np.NaN, df['PER'])
    
    return df




def features_after_imp(df):
    """
    Implements feature engineering to keep features consistent after imputing
    Relevant for features that are calculated based on other features that may have missing values
    
    Parameters
    ----------
    df : pd.DataFrame
        draft dataframe to implement feature engineering for
    
    Returns
    -------
    pd.DataFrame
        dataframe with implemented features (as specified above)
    """
    df['WS/Ht Ratio'] = df['Wingspan'] / df['Height']
    df['Wingspan_diff'] = df['Wingspan'] - df['Height']
    df['AST/USG'] = np.where(df['AST/USG'] < 0, 0, df['AST/USG'])
    df['log_AST/USG'] = np.log(df['log_AST/USG'] + 1)
    return df




def add_predictions(df, X, model, classifier = True):
    """
    Create a dataframe containing model predictions along with identification data
    
    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing identification information, including 'Name', 'Year', 'Draft Team', 'Team', 'Position', and 'Pick'
    X : pd.DataFrame or np.array
        predictor matrix to be used for model prediction
    model : sklearn model
        trained sklearn model that will be used to make predictions 
    classifier : bool, default = True
        specifies whether to provide probabilities (for a classifier) or numbers (for a regressor) as prediction
        
    Returns
    -------
    pd.DataFrame
        dataframe containing identification info along with predictions for each observation
    """
    # store identification info
    id_cols = ['Name', 'Year', 'Draft Team', 'Team', 'Position', 'Pick']
    # generate model predictions
    if classifier:
        yhat = pd.DataFrame(model.predict_proba(X))
    else:
        yhat = pd.DataFrame(model.predict(X))
    # get id info, combine with model predictions
    df2 = df.copy().reset_index(drop = True)
    df2 = df2[id_cols]
    preds = pd.concat([df2, yhat], axis = 1)
    return preds



def calibration_plot(y_true, y_pred, n_bins = 10, by_pct = False, logit_scale = False):
    """
    Assess model calibration for a classification model
    Outputs probability calibration scatterplot, probability distribution histogram, and table 
        comparing mean probabilities to mean actual outcomes for each bin
    
    Parameters
    ----------
    y_true : list or np.array
        array of true labels (1 or 0 for each element)
    y_pred : list of np.array
        array of probabilities for positive class 
    n_bins : int, default = 10
        number of bins to use in probability calibration scatterplot and comparison table
    by_pct : bool, default = False
        specifies whether to create evenly spaced bins (by_pct = False) or create bins with equal 
        numbers of observations (by_pct = True)
    logit_scale : bool, default = False
        specifies whether to plot the probability calibration on a logit scale (for x-axis) instead 
        of a traditional scale
        
    Returns
    -------
    pd.DataFrame
        Dataframe comparing predicted probabilities to actual outcomes
        Contains 95% CI's for bounds of predicted probabilities
        Also contains Z-score using the standard error of the mean probability
        Tells whether the actual mean outcome is within the predicted 95% CI of outcomes
        
    Other Plots (not returned, only displayed)
    ------------------------------------------
    probability calibration plot
        Plots actual mean outcomes vs predicted mean probabilities for each bin
        Dotted black line indicates perfectly calibrated model
        Point size indicates sample size for bin (relevant when by_pct = False)
    probability distribution histogram
        Displays distribution of predicted probabilities
        Uses the number of bins provided by n_bins
    """
    # create bins 
    if by_pct:
        p = pd.Series(y_pred).rank(pct = True)
        buckets = pd.cut(p, bins = n_bins)
    else:
        buckets = pd.cut(y_pred, bins = n_bins)
    # initialize comparison data
    cal_df = pd.DataFrame()
    cal_df['bucket'] = buckets
    cal_df['prob'] = y_pred
    cal_df['lab'] = y_true
    # get counts for each bin
    n = cal_df.groupby(['bucket'], observed = True)['prob'].count()
    # get mean actual and predicted outcomes
    cal_df = cal_df.groupby(['bucket'], observed = True)[['prob', 'lab']].mean()
    # create perfect calibration line
    line = np.arange(cal_df['prob'].min(), cal_df['prob'].max(), 0.001)
    # add sample sizes to bins data
    cal_df['n'] = n
    
    ### PROBABILITY CALIBRATION SCATTER ###
    # plot perfect calibration line
    plt.plot(line, line, color = 'black', linestyle = 'dotted')
    # plot actual vs predicted outcomes (line and scatter)
    plt.plot(cal_df['prob'], cal_df['lab'])
    plt.scatter(cal_df['prob'], cal_df['lab'], s = n)
    # specify scale and create axis labels
    if logit_scale:
        plt.xscale('logit') 
        plt.xlabel("Mean predicted probability (logit scale)")
    else:
        plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positive labels")
    plt.title("Calibration Plot")
    plt.show()
    
    ### PROBABILITY DISTRIBUTION ###
    # create histogram for predicted probability distribution
    plt.hist(y_pred, bins = n_bins)
    plt.xlabel("Predicted Probability")
    plt.title("Probability Distribution")
    plt.show()
    
    # add standard error, 95% CI bounds, and in range indicator to comparison data
    cal_df['se'] = np.sqrt(cal_df['prob'] * (1 - cal_df['prob']) / cal_df['n'])
    cal_df['lower'] = cal_df['prob'] - 2 * cal_df['se']
    cal_df['upper'] = cal_df['prob'] + 2 * cal_df['se']
    cal_df['Z'] = (cal_df['lab'] - cal_df['prob']) / cal_df['se']
    cal_df['in_range'] = np.where((cal_df['lab'] > cal_df['lower']) & (cal_df['lab'] < cal_df['upper']),
                                  True, False)
    
    # make columns look nicer
    cal_df = cal_df.rename(columns = {'prob' : 'Mean Predicted Probability',
                                      'lab' : 'Percent Positive Labels',
                                      'n' : 'N',
                                      'se' : 'SE',
                                      'lower' : 'Lower',
                                      'upper' : 'Upper',
                                      'in_range' : 'Inside Interval?'})
    return cal_df
    
    
    
def calibration_plotly(y_true, y_pred, n_bins = 10, by_pct = False, logit_scale = False):
    """
    See calibration_plot function
    only difference is this returns plotly figures instead of showing matplotlib plots
    Also does not return the calibration dataframe; only returns graphs
    """
    # create bins 
    if by_pct:
        p = pd.Series(y_pred).rank(pct = True)
        buckets = pd.cut(p, bins = n_bins)
    else:
        buckets = pd.cut(y_pred, bins = n_bins)
    # initialize comparison data
    cal_df = pd.DataFrame()
    cal_df['bucket'] = buckets
    cal_df['prob'] = y_pred
    cal_df['lab'] = y_true
    # get counts for each bin
    n = cal_df.groupby(['bucket'], observed = True)['prob'].count()
    # get mean actual and predicted outcomes
    cal_df = cal_df.groupby(['bucket'], observed = True)[['prob', 'lab']].mean()
    # create perfect calibration line
    line = np.arange(cal_df['prob'].min(), cal_df['prob'].max(), 0.001)
    # add sample sizes to bins data
    cal_df['n'] = n
    
    ### PROBABILITY CALIBRATION SCATTER ###
    fig1 = go.Figure()
    # plot perfect calibration line
    fig1.add_trace(go.Scatter(
        x = line,
        y = line,
        mode = "lines",
        line = dict(color = 'black', dash = 'dot'),
        name = "Perfect Calibration Line",
        showlegend = False
    ))
    # plot actual vs predicted outcomes (line and scatter)
    fig1.add_trace(go.Scatter(
        x = cal_df['prob'],
        y = cal_df['lab'],
        mode = "lines+markers",
        marker = dict(color = 'blue', size = 5, line = dict(color = 'black', width = 0.5)),
        line = dict(color = 'blue'),
        showlegend = False
    ))
    # specify scale and create axis labels
    if logit_scale:
        fig1.update_xaxes(type = 'log', title = "Mean predicted probability (log scale)")
        fig1.update_layout(xaxis_tickmode = 'array',
                           xaxis_tickvals = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5])
    else:
        fig1.update_xaxes(title = "Mean predicted probability")
    fig1.update_layout(yaxis_title = "Fraction of positive labels",
                       title = "Calibration Plot",
                       title_x = 0.5, title_xanchor = 'center',
                       yaxis_range = [0, max(cal_df['lab']) + 0.01],
                       yaxis_tickformat = ".0%",
                       xaxis_tickformat = ".0%",
                       height = 500, width = 500)
    
    ### PROBABILITY DISTRIBUTION ###
    # create histogram for predicted probability distribution
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x = y_pred,
        nbinsx = n_bins
    ))
    fig2.update_layout(xaxis_title = "Predicted Probability",
                       title = "Probability Distribution",
                       title_x = 0.5, title_xanchor = 'center',
                       height = 500, width = 500)
    
    return fig1, fig2