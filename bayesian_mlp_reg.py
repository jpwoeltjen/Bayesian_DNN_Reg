
import numpy as np
import pandas as pd
from math import sqrt
from numpy import concatenate
from pandas import concat
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Dropout
from keras.callbacks import EarlyStopping
from sklearn.pipeline import make_pipeline
from keras.regularizers import L1L2
import matplotlib.pyplot as plt
from keras import optimizers
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras import layers 
from keras import backend as K
from keras.callbacks import TensorBoard
import os 
# np.random.seed(777)


def multivariate_ts_to_supervised_extra_lag(data, ohlc, n_in=1, n_out=1, return_threshold=0):
    """
    Convert series to supervised learning problem. The holding period is assumed to be open to open.
    Alteratively, you may want to hold from open to close. 
    """
    
    df = pd.DataFrame(data)
    df.columns = range(df.shape[1])

    #get open to high/low return
    highs = ohlc.iloc[data.index,1]
    lows = ohlc.iloc[data.index,2]
    opens = ohlc.iloc[data.index,0]

    return_min = lows.shift(-1)/opens.shift(-1)   -1 
    return_max = highs.shift(-1)/opens.shift(-1)   -1 
    

    n_vars = df.shape[1]
    cols, names = list(), list()
    # exlude most recent data which, for trading, is not yet available
    for i in range(n_in, 1, -1):
       cols.append(df.shift(i))
       names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    
    cols.append(df)
    names += [('var%d' % (j+1)) for j in range(n_vars)]
    cols.append(df.iloc[:,0].shift(-(2)))
    names.append('periodic_return')
    agg = concat(cols, axis=1)
    agg.columns = names

    agg['return_min'] = return_min
    agg['return_max'] = return_max    
    
    agg['return'] = highs.shift(-n_out).rolling(n_out).max()/opens.shift(-1) -1 
    agg.loc[((-(lows.shift(-n_out).rolling(n_out).min()/opens.shift(-1)-1)) > agg['return']),'return'] = \
    (lows.shift(-n_out).rolling(n_out).min()/opens.shift(-1)-1)
    
        
    agg = agg[n_in:] # drop na at the head
    agg = agg.fillna(0)  #fill the next periods prediction with 0 so no error in scaling is raised
    return agg    


def get_returns(data, columns=[1,2,3,4], dropna=True):
    """
    Create new DataFrame with pct_change() applied to specified columns
    and other columns left unchanged.
    """

    cols =  list(data.columns)
    pct_change_cols = []
    data_returns= data.copy(deep=True)


    for i in columns:
        data_returns[i]=data_returns[i].pct_change()
    
    if dropna:
        data_returns.dropna(inplace=True)
        data_returns.replace([np.inf, -np.inf], 0,inplace=True)
    
       
    return data_returns  




def fit_model(model, train_X, train_y, val_X, val_y, batch, n_epochs, n_neurons,\
 n_layers, lags, n_features, breg, kreg, rreg, lr, lrd, do):
    
    """
    Define model and fit it
    """
    n_obs = n_features*(lags)

    tb = TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)

    neuron_decay_factor_per_layer = 1 #0.75

    # design network
    if model == None:
        model = Sequential()
        model.add(BatchNormalization( input_shape=(n_obs,)))
       



        for i in range(n_layers):   
            model.add(Dense(n_neurons,bias_regularizer=breg, kernel_regularizer=kreg))
            model.add(BatchNormalization()) 
            model.add(layers.Activation('relu'))
            model.add(Dropout(do))   
            n_neurons = int(n_neurons*neuron_decay_factor_per_layer)
        
        
        model.add(Dense(1))
        model.add(BatchNormalization()) 
        model.add(layers.Activation('linear'))
        
    # adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=lrd)
    nadam = optimizers.Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.compile(loss='mse', optimizer=nadam)#mean_squared_error

    history = model.fit(train_X, train_y, epochs=n_epochs,
                  validation_data=(val_X, val_y),
                  batch_size=batch, 
                  verbose=2, shuffle=True, 
                 callbacks= [tb]#[EarlyStopping(monitor='val_loss', patience=100, verbose=2, mode='auto')]
                    )
    return model, history    


def train(model, dataset, train_pct, lags, n_epochs, batch,\
 n_neurons, layers, n_features, breg, kreg, rreg, lr, lrd, do, p_out,rt):

    """
    Get dataset ready and train model
    """
    
    n_obs = (n_features)*(lags)

    dataset_returns = pd.DataFrame(dataset)
    
    dataset_returns = get_returns(dataset_returns)#, columns=[1,2,3,4,5,6])
    
    values = dataset_returns#.values.astype('float32')
    values_encoded = values#encode(values)
    
    reframed = multivariate_ts_to_supervised_extra_lag(values_encoded,dataset.iloc[:,[0,1,2,3]], lags, p_out,rt)
    print(reframed.head(10))


    reframed_values=reframed.values
    train, test = reframed_values[:int(train_pct*len(reframed)), :] , reframed_values[int(train_pct*len(reframed)):, :]
    # split into input and outputs
    train_X, train_y = train[:, :n_obs], train[:, -1]
    test_X, test_y = test[:, :n_obs], test[:, -1]
    periodic_return = test[:,-4]
    low_return = test[:,-3]
    high_return = test[:,-2]

    #standardize
    train_X_mean = np.mean(train_X, 0)
    train_X_std = np.std(train_X, 0)

    train_X = (train_X - np.full(train_X.shape, train_X_mean)) / \
            np.full(train_X.shape, train_X_std)


    train_y_mean = np.mean(train_y)
    train_y_std = np.std(train_y)

    #save mean and std to disk. This is needed at test time 
    scalers = pd.DataFrame()
    scalers["train_X_mean"] = train_X_mean
    scalers["train_X_std"] = train_X_std
    scalers["train_y_mean"] = train_y_mean
    scalers["train_y_std"] = train_y_std
    scalers.to_csv('scalers.csv', header = True, index=True, encoding='utf-8')


    train_y_normalized = (train_y - train_y_mean) / train_y_std
    train_y_normalized = np.array(train_y_normalized, ndmin = 2).T

    test_X = (test_X - np.full(test_X.shape, train_X_mean)) / \
            np.full(test_X.shape, train_X_std)

    test_y_normalized = (test_y - train_y_mean) / train_y_std
    test_y_normalized = np.array(test_y_normalized, ndmin = 2).T
    
    # fit the model
    fitted_model, history = fit_model(model, train_X, train_y_normalized, test_X, test_y_normalized,
     batch, n_epochs, n_neurons, layers, lags, n_features, breg, kreg, rreg, lr, lrd, do)

    return test_X, test_y, periodic_return, low_return, high_return, fitted_model#, train_y_mean, train_y_std

def out_of_sample_test(test_X, test_y, periodic_return, low_return, high_return, model):
    """
    Get OOS predictions, set mc_dropout = True to perform MC dropout at test time
    """

    #load scaler
    scaler = pd.read_csv('scalers.csv', header=0, index_col=0)
    train_y_mean = scaler["train_y_mean"]
    train_y_std = scaler["train_y_std"]

    mc_dropout = True

    if mc_dropout == True:
        ### MC Dropout
        T = 100
        # We want to use Dropout at test time, not just at training time as usual.
        # To do this we tell Keras to predict with learning_phase set to true.  
        predict_stochastic = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

        Y_hat = np.array([predict_stochastic([test_X, 1]) for _ in range(T)])

        # if y was standardized
        Y_hat = Y_hat * train_y_std[0] + train_y_mean[0]

        yhat_std = np.std(Y_hat, 0)
        yhat = np.mean(Y_hat, 0)
        yhat = yhat.squeeze()
        yhat_std = yhat_std.squeeze()
    else:
        #ordinary prediction
        yhat = model.predict(test_X, batch_size=512)
        yhat = yhat.squeeze()
        yhat_std = np.zeros(yhat.shape)



    output_df=pd.DataFrame()

    output_df['prediction'] = pd.Series(yhat)

    output_df['prediction_std'] = pd.Series(yhat_std)

    output_df['actual'] = pd.Series(test_y)

    output_df['periodic_return'] = pd.Series(periodic_return)
    output_df['low_return'] = pd.Series(low_return)
    output_df['high_return'] = pd.Series(high_return)


    # return dataset_returns with OOS predictions
    return output_df
def equity_curve(dataset, m, periods_in_year, plot, threshold, profit_taking_threshold, bayesian_threshold, p_out):
    """
    Compute equity curves and trade statistics
    """
    transaction_cost = 2/100000
    dataset.dropna(inplace=True)
    
    if profit_taking_threshold >= 0:
        dynamic_profit_taking = False
    else:
        dynamic_profit_taking = True

    print(f">profit_taking_threshold: {profit_taking_threshold}" )
    print(f">dynamic_profit_taking: {dynamic_profit_taking}" )
    
    # loop over Bayesian confidence
    for p in bayesian_threshold:
        print('>FOR MODEL: %s' %m)
        print("return > %s  x std: "%p)

        # loop over softmax threshold. The softmax output can provide information about the strength of the
        # trading signal. However, it can be very high even if the cofidence of the network
        # is low this is the reason we need a Bayesian uncertainty.
        for i in threshold:
            # if prediction confidence is less than p*std ignore prediction as it is deemed not stat significant
            dataset.loc[(dataset['prediction'].abs() < p*dataset['prediction_std']), 'prediction'] = 0 
            # dataset.loc[(dataset['up_prediction'] < p*dataset['up_prediction_std']),'up_prediction'] = 0 

            #get signal according to softmax output
            dataset['signal_%.2f_sigma' %i] = 0

            dataset.loc[(dataset['prediction'] > i/100000) , 'signal_%.2f_sigma' %i] = 1

            dataset.loc[(dataset['prediction'] <- i/100000) ,'signal_%.2f_sigma' %i] = -1


            dataset['signal_%.2f_sigma' %i]=dataset['signal_%.2f_sigma' %i].fillna(0)
            dataset.reset_index(drop=True, inplace=True)

            long_value = np.zeros(shape=dataset.shape[0])
            short_value = np.zeros(shape=dataset.shape[0])
            profit_taking_long = np.zeros(shape=dataset.shape[0])
            profit_taking_short = np.zeros(shape=dataset.shape[0])
            trade_result = np.zeros(shape=dataset.shape[0])
            time_in_long = np.zeros(shape=dataset.shape[0])
            time_in_short = np.zeros(shape=dataset.shape[0])
            trade = np.zeros(shape=dataset.shape[0])
            long_trade_return = np.zeros(shape=dataset.shape[0])
            short_trade_return = np.zeros(shape=dataset.shape[0])

            # the next snippet implements the trading logic
            for j in dataset.index:
                if  short_value[j]==0:
                    if (dataset['signal_%.2f_sigma' %i][j] == 1) and (long_value[j]==0):
                        long_value[j] = 1
                        time_in_long[j] = 1
                        trade[j] = trade[j]+1
                        if dynamic_profit_taking:
                            profit_taking_threshold = dataset['prediction'][j]
                        
                    if long_value[j]!=0:
                        if long_value[j]*(1+dataset['high_return'][j])>(1+profit_taking_threshold):
                            profit_taking_long[j] = 1 
                            trade[j] = trade[j]+1
                            trade_result[j] = (1+profit_taking_threshold-long_value[j])/long_value[j]
                            long_trade_return[j] = profit_taking_threshold
                            if j != dataset.index[-1]: long_value[j+1] = 0
                            
                        else:
                            trade_result[j] = dataset['periodic_return'][j]
                            if j != dataset.index[-1]:
                                
                                if time_in_long[j] == p_out:
                                    long_value[j+1] = 0 
                                    time_in_long[j+1] = 0
                                    trade[j+1] = trade[j+1]+1
                                    long_trade_return[j] = long_value[j]*(1+dataset['periodic_return'][j]) -1
                                else:
                                    long_value[j+1] = long_value[j]*(1+dataset['periodic_return'][j])
                                    time_in_long[j+1] = time_in_long[j]+1
                                
                if long_value[j]==0:
                    if (dataset['signal_%.2f_sigma' %i][j] == -1) and (short_value[j]==0):
                        short_value[j] = 1
                        time_in_short[j] = 1
                        trade[j] = trade[j]+1
                        if dynamic_profit_taking:
                            profit_taking_threshold = -dataset['prediction'][j]
                        
                    if short_value[j]!=0:
                        if short_value[j]*(1-dataset['low_return'][j])>(1+profit_taking_threshold):
                            profit_taking_short[j] = 1 
                            trade[j] = trade[j]+1
                            trade_result[j] = (1+profit_taking_threshold-short_value[j])/short_value[j]
                            short_trade_return[j] = profit_taking_threshold
                            if j != dataset.index[-1]: short_value[j+1] = 0
                            
                        else:
                            trade_result[j] = -dataset['periodic_return'][j]
                            if j != dataset.index[-1]:
                                
                                if time_in_short[j] == p_out:
                                    short_value[j+1] = 0 
                                    time_in_short[j+1] = 0
                                    trade[j+1] = trade[j+1]+1
                                    short_trade_return[j] = short_value[j]*(1-dataset['periodic_return'][j]) -1
                                else:
                                    short_value[j+1] = short_value[j]*(1-dataset['periodic_return'][j])
                                    time_in_short[j+1] = time_in_short[j]+1


            dataset['long_value_%.2f_sigma' %i] = long_value
            dataset['short_value_%.2f_sigma' %i] = short_value
            dataset['profit_taking_long_%.2f_sigma' %i] =profit_taking_long
            dataset['profit_taking_short_%.2f_sigma' %i] =profit_taking_short
            dataset['trade_result_%.2f_sigma' %i] =trade_result
            dataset['time_in_long_%.2f_sigma' %i] =time_in_long
            dataset['time_in_short_%.2f_sigma' %i] =time_in_short
            dataset['trade_%.2f_sigma' %i]= trade 
            dataset['long_trade_return_%.2f_sigma' %i]=long_trade_return
            dataset['short_trade_return_%.2f_sigma' %i]=short_trade_return
            dataset['total_trade_return_%.2f_sigma' %i]=dataset['long_trade_return_%.2f_sigma' %i]+ \
                dataset['short_trade_return_%.2f_sigma' %i]

          
            dataset['equity_curve_%.2f_sigma' %i]=(dataset['trade_result_%.2f_sigma' %i]+1).cumprod()
            dataset['noncomp_curve_%.2f_sigma' %i]=(dataset['trade_result_%.2f_sigma' %i]).cumsum()

            # compute transaction costs taking into account that if profit taking is triggered, two trades happend. 
            dataset['trade_result_%.2f_sigma_after_tc' %i] = (dataset['trade_result_%.2f_sigma' %i]+1) * \
            (1.0-transaction_cost*dataset['trade_%.2f_sigma' %i]) -1
            dataset['equity_curve_%.2f_sigma_after_tc' %i]=(dataset['trade_result_%.2f_sigma_after_tc' %i]+1).cumprod()

            # compute the percentage of correct predictions 
            dataset['correct_prediction_%.2f_sigma' %i]= None 
            dataset.loc[dataset['total_trade_return_%.2f_sigma' %i]>0, 'correct_prediction_%.2f_sigma' %i] = 1
            dataset.loc[dataset['total_trade_return_%.2f_sigma' %i]<0, 'correct_prediction_%.2f_sigma' %i] = 0

            #If there are any trades at all, calculate some statistics.
            if (len(dataset['correct_prediction_%.2f_sigma' %i].dropna()))>0:

                pct_correct = sum(dataset['correct_prediction_%.2f_sigma' %i].dropna())/\
                len(dataset['correct_prediction_%.2f_sigma' %i].dropna())
                print('Win rate %.2f_sigma: ' %i + str((pct_correct)*100)+" %")


                # Does the model have a long or short bias?
                percent_betting_up = dataset['signal_%.2f_sigma' %i][dataset['signal_%.2f_sigma' %i]>0].sum()/\
                len(dataset['signal_%.2f_sigma' %i])
                percent_betting_down = -dataset['signal_%.2f_sigma' %i][dataset['signal_%.2f_sigma' %i]<0].sum()/\
                len(dataset['signal_%.2f_sigma' %i])
                out_of_market = 1.00 - (percent_betting_up + percent_betting_down)
                print('Percentage of periods betting up %.2f_sigma : ' %(i)+str(percent_betting_up*100)+' %\n'
                      +'Percentage of periods betting down: %.2f_sigma  ' %i+str(percent_betting_down*100)+' %\n'
                      +'Percentage of periods staying out of the market: %.2f_sigma  ' %i+str(out_of_market*100)+' %\n')
                
                #How many trades were there
                total_trades = dataset['trade_%.2f_sigma' %i].sum()
                print('There were %s total trades for %.2f_sigma.' %(total_trades, i))
                print('The annualised_sharpe for %.2f_sigma. is: %.2f.' %\
                	(i, annualised_sharpe(dataset['trade_result_%.2f_sigma' %i], periods_in_year)))
                print('The CAGR for %.2f_sigma. is: %.2f percent.' %\
                	(i, annual_return(dataset['equity_curve_%.2f_sigma' %i],periods_in_year)*100))

                print('The annualised_sharpe for %.2f_sigma. after commissions is: %.2f.' %\
                	(i, annualised_sharpe(dataset['trade_result_%.2f_sigma_after_tc' %i], periods_in_year)))
                print('The CAGR for %.2f_sigma. is: %.2f percent. after commissions' %\
                	(i, annual_return(dataset['equity_curve_%.2f_sigma_after_tc' %i],periods_in_year)*100))

                average_gain = (dataset['total_trade_return_%.2f_sigma' %i][dataset['total_trade_return_%.2f_sigma' %i]>0]).mean()
                average_loss = (dataset['total_trade_return_%.2f_sigma' %i][dataset['total_trade_return_%.2f_sigma' %i]<0]).mean()
                print('Average winning trade: ' +str(average_gain))
                print('Average losing trade: ' +str(average_loss))
                print('Average trade: ' + \
                    str((dataset['total_trade_return_%.2f_sigma' %i][dataset['total_trade_return_%.2f_sigma' %i]!=0]).mean()))
                print('Average long trade: ' + \
                    str(dataset['long_trade_return_%.2f_sigma' %i][dataset['long_trade_return_%.2f_sigma' %i]!=0].mean()))
                print('Average short trade: ' + \
                    str(dataset['short_trade_return_%.2f_sigma' %i][dataset['short_trade_return_%.2f_sigma' %i]!=0].mean()))
                print('Average time in long trade: ' + \
                    str((dataset['time_in_long_%.2f_sigma' %i][dataset['time_in_long_%.2f_sigma' %i]!=0]).mean()))
                print('Average time in short trade: ' + \
                    str((dataset['time_in_short_%.2f_sigma' %i][dataset['time_in_short_%.2f_sigma' %i]!=0]).mean()))

                print("\n")

                if plot:
                    if not os.path.exists('./Equity_curves'):
                        os.makedirs('Equity_curves')
                    dataset['equity_curve_%.2f_sigma' %i].plot()
                    plt.title('Equity Curve %.2f softmax %.2f Bayesian_z_score' %(i,p))
                    plt.ylabel('Value')
                    plt.xlabel('Period')
                    plt.savefig('Equity_curves/equity_curve_%.2f_softmax_%.2f_Bayesian_z_score.png' %(i,p))
                    plt.close() 
    if plot:
        ((dataset['periodic_return']+1).cumprod()).plot()
        plt.title('Asset price series')
        plt.ylabel('price')
        plt.xlabel('period')
        plt.savefig('Equity_curves/Asset_price_series.png')
        plt.close() 
    return dataset

def annualised_sharpe(returns, periods_in_year):
    """
    Assumes daily returns are supplied. If not change periods in year.
    """
    return np.sqrt(periods_in_year) * returns.mean() / returns.std()

def annual_return(equity_curve, periods_in_year):
    return equity_curve.values[-1]**(periods_in_year/len(equity_curve))-1
    

        