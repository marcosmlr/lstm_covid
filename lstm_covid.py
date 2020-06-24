# coding: utf-8

# --------------------------
#        Imports
# --------------------------
import pandas as pd
import os
import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle
import csv
import math
import resource
import logging
import matplotlib.pyplot as plt
import ast #to eval python expression from strings
from Utils import mkdir_p

plt.style.use('ggplot')

path = os.getcwd()

# --------------------------
#        Functions
# --------------------------

def organizar_dados(input_file,pais):
    logging.captureWarnings(True)
    logger = logging.getLogger(__name__)

    serie_analise = pd.read_csv(input_file,header = None)
    select_analise = serie_analise[serie_analise[0] == pais]    
    select_analise = select_analise[[0,2,3]]

    select_analise.columns = ['region','date','daily_cases']
    select_analise = select_analise.astype(dtype= {"region":"object","daily_cases":"int"})
    select_analise["date"] = pd.to_datetime(select_analise["date"])

    try:
        #Find date of first case in Region Selected:
        index_start = select_analise[select_analise.daily_cases.gt(0)].index[0]
        select_analise = select_analise[select_analise.index >= index_start] #Drop lines until first case            
    except Exception as err:
        print('Please, see the log file for information about the exception occurred!')
        #Getting the Stack Trace:
        logger.exception("Error during the query of the first case of region {}. This is really a valid region? The file contains data about this region?:\n {}".format(pais,err))
        raise
        return 1

    #set date as index
    select_analise.set_index('date',inplace=True)
    
    return select_analise


#split a multivariate sequence into samples 

def split_sequences(sequences, n_steps_in, n_steps_out):
    '''
    Brownlee, J. (2018). Deep Learning for Time Series Forecasting: 
    Predict the Future with MLPs, CNNs and LSTMs in Python. 
    Machine Learning Mastery.
    '''
    X, y = list(), list() 
    for i in range(len(sequences)): 
        #find the end of this pattern 
        end_ix = i + n_steps_in 
        out_end_ix = end_ix + n_steps_out 
        #check if we are beyond the dataset 
        if out_end_ix > len(sequences): 
            break 
        #gather input and output parts of the pattern 
        seq_x, seq_y = sequences[i:end_ix], sequences[end_ix:out_end_ix] 
        X.append(seq_x) 
        y.append(seq_y) 
    return X, y


def preparar_dados(dados_covid, n_dias_entrada, n_dias_saida):    
    # choose a number of time steps 
    # covert into input/output
    dados_covid = dados_covid.daily_cases
    dados_covid = np.float64(dados_covid.tolist())
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = dados_covid.reshape(-1, 1)
    data = scaler.fit_transform(data)
    data = data.reshape(len(data),)
    
    #lista_covid = dados_covid.tolist()
    X, y = split_sequences(data, n_dias_entrada, n_dias_saida)
    
    inputarray = data[-n_dias_entrada:]
    inputarray = [float(i) for i in inputarray]
    inputarray = np.asarray(inputarray)
    X = np.float64(X)
    y = np.float64(y)
    
    return data, X,y, inputarray,scaler


def definir_modelo_lstm_simples(modelo,n_dias_entrada, n_dias_saida):
    n_features = 1
    modelo.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_dias_entrada, n_features)))
    modelo.add(LSTM(200, activation='relu'))
    modelo.add(Dense(n_dias_saida))
    
    modelo.compile(optimizer='adam', loss='mse')
    return modelo

def treinamento(X,y,modelo,epocas,batch):
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # fit model
    history = modelo.fit(X, y, epochs=epocas,batch_size=batch)

    return history

def predict(modelo,inputarray,n_dias_entrada):
    n_features = 1
    x_input = inputarray
    x_input = x_input.reshape((1, n_dias_entrada, n_features))
    yhat = modelo.predict(x_input, verbose=0)
    
    y_future = np.asarray(yhat)
    y_future = yhat.ravel()
    y_future = np.float64(y_future)
    return y_future

def validacao(modelo,n_dias_entrada,inputarray):
    n_features = 1
    x_input = inputarray
    x_input = x_input.reshape((1, n_dias_entrada, n_features))
    yhat = modelo.predict(x_input, verbose=0)
    
    y_future = np.asarray(yhat)
    y_future = yhat.ravel()
    y_future = np.float64(y_future)
    
    return y_future


   
def salvarmodelo(nome,modelo):
    #Salvando
    filename = nome+'.sav'
    pickle.dump(modelo, open(filename, 'wb'))
    

def salvardicionario(nome, dictionary):
    if os.path.exists(nome+".txt"):
        with open(nome+".txt","r") as file:
            x = ast.literal_eval(file.read())
        dictionary = {**x, **dictionary}
        
    #Salvando Dicionario
    with open(nome+".txt", "w") as file:
        file.write( str(dictionary) )
    

def carregaModelo(nomeModelo):
    return pickle.load(open(nomeModelo, 'rb'))


'''
# =============================================================================
# Treinamento do modelo - 
# =============================================================================
'''

def gerarTreinamento_parametros(input_file,pais,**kwargs):
    logging.captureWarnings(True)

    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory usage at Starting gerarTreinamento is: {0} KB".format(mem))

    """
    :type input_file: input data with daily cases of COVID in csv format with collums (Entity,Code,Date,Daily)
    :type pais: Name country to process
    :type kwargs: dict extra args
    
    """
    #Parametros: (path_input,pais)
    daily_cases = organizar_dados(input_file,pais)
    
    #Parametros: (dados_covid, n_dias_entrada, n_dias_saida) -- scaler - toback oringial data
    dados_covid, X,y, input_array, scaler = preparar_dados(daily_cases, kwargs['n_entradas'], kwargs['n_saidas'])
    
    ##Parametros: (modelo,n_dias_entrada, n_dias_saida)
    modelo = Sequential()
    modelo = definir_modelo_lstm_simples(modelo,kwargs['n_entradas'], kwargs['n_saidas'])
    
    #Parametros: (X,y,modelo,epocas,batch)
    history = treinamento(X,y,modelo,kwargs['epochs'],kwargs['batch'])
    #cria dicionario
    history = history.history
    history.keys()
    
    #Salvando Dicionario
    mkdir_p('saved_models')
    salvardicionario(os.path.join('saved_models',"History_"+pais.replace(" ", "")+"_in"+str(kwargs['n_entradas'])+'_out'+str(kwargs['n_saidas'])+'_epochs'+str(kwargs['epochs'])+'_batch'+str(kwargs['batch'])),history)
    salvarmodelo(os.path.join('saved_models',"Modelo_"+pais.replace(" ", "")+"_in"+str(kwargs['n_entradas'])+'_out'+str(kwargs['n_saidas'])+'_epochs'+str(kwargs['epochs'])+'_batch'+str(kwargs['batch'])),modelo)

    

    return modelo
   

'''
# =============================================================================
# GERAR A VALIDAÇÃO DE SÉRIE INTEIRA AUTOMATICAMENTE
# =============================================================================
'''
def geraValidacao(input_file,pais,modelo,**kwargs):
    logging.captureWarnings(True)

    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory usage at Starting geraValidacao is: {0} KB".format(mem))

    """
    :type input_file: input data with daily cases of COVID in csv format with collums (Entity(Region),Code,Date,Daily)
    :type pais: Name country to process
    :type model: Keras object with trained model
    :type kwargs: dict extra args
    
    """
    
    daily_cases = organizar_dados(input_file,pais)
    dados_covid, X,y, input_array, scaler = preparar_dados(daily_cases, kwargs['n_entradas'], kwargs['n_saidas'])

##    times = pd.date_range(daily_cases.index[-1], periods=kwargs['n_saidas']+1, freq='D')
##    df_predict = pd.DataFrame(data=predicted_serie_transform[len(dados_covid)-1:],    # values
##                          index=times,    # index
##                          columns=['daily_cases'])
    
    list_pred = []
    date_pred = []
    list_true = []
    #for i in range(0, len(dados_covid), kwargs['n_saidas']):
    i=0
    while i < len(dados_covid) - kwargs['n_entradas']:
        input_array_test =  np.array(dados_covid[i:(i+kwargs['n_entradas'])])
        #while len(input_array_test) < kwargs['n_entradas']:
        #      input_array_test = np.append(input_array_test,[0],axis = None)
        y_predict_val_test = predict(modelo,input_array_test,kwargs['n_entradas'])
        list_pred.append(y_predict_val_test)
        
        times = pd.date_range(daily_cases.index[i+kwargs['n_entradas']], periods=kwargs['n_saidas'], freq='D')
        date_pred.append(times)
        list_true.append(np.array(dados_covid[(i+kwargs['n_entradas']):(i+kwargs['n_entradas']+kwargs['n_saidas'])]))
##        print('i:',i)
##        print('input_array_test:',input_array_test)
##        print('y_predict_val_test:',y_predict_val_test)
##        print('true:',true)
##
##        print('dados_covid entrada +1:',np.array(dados_covid[i:(i+kwargs['n_entradas']+1)]))

##        if i > 25:
##            break
                    
        i += 1

    #dict_pred = {'date':np.concatenate((date_pred[:])),'daily_cases':np.concatenate((list_pred[:]))}
    #df_predict = pd.DataFrame(dict_pred)
        
    concatenado = np.concatenate((list_pred[:]))
    true = np.concatenate((list_true[:]))
##    true = dados_covid[kwargs['n_entradas']:len(concatenado)+kwargs['n_entradas']]

    if len(true) < len(concatenado):
        lim = (len(concatenado) - len(true))
        concatenado = concatenado[0:len(concatenado) - lim]
        
    score_rmse = math.sqrt(mean_squared_error(concatenado, true))

    print(daily_cases.daily_cases.values)

    #plot data    
    plt.figure() # In this example, all the plots will be in one figure.
    plt.grid()
    for i in range(len(date_pred)):
        ax = plt.plot(date_pred[i],list_pred[i])

    plt.plot(daily_cases.index,dados_covid, color='black');
    plt.show()
    exit()
##    ax = daily_cases.plot(color='blue');
##    #ax = df_predict.plot(style='--', color='red');
    plt.plot(df_predict.date,df_predict.daily_cases,'r--', label = 'Predicted Values');
##    daily_cases.plot(ax=ax,color='blue');
##
##    # specify the lines and labels of the first legend
##    ax.legend(['Predicted Values', 'True Values'],
##          loc='upper right', frameon=False)

##    plt.grid()    
##    plt.plot(concatenado,'r--', label = 'Predicted Values')
##    plt.plot(true,'b',label = 'True Values')
##    plt.xlabel('Days Since First Case')
##    plt.ylabel('Daily Confirmed Cases')
##    plt.legend(loc ='upper right')
##    plt.title('Prediction Score to %s: %.4f RMSE' % (pais,score_rmse))
    
    print("Prediction Score to ",pais," [RMSE]: ", score_rmse)
    score = {'score_rmse':score_rmse}
    
    #Save score at dict
    mkdir_p('saved_models')
    salvardicionario(os.path.join('saved_models',"History_"+pais.replace(" ", "")+"_in"+str(kwargs['n_entradas'])+'_out'+str(kwargs['n_saidas'])+'_epochs'+str(kwargs['epochs'])+'_batch'+str(kwargs['batch'])),score)

    #Save result graph
    output_dir = 'results'
    mkdir_p(output_dir)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(os.path.join(output_dir,"Validation_"+pais.replace(" ", "")+"_in"+str(kwargs['n_entradas'])+'_out'+str(kwargs['n_saidas'])+'_epochs'+str(kwargs['epochs'])+'_batch'+str(kwargs['batch']))+".png", dpi=100)




'''
# =============================================================================
# GERAR A PREDIÇÃO DE PONTOS FUTUROS
# =============================================================================
'''
def geraPrevisao(input_file,pais,modelo,**kwargs):
    logging.captureWarnings(True)
    import datetime

    print(datetime.datetime(2015,7,1).strftime('%B'))

    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory usage at Starting geraPrevisao is: {0} KB".format(mem))

    """
    :type input_file: input data with daily cases of COVID in csv format with collums (Entity(Region),Code,Date,Daily)
    :type pais: Name country to process
    :type model: Keras object with trained model
    :type kwargs: dict extra args
    
    """
    
    daily_cases = organizar_dados(input_file,pais)    
    dados_covid, X,y, input_array, scaler = preparar_dados(daily_cases, kwargs['n_entradas'], kwargs['n_saidas'])

    start_ts = len(dados_covid) - kwargs['n_entradas']
    end_ts = len(dados_covid)
    input_array_2 =  np.array(dados_covid[start_ts:end_ts])

    y_predict_val = predict(modelo,input_array_2,kwargs['n_entradas'])
    predicted_serie = np.concatenate((dados_covid, y_predict_val))
    predicted_serie_transform = scaler.inverse_transform(predicted_serie.reshape(len(predicted_serie),1))
    dados_covid_transform = scaler.inverse_transform(dados_covid.reshape(len(dados_covid),1))

    last_day = daily_cases.index[-1]
    last_day = str('{:%d %B %Y}'.format(last_day))

    times = pd.date_range(daily_cases.index[-1], periods=kwargs['n_saidas']+1, freq='D')
    df_predict = pd.DataFrame(data=predicted_serie_transform[len(dados_covid)-1:],    # values
                          index=times,    # index
                          columns=['daily_cases'])

    #plot data
    plt.grid()
    ax = df_predict.plot(style='--', color='red');
    daily_cases.plot(ax=ax,color='blue');

    # specify the lines and labels of the first legend
    ax.legend(['Predicted Values', 'Confirmed Cases until '+ last_day],
          loc='upper right', frameon=False)

    ax.set_title('Forecasting for '+pais+' daily COVID-19 cases, next '+str(kwargs['n_saidas'])+' days')
    ax.set_ylabel('Number of Daily Cases')
    ax.set_xlabel('Date since first case ['+str('{:%Y-%m-%d}'.format(daily_cases.index[0]))+']')

    #Save result graph
    output_dir = 'results'
    mkdir_p(output_dir)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(os.path.join(output_dir,"Prediction_"+pais.replace(" ", "")+"_in"+str(kwargs['n_entradas'])+'_out'+str(kwargs['n_saidas'])+'_epochs'+str(kwargs['epochs'])+'_batch'+str(kwargs['batch']))+".png", dpi=100)

        


