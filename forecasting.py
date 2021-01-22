# coding: utf-8

# --------------------------
#        Imports
# --------------------------
import argparse
import sys, os
import glob
import re
import numpy as np
import resource
import logging
#include parent directory in sys.path:
sys.path.append(os.path.dirname(os.getcwd()))
import lstm_covid

logging.basicConfig(filename='lstm_covid.log',\
                        format='%(asctime)s %(levelname)s:%(message)s', datefmt='%d/%m/%Y %I:%M:%S %p',\
                        level=logging.INFO, \
                        filemode = 'w')

class Action(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        allowed_keywords = ['n_entradas','n_saidas','epochs','batch']
        keyword_dict = {}

        for arg in values:  #values => The args found for keyword_args
            pieces = arg.split('=')

            if len(pieces) == 2 and pieces[0] in allowed_keywords:
                keyword_dict[pieces[0]] = pieces[1]
            else: #raise an error
                #Create error message:
                msg_inserts = ['{}='] * len(allowed_keywords)
                msg_template = 'Example usage: n_entradas=5 n_saidas=7 epochs=50 batch=2. Only {} allowed.'.format(', '.join(msg_inserts))
                msg = msg_template.format(*allowed_keywords)

                raise argparse.ArgumentTypeError(msg)

        setattr(namespace, self.dest, keyword_dict) #The dest key specified in the
                                                    #parser gets assigned the keyword_dict--in
                                                    #this case it defaults to 'keyword_args'


def main(argv):
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory usage at Starting program is: {0} KB".format(mem))

    logging.info('Started')
    log = logging.getLogger()

    args_dict = vars(argv)
    if args_dict['keyword_args'] != None:
        keyword_args = args_dict['keyword_args']
    else:
        keyword_args = {}   
         
    if 'n_entradas' not in keyword_args:
        keyword_args['n_entradas'] = 5 # add default value of input data days of time serie

    if 'n_saidas' not in keyword_args:
        keyword_args['n_saidas'] = 5 # add default value of output days of forecasting

    if 'epochs' not in keyword_args:
        keyword_args['epochs'] = 100 # add default value of number of epochs to training

    if 'batch' not in keyword_args:
        keyword_args['batch'] = 2 # add default value of batch size

    #https://stackoverflow.com/questions/18157376/handle-spaces-in-argparse-input
    argv.country = ' '.join(argv.country) #Handle arguments with spaces

    model_name = os.path.join('saved_models',"Modelo_"+argv.country.replace(" ", "")+"_in"+str(keyword_args['n_entradas'])+'_out'+str(keyword_args['n_saidas'])+'_epochs'+str(keyword_args['epochs'])+'_batch'+str(keyword_args['batch'])+".sav")
        
    if not os.path.exists(model_name):
        models_list = glob.glob(os.path.join("saved_models","*"+argv.country.replace(" ", "")+"*.sav"))

        for i, item in enumerate(models_list, 1):
            print(str(i)+'. ' + item)

        opc = input('\nWhat model do you want to choose? ')

        best_choise = False
        while best_choise != True:
            while not opc.isnumeric():
                opc = input('What model do you want to choose? ')

            if 0 < int(opc) <= (len(models_list)):
                best_choise = True
            else:
                opc = 'NaN'

        model_name = models_list[int(opc)-1]            
        
        m = re.search('_in(\d+)', model_name, re.IGNORECASE)
        if m is not None:
            keyword_args['n_entradas'] = int(m.group(1))

        m = re.search('_out(\d+)', model_name, re.IGNORECASE)
        if m is not None:
            keyword_args['n_saidas'] = int(m.group(1))

        m = re.search('_epochs(\d+)', model_name, re.IGNORECASE)
        if m is not None:
            keyword_args['epochs'] = int(m.group(1))

        m = re.search('_batch(\d+)', model_name, re.IGNORECASE)
        if m is not None:
            keyword_args['batch'] = int(m.group(1))
    
    try:
        model = lstm_covid.carregaModelo(model_name)
        lstm_covid.geraPrevisao(argv.i,argv.country,model,**keyword_args)            
    except Exception as err:
        print('Please, see the log file for information about the exception occurred!')
        log.exception("Error while running carregaModelo e geraPrevisao:\n %s", err)
        return 1
       
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory usage at Finishing program is: {0} KB".format(mem))

    logging.info('Finished')
    return 0


if __name__ == "__main__":

    # Prompt user for (optional) command line arguments, when run from IDLE:
    if 'idlelib' in sys.modules: sys.argv.extend(input("Args: ").split())

    # Process the arguments
    import argparse
    import arghelper

    parser = argparse.ArgumentParser(
        description='Forecasting daily cases of COVID-19 using LSTM models tunned.')

    parser.add_argument('-i',
                       help='Input file of time serie, for example /home/daily_case.csv. With collums (Region,Code,Date,DailyCases)',
                        metavar='input_file', type=lambda x: arghelper.is_valid_file(parser, x), required=True)
    
    parser.add_argument('-country', nargs='+', help='Name of region as like in input time serie data, for example Italy or Bosnia and Herzegovina.',
                        metavar='country_name', type=str, required=True)

    # Using argparse with function that takes kwargs argument - https://stackoverflow.com/a/33712815
    parser.add_argument("-keyword_args", help="Extra args (Not required, program has default values). Example usage: n_entradas=5 n_saidas=7 epochs=50 batch=2", nargs='*', action=Action)

    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    sys.exit(main(parser.parse_args()))
