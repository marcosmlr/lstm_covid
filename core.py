# coding: utf-8

# --------------------------
#        Imports
# --------------------------
import argparse
import sys, os
import numpy as np
import resource
import logging
#include parent directory in sys.path:
sys.path.append(os.path.dirname(os.getcwd()))
import lstm_covid

#https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
#https://www.internalpointers.com/post/logging-python-sub-modules-and-configuration-files
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
                keyword_dict[pieces[0]] = int(pieces[1])
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
        keyword_args['epochs'] = 10 # add default value of number of epochs to training

    if 'batch' not in keyword_args:
        keyword_args['batch'] = 2 # add default value of batch size
        
    #https://stackoverflow.com/questions/18157376/handle-spaces-in-argparse-input
    argv.country = ' '.join(argv.country) #Handle arguments with spaces


    if argv.i is not None:
        try:
            model = lstm_covid.gerarTreinamento_parametros(argv.i,argv.country,**keyword_args)
            lstm_covid.geraValidacao(argv.i,argv.country,model,**keyword_args)            
        except Exception as err:
            print('Please, see the log file for information about the exception occurred!')
            log.exception("Error while running gerarTreinamento e geraValidacao:\n %s", err)
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
        description='Tunning LSTM models to COVID-19 data to forercasting cases.')

    parser.add_argument('-i',
                       help='Input file of time serie, for example /home/daily_case.csv. With collums (Region,Code,Date,DailyCases)',
                        metavar='input_file', type=lambda x: arghelper.is_valid_file(parser, x), required=True)
    
    parser.add_argument('-country', nargs='+', help='Name of region as like in input time serie data, for example Italy or Bosnia and Herzegovina.',
                        metavar='country_name', type=str, required=True)

    # Using argparse with function that takes kwargs argument - https://stackoverflow.com/a/33712815
    parser.add_argument("-keyword_args", help="Extra args. Example usage: n_entradas=5 n_saidas=7 epochs=50 batch=2", nargs='*', action=Action)

    
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    sys.exit(main(parser.parse_args()))
