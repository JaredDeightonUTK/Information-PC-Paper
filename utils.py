from datetime import datetime


def generate_run_ID(options):
    ''' 
    Create a unique run ID from the most relevant
    parameters. Remaining parameters can be found in 
    params.npy file. 
    '''

    params = [
        datetime.today().strftime('%Y-%m-%d'), 
        str(options.loss_fn),
        'Np', str(options.Np),
        'Ng', str(options.Ng),
        'steps', str(options.sequence_length),
        'batch', str(options.batch_size),
        'size', str(options.box_height)
        ]
    
    separator = '_'
    run_ID = separator.join(params)
    run_ID = run_ID.replace('.', '')

    return run_ID




