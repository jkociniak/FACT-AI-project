import os


def make_syn_data():

    return


def string_to_data(string):
    # TODO
    return 


def data_to_string(data):
    # TODO
    return


def load_syn_data():
    """
    Returns synthetic data by loading it from disk,
    or making the data if the file does not exists
    and saving it to disk.
    """
    
    path = "./data/syn_data.txt"

    # Load data from folder if it is available
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = string_to_data(f.read())

    # Otherwise make the data and save it to the folder
    else:
        # Make data
        data = make_syn_data()

        # Save to file
        with open(path, 'w') as f:
            f.write(data_to_string(data))


    return data

