# import packages
import argparse

# import my functions
from load_data import load_config_file, initialize_results_directory, save_config, load_data
from experiments import run_experiment
from Settings import Settings

# tf.config.run_functions_eagerly(True)

if __name__ == "__main__":
    # parse command-line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("configuration_file", help="a yaml file containing the configuration")
    args = parser.parse_args()

    # load config file to parameter config
    config = load_config_file(args.configuration_file)

    # make directory to store results of this experiment
    dir_main_experiment = initialize_results_directory(config)

    # save config file information for future reference
    save_config(config, dir_main_experiment)

    # load data
    load_data(config)

    # initialize dir_settings class to store the results directories
    dir_settings = Settings(dir_experiment=dir_main_experiment)

    # run experiment
    run_experiment(config, dir_settings)
