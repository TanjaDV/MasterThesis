# import packages
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from sklearn import model_selection

# import my functions
from pretext_task import create_pretext_model, train_pretext_model
from downstream_task import create_downstream_model, train_downstream_model
from make_plot import plot_history, plot_history_finalization, plot_comparison, plot_comparison_three
from load_data import split_train_val
from my_callbacks import RecordMetricsEvaluation


def run_experiment(config, dir_settings):
    """
    Call the function corresponding to the experiment given in the configuration file
    """

    if 'evaluate-test' in config['experiment']:
        evaluate_downstream(config, dir_settings)
        return

    if config['experiment'] == 'pretext':
        part = 'pretext'
    else:
        part = 'downstream'

    parameter_experiment(config, dir_settings, part)

    return


def parameter_experiment(config, dir_settings, part):
    """
    Framework for the experiments.
    It selects the hyperparameters from the configuration file, makes the folder structure,
    runs the training and makes the plots.
    """
    history = None

    for wd_run in range(len(config[part]['weight_decay'])):
        weight_decay = config[part]['weight_decay'][wd_run]
        for m_run in range(len(config[part]['momentum'])):
            momentum = config[part]['momentum'][m_run]

            # reset to dir_experiment and make subfolder if needed
            results_folder = dir_settings.dir_experiment
            if len(config[part]['weight_decay']) > 1:
                results_folder = os.path.join(results_folder, "w_" + str(weight_decay), "")
            if len(config[part]['momentum']) > 1:
                results_folder = os.path.join(results_folder, "m_" + str(momentum), "")

            lr_runs = len(config[part]['learning_rate'])
            fig, axes = plt.subplots(nrows=lr_runs, ncols=2, sharex='col')
            for lr_run in range(lr_runs):
                learning_rate = config[part]['learning_rate'][lr_run]

                print("Experiment: momentum = {}, learning rate = {} and weight_decay = {}".
                      format(momentum, learning_rate, weight_decay))

                # make subfolder if needed, and save folder to dir_settings
                if lr_runs > 1:
                    dir_settings.set_dir_results(os.path.join(results_folder, "lr_" + str(learning_rate), ""))
                else:
                    dir_settings.set_dir_results(results_folder)

                # training
                model, history = training_model(config, learning_rate, momentum, weight_decay, dir_settings)

                # visualize training
                if history is not None:
                    if config['dataset_mode'] == 'cross-validation':
                        # make table of val_acc on best val_loss epoch
                        with open(dir_settings.dir_experiment + part + "_hpsearch_table.txt", 'a') as f:
                            f.write("weight decay = {:1.4f},  momentum = {:1.4f}, learning rate = {:1.4f}, "
                                    "iterations = {}, val_acc_mean = {:1.4f}, val_acc_std = {:1.4f}\n"
                                    .format(weight_decay, momentum, learning_rate,
                                            history[2], history[0], history[1]))
                    else:
                        plot_history(axes, history, lr_run, lr_runs, title="lr = {}".format(learning_rate))

                        # evaluate model
                        compute_confusion_matrix(model, config, dir_settings, part)

                        # make table of val_acc on best val_loss epoch
                        with open(dir_settings.dir_experiment + part + "_hpsearch_table.txt", 'a') as f:
                            acc_best = history.history['val_acc'][tf.math.argmin(history.history['val_loss'])]
                            f.write("weight decay = {:1.4f},  momentum = {:1.4f},  "
                                    "learning rate = {:1.4f},  val_acc = {:1.4f} \n"
                                    .format(weight_decay, momentum, learning_rate, acc_best))

                # clear session
                tf.keras.backend.clear_session()
            if history is not None and config['dataset_mode'] != 'cross-validation':
                plot_history_finalization(results_folder + "\\" + part + "_", lr_runs, fig, axes)
    return


def training_model(config, learning_rate, momentum, weight_decay, dir_settings):
    if config['dataset_mode'] == "cross-validation":
        model, history = training_cross_validation(config, learning_rate, momentum, weight_decay, dir_settings)
    else:
        model, history = training_val_model(config, learning_rate, momentum, weight_decay, dir_settings)
    return model, history


def training_cross_validation(config, learning_rate, momentum, weight_decay, dir_settings):
    if 'n_folds' not in config.keys():
        n_folds = 5
    else:
        n_folds = config['n_folds']

    if 'cv_min_acc' not in config.keys():
        config['cv_min_acc'] = 0

    # make folds indices
    folds = model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=3)
    splits = folds.split(X=config['cross-val-data'][0], y=config['cross-val-data'][1])

    accuracy_array = []
    dir_parameter = dir_settings.dir_results
    for i, (train_index, val_index) in enumerate(splits):
        dir_settings.set_dir_results(dir_parameter + "fold" + str(i) + "\\")

        # make folds
        train_data = config['cross-val-data'][0][train_index]
        train_labels = config['cross-val-data'][1][train_index]

        val_data = config['cross-val-data'][0][val_index]
        val_labels = config['cross-val-data'][1][val_index]

        # save folds to configuration
        config['pretext']['train_data'] = (train_data, train_labels)
        config['pretext']['val_data'] = (val_data, val_labels)

        config['downstream']['train_data'] = (train_data, train_labels)
        config['downstream']['val_data'] = (val_data, val_labels)

        # train model
        _, history = training_val_model(config, learning_rate, momentum, weight_decay, dir_settings)

        # save validation accuracy of last epoch
        accuracy_array.append(history.history['val_acc'][-1])
        tf.keras.backend.clear_session()

        if accuracy_array[-1] < float(config['cv_min_acc']):
            break

    with open(dir_parameter + "folds_acc_table.txt", 'a') as f:
        f.write("folds accuracy array = {}\n".format(accuracy_array))

    # make history mean and std
    history = (np.mean(accuracy_array), np.std(accuracy_array), i + 1)
    return None, history


def training_val_model(config, learning_rate, momentum, weight_decay, dir_settings):
    """
    Calls the training function corresponding to the experiment given in the configuration file
    """
    if config['experiment'] == 'pretext':
        model, history = training_pretext(config, learning_rate, momentum, weight_decay, dir_settings)
    elif config['experiment'] == 'downstream':
        model, history = training_downstream(config, learning_rate, momentum, weight_decay, dir_settings)
    elif config['experiment'] == 'both':
        model, history = training_both(config, learning_rate, momentum, weight_decay, dir_settings)
    elif config['experiment'] == 'supervised':
        model, history = training_supervised(config, learning_rate, momentum, weight_decay, dir_settings)
    elif config['experiment'] == 'downstream-random':
        model, history = training_downstream_random(config, learning_rate, momentum, weight_decay, dir_settings)
    elif config['experiment'] == 'effect-pretext':
        model, history = training_downstream_effect_pretext(
            config, learning_rate, momentum, weight_decay, dir_settings)
    elif config['experiment'] == 'less-data':
        model, history = training_downstream_with_less_data(
            config, learning_rate, momentum, weight_decay, dir_settings)
    else:
        raise ValueError("experiment '{}' is not recognized".format(config['experiment']))
    return model, history


def training_pretext(config, learning_rate, momentum, weight_decay, dir_settings):
    """
    Trains the pretext task model
    """
    pretext_model = create_pretext_model(config, learning_rate, momentum, weight_decay, dir_settings.dir_experiment)
    pretext_history = train_pretext_model(pretext_model, config, dir_settings)
    return pretext_model, pretext_history


def training_downstream(config, learning_rate, momentum, weight_decay, dir_settings):
    """
    Trains the downstream task model on a loaded pretrained pretext task model
    """
    # test to make sure a pretext_weights_file is specified
    if 'pretext_weights_file' not in config.keys():
        raise ValueError("pretext_weights_file is not given")

    # load pretext task model
    pretext_model = create_pretext_model(config, 0, 0, 0)  # lr, momentum and weight_decay not used
    pretext_model.build(
        input_shape=(None, config['resized_img_size'], config['resized_img_size'], config['num_channels']))
    pretext_model.load_weights(config['pretext_weights_file'])

    # train downstream task model
    downstream_model = create_downstream_model(pretext_model, config, learning_rate, momentum, weight_decay,
                                               dir_settings.dir_experiment)
    downstream_history = train_downstream_model(downstream_model, config, dir_settings)
    return downstream_model, downstream_history


def evaluate_downstream(config, dir_settings):
    # test to make sure a downstream_weights_file is specified
    if 'downstream_weights_file' not in config.keys():
        raise ValueError("downstream_weights_file is not given")

    if config['experiment'] == 'evaluate-test-self-supervised':
        evaluate_downstream_self_supervised(config, dir_settings)
    elif config['experiment'] == 'evaluate-test-supervised':
        evaluate_downstream_supervised(config, dir_settings)
    elif config['experiment'] == 'evaluate-test-random':
        evaluate_downstream_random(config, dir_settings)
    return


def evaluate_downstream_self_supervised(config, dir_settings):
    if 'pretext_weights_file' not in config.keys():
        raise ValueError("pretext_weights_file is not given")

    # load pretext task model
    pretext_model = create_pretext_model(config, 0, 0, 0)  # lr, momentum and weight_decay not used
    pretext_model.build(
        input_shape=(None, config['resized_img_size'], config['resized_img_size'], config['num_channels']))
    pretext_model.load_weights(config['pretext_weights_file'])

    # load downstream task model
    downstream_model = create_downstream_model(pretext_model, config, 0, 0, 0, dir_settings.dir_experiment,
                                               trainable=False)
    downstream_model.build(
        input_shape=(None, config['resized_img_size'], config['resized_img_size'], config['num_channels']))
    downstream_model.load_weights(config['downstream_weights_file'])
    downstream_model.evaluate(x=config['downstream']['val_data'][0],
                              y=config['downstream']['val_data'][1],
                              batch_size=config['batch_size'],
                              callbacks=RecordMetricsEvaluation(dir_settings.dir_results,
                                                                "downstream_evaluation"))
    return


def evaluate_downstream_supervised(config, dir_settings):
    # load pretext task model
    pretext_model = create_pretext_model(config, 0, 0, 0)  # lr, momentum and weight_decay not used

    # load downstream task model
    downstream_model = create_downstream_model(pretext_model, config, 0, 0, 0, dir_settings.dir_experiment,
                                               trainable=True)

    downstream_model.build(
        input_shape=(None, config['resized_img_size'], config['resized_img_size'], config['num_channels']))
    downstream_model.load_weights(config['downstream_weights_file'])
    downstream_model.evaluate(x=config['downstream']['val_data'][0],
                              y=config['downstream']['val_data'][1],
                              batch_size=config['batch_size'],
                              callbacks=RecordMetricsEvaluation(dir_settings.dir_results,
                                                                "supervised_downstream_evaluation"))
    return


def evaluate_downstream_random(config, dir_settings):
    # create untrained pretext task model
    pretext_model = create_pretext_model(config, 0, 0, 0)  # lr, momentum and weight_decay not used

    # train downstream task model
    downstream_model = create_downstream_model(pretext_model, config, 0, 0, 0, dir_settings.dir_experiment,
                                               trainable=False)
    downstream_model.build(
        input_shape=(None, config['resized_img_size'], config['resized_img_size'], config['num_channels']))
    downstream_model.load_weights(config['downstream_weights_file'])
    downstream_model.evaluate(x=config['downstream']['val_data'][0],
                              y=config['downstream']['val_data'][1],
                              batch_size=config['batch_size'],
                              callbacks=RecordMetricsEvaluation(dir_settings.dir_results,
                                                                "random_downstream_evaluation"))
    return


def training_both(config, learning_rate, momentum, weight_decay, dir_settings):
    """
    Trains both the pretext task and downstream task model,
    they use the same dir_settings for learning rate, momentum and weight_decay
    """
    # train pretext task model
    pretext_model = create_pretext_model(config, learning_rate, momentum, weight_decay,
                                         dir_settings.dir_experiment)
    pretext_history = train_pretext_model(pretext_model, config, dir_settings)

    # train downstream task model
    downstream_model = create_downstream_model(pretext_model, config, learning_rate, momentum, weight_decay,
                                               dir_settings.dir_experiment)
    downstream_history = train_downstream_model(downstream_model, config, dir_settings)

    # save pretext plot
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex='col')
    plot_history(axes, pretext_history, run=1, runs=1, title="w_d = {}".format(weight_decay))
    plot_history_finalization(dir_plot=dir_settings.dir_results + "\\pretext_",
                              runs=1, fig=fig, axes=axes)

    return downstream_model, downstream_history


def training_supervised(config, learning_rate, momentum, weight_decay, dir_settings):
    """
    Trains the downstream task model in a supervised way,
    the pretext task model is only used for constructing the model.
    """

    # create untrained pretext task model
    pretext_model = create_pretext_model(config, 0, 0, 0)  # lr, momentum and weight_decay not used

    # train downstream task model
    downstream_model = create_downstream_model(pretext_model, config, learning_rate, momentum, weight_decay,
                                               dir_settings.dir_experiment, trainable=True)
    downstream_history = train_downstream_model(downstream_model, config, dir_settings,
                                                title="supervised_downstream_history")
    return downstream_model, downstream_history


def training_downstream_random(config, learning_rate, momentum, weight_decay, dir_settings):
    # create untrained pretext task model
    pretext_model = create_pretext_model(config, 0, 0, 0)  # lr, momentum and weight_decay not used

    # train downstream task model
    downstream_model = create_downstream_model(pretext_model, config, learning_rate, momentum, weight_decay,
                                               dir_settings.dir_experiment, trainable=False)
    downstream_history = train_downstream_model(downstream_model, config, dir_settings,
                                                title="random_downstream_history")
    return downstream_model, downstream_history


def compute_confusion_matrix(model, config, dir_settings, part, title=None):
    if not title:
        title = part

    # evaluate training set
    train_pred = model.predict(x=config[part]['train_data'], batch_size=config['batch_size'], verbose=1)
    train_confusion_matrix = tf.math.confusion_matrix(train_pred['labels'], train_pred['y_pred'])

    # evaluate validation set
    val_pred = model.predict(x=config[part]['val_data'], batch_size=config['batch_size'], verbose=1)
    val_confusion_matrix = tf.math.confusion_matrix(val_pred['labels'], val_pred['y_pred'])

    # write confusion matrices to file
    with open(dir_settings.dir_results + title + "_confusion_matrix.txt", 'a') as file:
        file.write("training confusion matrix:\n {}".format(train_confusion_matrix))
        file.write("\n\nvalidation confusion matrix:\n {}".format(val_confusion_matrix))
    return


def training_downstream_effect_pretext(config, learning_rate, momentum, weight_decay, dir_settings):
    """
    This function recreates figure 5a from the paper
    In this figure the pretext weights are taken from different epochs,
    we then compare how well they perform in the downstream task.
    """

    # train pretext task
    config['save_pretext_weights'] = 'all'
    _, pretext_history = training_pretext(config, learning_rate, momentum, weight_decay, dir_settings)

    # set weights folder
    config['pretext_weights_folder'] = dir_settings.dir_weights

    # determine the relevant epochs
    epochs = []
    pretext_acc = []
    pretext_loss = []
    value = 0
    for epoch in range(len(pretext_history.history['val_acc'])):
        if pretext_history.history['val_acc'][epoch] >= value or epoch == len(pretext_history.history['val_acc']) - 1:
            value = pretext_history.history['val_acc'][epoch] + 0.02

            epochs.append(epoch)
            pretext_acc.append(pretext_history.history['val_acc'][epoch])
            pretext_loss.append(pretext_history.history['val_loss'][epoch])

    print(epochs)
    print(pretext_acc)

    # initialize arrays
    end_val_loss = []
    end_val_acc = []

    # loop over epochs for training
    n_epochs = len(epochs)
    fig, axes = plt.subplots(nrows=n_epochs, ncols=2, sharex='col')
    for index in range(n_epochs):
        epoch = epochs[index]
        config['pretext_weights_file'] = config['pretext_weights_folder'] + "\\pretext_weights." + \
                                         str(epoch + 1) + ".hdf5"

        print("Experiment: pretext epoch = {}".format(epoch))

        # make subfolder
        dir_settings.set_dir_results(dir_settings.dir_experiment + "\\epoch" + str(epoch) + "\\")

        downstream_model, downstream_history = training_downstream(
            config, learning_rate, momentum, weight_decay, dir_settings)

        # store end results
        end_val_loss.append(downstream_history.history['val_loss'][-1])
        end_val_acc.append(downstream_history.history['val_acc'][-1])

        # visualize training
        plot_history(axes, downstream_history, index, n_epochs, title="epoch = {}".format(epoch))

        # evaluate model
        compute_confusion_matrix(downstream_model, config, dir_settings, 'downstream')

        # clear session
        tf.keras.backend.clear_session()

    # finish and store the history 
    plot_history_finalization(dir_settings.dir_experiment + "\\", n_epochs, fig, axes)

    # plot in one figure
    plot_comparison(dir_settings.dir_experiment,
                    iterable=epochs,
                    loss1=end_val_loss, loss2=pretext_loss,
                    acc1=end_val_acc, acc2=pretext_acc,
                    label1="downstream", label2="pretext",
                    title="effectpretext_", x_label="Pretext epochs",
                    x_ticks=10, y_ticks=0.1)
    return None, None


def training_downstream_with_less_data(config, learning_rate, momentum, weight_decay, dir_settings):
    # initialize arrays
    end_loss = []
    end_supervised_loss = []
    end_random_loss = []
    end_acc = []
    end_supervised_acc = []
    end_random_acc = []

    # train pretext task with all training data
    _, pretext_history = training_pretext(config, learning_rate, momentum, weight_decay, dir_settings)
    config['pretext_weights_file'] = dir_settings.dir_experiment + "pretext_weights.hdf5"

    # save pretext plot
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex='col')
    plot_history(axes, pretext_history, run=1, runs=1, title="w_d = {}".format(weight_decay))
    plot_history_finalization(dir_plot=dir_settings.dir_results + "\\pretext_",
                              runs=1, fig=fig, axes=axes)

    # loop over subsets for training
    n_subsets = len(config['less_data_subsets'])
    fig, axes = plt.subplots(nrows=n_subsets, ncols=2, sharex='col')
    for index in range(n_subsets):
        subset_split = config['less_data_subsets'][index]

        if subset_split == 1:
            config['downstream']['train_data'] = config['pretext']['train_data']
        else:
            train_data, _ = split_train_val(config, config['pretext']['train_data'][0],
                                            config['pretext']['train_data'][1],
                                            val_split=(1 - subset_split))
            config['downstream']['train_data'] = train_data

        # make subfolder
        dir_settings.set_dir_results(dir_settings.dir_experiment + "\\subset" + str(subset_split) + "\\")

        downstream_model, downstream_history = training_downstream(
            config, learning_rate, momentum, weight_decay, dir_settings)

        if 'supervised' in config.keys():
            # supervised does not use the same parameters as self-supervised
            supervised_model, supervised_history = training_supervised(
                config, config['supervised']['learning_rate'][0], config['supervised']['momentum'][0],
                config['supervised']['weight_decay'][0], dir_settings)
        else:
            supervised_model, supervised_history = training_supervised(
                config, learning_rate, momentum, weight_decay, dir_settings)

        if 'random' in config.keys():
            # random does not use the same parameters as self-supervised
            random_model, random_history = training_downstream_random(
                config, config['random']['learning_rate'][0], config['random']['momentum'][0],
                config['random']['weight_decay'][0], dir_settings)
        else:
            random_model, random_history = training_downstream_random(
                config, learning_rate, momentum, weight_decay, dir_settings)

        # store end results
        end_loss.append(downstream_history.history['val_loss'][-1])
        end_acc.append(downstream_history.history['val_acc'][-1])
        end_supervised_loss.append(supervised_history.history['val_loss'][-1])
        end_supervised_acc.append(supervised_history.history['val_acc'][-1])
        end_random_loss.append(random_history.history['val_loss'][-1])
        end_random_acc.append(random_history.history['val_acc'][-1])

        # visualize training
        plot_history(axes, downstream_history, index, n_subsets, title="subset = {}".format(subset_split))

        # evaluate model
        compute_confusion_matrix(downstream_model, config, dir_settings, 'downstream')
        compute_confusion_matrix(supervised_model, config, dir_settings, 'downstream', title='supervised')
        compute_confusion_matrix(random_model, config, dir_settings, 'downstream', title='random')

        # clear session
        tf.keras.backend.clear_session()

    # finish and store the history
    plot_history_finalization(dir_settings.dir_experiment + "\\", n_subsets, fig, axes)

    # plot in one figure
    plot_comparison(dir_settings.dir_experiment,
                    iterable=config['less_data_subsets'],
                    loss1=end_loss, loss2=end_supervised_loss,
                    acc1=end_acc, acc2=end_supervised_acc,
                    label1="self-supervised", label2="supervised",
                    title="lessdata2_", x_label="Percentage of labelled data",
                    x_ticks=0.1, y_ticks=0.1)

    plot_comparison_three(dir_settings.dir_experiment,
                          iterable=config['less_data_subsets'],
                          acc1=end_acc, acc2=end_supervised_acc, acc3=end_random_acc,
                          label1="self-supervised", label2="supervised", label3="random",
                          title="lessdata_", x_label="Percentage of labelled data",
                          x_ticks=0.1, y_ticks=0.1)
    return None, None
