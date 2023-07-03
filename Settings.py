import os


class Settings:
    """
    This class keeps track of the directory we want to store results in
    """

    def __init__(self, dir_experiment):
        if not os.path.exists(dir_experiment):
            os.makedirs(dir_experiment)
        self.dir_experiment = dir_experiment
        self.dir_results = dir_experiment
        self.dir_weights = dir_experiment

    def set_directories(self, dir_experiment):
        if not os.path.exists(dir_experiment):
            os.makedirs(dir_experiment)
        self.dir_experiment = dir_experiment
        self.dir_results = dir_experiment

    def set_dir_results(self, dir_results):
        if not os.path.exists(dir_results):
            os.makedirs(dir_results)
        self.dir_results = dir_results

    def set_dir_weights(self, mode):
        if mode == 'all':
            if not os.path.exists(self.dir_results + "\\pretext_weights"):
                os.makedirs(self.dir_results + "\\pretext_weights")
            self.dir_weights = self.dir_results + "\\pretext_weights\\"
        elif mode == 'best':
            self.dir_weights = self.dir_results
        else:
            raise ValueError("save_pretext_weights mode '{}' is not recognized".format(mode))
