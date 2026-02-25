from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


class WineDatasetHandler():
    def __init__(self):
        wine_dataset = load_wine()
        self.input_data  = wine_dataset.data
        self.target_data = wine_dataset.target
        self.class_names = wine_dataset.target_names
        self.data = {}

    def get_dataset_info(self):
        num_samples, num_features = self.input_data.shape
        return num_samples, num_features, self.class_names

    def split_dataset(self,
                      test_size:  float = 0.2,
                      random_state: int = 42):
        (self.data['X_train'], self.data['X_test'],
         self.data['y_train'], self.data['y_test']) = train_test_split(
            self.input_data,
            self.target_data,
            test_size    = test_size,
            random_state = random_state
        )
