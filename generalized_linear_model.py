from data_loader import DataLoader
import numpy as np
import pandas as pd
import pprint
import time as tm

class GeneralizedLinearModel:

    def __init__(self, input_vector_degree, feature_vector_degree, lambda_val=2):
        self.M = input_vector_degree
        self.basis_function_degree = feature_vector_degree
        self.basis_vector_size = ((self.basis_function_degree + 1) * (self.basis_function_degree + 2)) // 2
        self.lambda_val = lambda_val
        self.gram_matrix = None
        self.a_vector = None
        self.mse_error = None
        self.training_time = None

    def get_basis_function_vector(self, x):

        basis_function_vector = np.empty(shape=(0, 0), dtype=float)
        for p in range(self.basis_function_degree + 1):
            for q in range(p + 1):
                basis_function_vector = np.append(basis_function_vector, (x[0]**q) * (x[1]**(p-q)))

        return basis_function_vector.reshape((self.basis_vector_size, 1))

    def compute_gram_matrix(self, dataset):

        phi_matrix = None

        for train_set_attrs, train_set_labels in dataset:

            if len(train_set_attrs) != len(train_set_labels):
                raise ValueError('Count mismatch between attributes and labels')

            for i, row in train_set_attrs.iterrows():
                xi = row.values.reshape((self.M, 1))
                phi_xi = self.get_basis_function_vector(xi)
                if phi_matrix is None:
                    phi_matrix = phi_xi
                else:
                    phi_matrix = np.hstack((phi_matrix, phi_xi))

        self.gram_matrix = np.matmul(np.transpose(phi_matrix), phi_matrix)
        return self.gram_matrix

    def compute_a_vector(self, dataset):
        inv_matrix = np.linalg.inv(np.add(self.gram_matrix,
                                          self.lambda_val * np.identity(self.gram_matrix.shape[0], dtype=float)))

        y_vector = []
        for train_set_attrs, train_set_labels in dataset:

            if len(train_set_attrs) != len(train_set_labels):
                raise ValueError('Count mismatch between attributes and labels')

            y_vector += train_set_labels.ix[:, 0].tolist()

        y_vector = np.array(y_vector, dtype=float)
        y_vector.reshape((y_vector.shape[0], 1))

        self.a_vector = np.matmul(inv_matrix, y_vector)
        return self.a_vector

    def compute_kernel_vector(self, dataset, new_x):

        kernel_vector = np.empty(shape=(0, 0), dtype=float)
        phi_new_x = self.get_basis_function_vector(new_x)

        for train_set_attrs, train_set_labels in dataset:

            if len(train_set_attrs) != len(train_set_labels):
                raise ValueError('Count mismatch between attributes and labels')

            for i, row in train_set_attrs.iterrows():
                xi = row.values.reshape((self.M, 1))
                phi_xi = self.get_basis_function_vector(xi)
                kernel_vector = np.append(kernel_vector,
                                          np.matmul(phi_new_x.reshape((1, self.basis_vector_size)), phi_xi))

        return kernel_vector

    def learn(self, dataset, report_error=False):

        if report_error:
            start_time = tm.time()

        self.compute_gram_matrix(dataset)
        self.compute_a_vector(dataset)

        if report_error:
            self.mse_error = self.k_fold_cross_validation(dataset)
            print('Mean Square Error = %.3f ' % self.mse_error)
            end_time = tm.time()
            self.training_time = (int((end_time - start_time) * 100)) / 100
            print('Training time =', self.training_time, 'seconds')

    def predict_point(self, dataset, new_x):
        prediction = np.matmul(self.compute_kernel_vector(dataset, new_x), self.a_vector)
        return prediction

    def predict(self, train_dataset, test_attrs, true_values=None):
        N = len(test_attrs)
        if not true_values.empty:
            if len(test_attrs) != len(true_values):
                raise ValueError('count mismatch in attributes and labels')
            error = 0.0

        predicted_values = []
        for i, row in test_attrs.iterrows():
            xi = row.values.reshape((self.M, 1))
            predicted_value = self.predict_point(train_dataset, xi)
            predicted_values.append(predicted_value)
            if not true_values.empty:
                true_value = true_values.iat[i, 0]
                error += (true_value - predicted_value) ** 2

        E_MSE = None
        if true_values is not None:
            E_MSE = error / N

        predicted_values = pd.DataFrame(np.array(predicted_values))
        return predicted_values, E_MSE

    def k_fold_cross_validation(self, dataset, k=10):
        cv_test_model = GeneralizedLinearModel(input_vector_degree=self.M,
                                               feature_vector_degree=self.basis_function_degree)
        avg_E_MSE = 0.0
        for i in range(k):
            test_attrs, test_labels = dataset.pop(0)
            cv_test_model.learn(dataset)
            E_MSE = cv_test_model.predict(dataset, test_attrs, true_values=test_labels)[1]
            dataset.append((test_attrs, test_labels))
            avg_E_MSE += E_MSE

        avg_E_MSE = avg_E_MSE / k
        return avg_E_MSE

    def summary(self):
        print('=====Model Summary=====')
        print('Input vector size =', self.M)
        print('Basis function degree =', self.basis_function_degree)
        print('Feature vector size =', self.basis_vector_size)

        print('\nGram Matrix of size', end=' ')
        print(self.gram_matrix.shape, ':')
        pprint.pprint(self.gram_matrix)
        print('\na vector of size', end=' ')
        print(self.a_vector.shape, ':')
        pprint.pprint(self.a_vector)
        if self.mse_error:
            print('Mean Square Error = %.3f ' % self.mse_error)


if __name__ == '__main__':

    # Test get_basis_function_vector
    print('\n===Test get_basis_function_vector===')
    model = GeneralizedLinearModel(input_vector_degree=2, feature_vector_degree=2)
    result = model.get_basis_function_vector(np.array([3, 4]))
    print('phi(x) shape=', result.shape)
    print('phi(x) = ', result)

    # Test compute_gram_matrix
    print('\n===Test compute_gram_matrix===')
    model = GeneralizedLinearModel(input_vector_degree=2, feature_vector_degree=2)
    train_attrs, train_labels = DataLoader.load_dataset(
        './regression-dataset/fData1.csv',
        './regression-dataset/fLabels1.csv'
    )
    result = model.compute_gram_matrix([(train_attrs, train_labels)])
    print('Gram Matrix shape=', result.shape)
    print('=====Gram Matrix====')
    pprint.pprint(result)

    # Test compute_kernel_vector
    print('\n===Test compute_kernel_vector===')
    model = GeneralizedLinearModel(input_vector_degree=2, feature_vector_degree=2)
    train_attrs, train_labels = DataLoader.load_dataset(
        './regression-dataset/fData1.csv',
        './regression-dataset/fLabels1.csv'
    )
    test_new_x = np.array([2, 3]).reshape((2, 1))
    result = model.compute_kernel_vector([(train_attrs, train_labels)], test_new_x)
    print('Kernel Vector shape=', result.shape)
    print('=====Kernel Vector====')
    print(result)

    # Test compute_a_vector
    print('\n===Test compute_a_vector===')
    model = GeneralizedLinearModel(input_vector_degree=2, feature_vector_degree=2)
    train_attrs, train_labels = DataLoader.load_dataset(
        './regression-dataset/fData1.csv',
        './regression-dataset/fLabels1.csv'
    )
    result = model.compute_gram_matrix([(train_attrs, train_labels)])
    result = model.compute_a_vector([(train_attrs, train_labels)])
    print('a Vector shape=', result.shape)
    print('=====a Vector====')
    print(result)

    # Test learn with cross validation for different values of basis function degree
    print('\n===Test learn with cross validation for different values of basis function degree===')
    for d in range(1, 5):
        model = GeneralizedLinearModel(input_vector_degree=2, feature_vector_degree=d)
        full_dataset = DataLoader.load_full_dataset('./regression-dataset')
        print('\nLearning in progress for basis function degree =', d)
        model.learn(full_dataset, report_error=True)
        model.summary()








