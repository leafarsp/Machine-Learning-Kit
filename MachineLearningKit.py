import numpy as np
import pandas as pd
from enum import Enum

class activation_function_name(Enum):
    TANH=1
    LOGISTIC=2

class selection_parents_mode(Enum):
    K_TOURNAMENT=1
    ROULETTE_WHEEL=2
    RANK_SELECTION=3

class solver(Enum):
    BACKPROPAGATION=1
    GENETIC_ALGORITHM=2

class layer():
    def __init__(self, m, m_ant):
        self.w = np.ones((m, m_ant + 1))
        self.w_ant = np.zeros((m, m_ant + 1))
        self.y = np.ones(m)
        self.d = np.ones(m)
        self.v = np.ones(m)
        self.delta = np.ones(m)
        self.e = np.ones(m)


class MLPClassifier:

    def __init__(self,
                 hidden_layer_sizes=((10)),
                 activation:activation_function_name=activation_function_name.TANH,
                 learning_rate='constant',
                 solver:solver=solver.BACKPROPAGATION,
                 learning_rate_init=0.001,
                 max_iter=200,
                 shuffle=True,
                 random_state=1,
                 n_individuals=10
                 ):

        self.activation=activation
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.n_individuals = 10

        if type(hidden_layer_sizes) == int:
            self.hidden_layer_sizes=(hidden_layer_sizes,)
            self.L = 2
        else:
            self.hidden_layer_sizes = hidden_layer_sizes
            self.L = len(hidden_layer_sizes)+1
        self.m = [10]
        self.a = 1
        self.b = 1
        self.id = 0.
        self.uniqueId = self.__hash__()
        self.l = list()
        self.weights_initialized = False
        self.fitness = 0.
        self.acertividade = 0
        self.generation = 0
        self.class_distinction_rate = 0.
        self.flag_test_acertividade = False
        self.coefs_=None
        self.intercepts_=None

    def initialize_layers(self, n_input_nodes, n_classes):
        input_nodes = np.array((n_input_nodes))
        output_classes = np.array((n_classes))
        m = np.append(input_nodes, self.hidden_layer_sizes)
        m = np.append(m, output_classes)
        self.m = m

        for i in range(0, self.L):
            self.l.append(layer(m[i + 1], m[i]))

    def get_weights_connected_ahead(self, j, l):
        wlLkj = np.zeros(self.m[l + 2])
        for k in range(0, self.m[l + 2]):
            wlLkj[k] = self.l[l + 1].w[k][j]
        return wlLkj

    def initialize_weights_random(self, weight_limit=10., random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        # weight_limit = 10.
        for l in range(0, self.L):
            self.l[l].w = np.random.rand(self.m[l + 1], self.m[l] + 1) * 2. * (weight_limit) - weight_limit
            # Inicializa o Bias como zero
            for j in range(0, self.m[l + 1]):
                self.l[l].w[j][-1] = 0
        self.weights_initialized = True

    def save_neural_network(self, filename='neural_network.xlsx'):
        max_layer = np.max(self.m)

        data = np.zeros((max_layer + 1, np.sum(self.m[1:])))
        data[:] = np.nan
        arrays = np.zeros((2, np.sum(self.m[1:])))

        end_array = 0
        start_array = 0
        for l in range(0, self.L):

            if l == 0:
                start_array = 0
                end_array = start_array + self.m[l + 1]
            else:
                start_array += self.m[l]
                end_array += self.m[l + 1]

            arrays[0][start_array:end_array] = int(l + 1)
            arrays[1][start_array:end_array] = np.arange(0, self.m[l + 1])

        tuples = list(zip(*arrays))

        columns = pd.MultiIndex.from_tuples(tuples, names=['Layer:', 'Neuron:'])
        df = pd.DataFrame(data=data, columns=columns)
        # print(df)
        for l in range(0, self.L):
            temp_array = np.zeros((self.m[l] + 1, self.m[l + 1]))
            for n in range(0, self.m[l + 1]):
                temp_l = np.transpose(self.l[l].w[n])
                # temp_l = np.transpose(temp_l)
                # print(f'camada={l}, neurônio={n}')
                # print(temp_l)
                # print(df.loc[0:self.m[l], l+1].loc[:,n])
                # print(df.loc[0:self.m[l] + 1, l + 1])
                # df.loc[0:self.m[l], l + 1].loc[:, n] = temp_l
                temp_array[:, n] = temp_l
                # df.loc[0:self.m[l] + 1, l + 1] = temp_l
            df.loc[0:self.m[l], l + 1] = temp_array
        # exit()
        data2 = np.zeros((len(self.m), 4))
        data2[:] = np.nan
        df2 = pd.DataFrame(data=data2, columns=['L', 'm', 'a', 'b'])
        df2['L'][0] = self.L
        df2['m'][0:len(self.m)] = self.m
        df2['a'][0:len(self.m) - 1] = self.a
        df2['b'][0:len(self.m) - 1] = self.b

        with pd.ExcelWriter(filename) as writer:
            df.to_excel(writer, sheet_name='weights')
            df2.to_excel(writer, sheet_name='params')

    def activation_func(self, a, b, v):
        # return 1/(1+ np.exp(-a * v))
        return a * np.tanh(b * v)

    def forward_propagation(self, x):
        if len(x) != self.m[0]:
            print(
                f'Error, input vector has different size from expected. Input size= {len(x)}, Input nodes = {self.m[0]}')
        input = np.append(x, 1)  # acrescenta 1 relativo ao bias
        for l in range(0, self.L):
            for j in range(0, self.m[l + 1]):
                self.l[l].v[j] = np.matmul(np.transpose(self.l[l].w[j]), input)
                self.l[l].y[j] = self.activation_func(self.a[l], self.b[l], self.l[l].v[j])
            input = np.append(self.l[l].y, 1)
        return self.l[self.L - 1].y


    def get_sum_eL(self):
        return np.sum(self.l[-1].e ** 2)

    def calculate_error_inst(self, x, d):
        self.forward_propagation(x)
        return np.sum((d - self.l[self.L - 1].y) ** 2)

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_fitness(self):
        return self.fitness

    def set_acertividade(self, acertividade):
        self.acertividade = acertividade
        self.flag_test_acertividade = True

    def get_acertividade(self):
        return self.acertividade

    def get_flag_teste_acertividade(self):
        return self.flag_test_acertividade

    def get_generation(self):
        return self.generation

    def set_generation(self, generation):
        self.generation = generation

    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id

    def get_output_class(self, threshold=0.8):
        num_out = np.nan
        cont_neuronio_ativo = 0
        for j in range(0, self.m[self.L]):
        # for j in range(self.m[self.L]-1, -1, -1):
            if (self.l[self.L - 1].y[j] > (1 * threshold)):
                # num_out = j
                num_out = j
                cont_neuronio_ativo += 1
            if (cont_neuronio_ativo > 1):
                num_out = np.nan
                break
        return num_out

    def clone(self):
        clone = MLPClassifier(self.L, self.m, self.a, self.b)
        clone.set_fitness(self.get_fitness())
        clone.set_generation(self.get_generation())
        clone.set_id(self.get_id())
        clone.acertividade = self.get_acertividade()
        clone.flag_test_acertividade = self.get_flag_teste_acertividade()
        clone.uniqueId = self.uniqueId
        for l in range(0, self.L):
            for j in range(0, self.m[l + 1]):
                for w in range(0, self.m[l] + 1):
                    clone.l[l].w[j][w] = self.l[l].w[j][w]
        return clone

    # # # function optimized to run on gpu
    # @jit(target_backend='cuda')
    # def output_layer_activation_GPU(self, output_value, num_classes):
    #     d = np.ones(num_classes, dtype=np.float64) * -1
    #     # num = dataset_shufle.iloc[ni, 0]
    #     d[output_value] = 1.
    #     return d

    def output_layer_activation(self, output_value, num_classes):
        d = np.ones(num_classes, dtype=np.float64) * -1.
        # num = dataset_shufle.iloc[ni, 0]
        d[output_value] = 1.
        return d



    def fit(self,X,y):
        if self.solver == 'BackPropagation':
            pass
        elif self.solver == 'Genetic':
            pass

    def predict(self,X):
        pass

