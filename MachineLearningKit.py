import copy
import datetime as dt
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
        self.w_correction_ant = np.zeros((m, m_ant + 1))
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
                 momentum=0,
                 n_individuals=10,
                 weight_limit=1
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
        input_nodes = np.array([2])
        output_nodes = np.array([2])
        self.m=np.append(np.append(input_nodes,np.array(self.hidden_layer_sizes)),output_nodes)
        self.a = np.ones(self.L)
        self.b = np.ones(self.L)

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
    def d_func_ativacao(self, a, b, v):
        # return (a * np.exp(-a * v)) / ((1 + np.exp(-a * v))**2)
        return a * b * (1 - np.tanh(b * v) ** 2)

    def forward_propagation(self, x):
        if len(x) != self.m[0]:
            raise ValueError(
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
        return copy.deepcopy(self)

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

    def backward_propagation(self, x, d, alpha, eta):
        if len(d) != self.m[-1]:
            raise ValueError(
                f'Error, input vector has different size from expected. Input size= {len(x)}, Input nodes = {self.m[-1]}')
        # self.forward_propagation(x)
        output_d = np.append(d, 1)
        for l in range(self.L - 1, -1, -1):
            for j in range(0, self.m[l + 1]):
                # print(f'l={l}, j= {j}')
                if l == (self.L - 1):
                    self.l[l].e[j] = output_d[j] - self.l[l].y[j]
                else:
                    self.l[l].e[j] = np.sum(self.l[l + 1].delta * self.get_weights_connected_ahead(j, l))

                self.l[l].delta[j] = self.l[l].e[j] * self.d_func_ativacao(self.a[l], self.b[l], self.l[l].v[j])
                if l == (0):
                    input = np.append(x, 1)
                else:
                    input = np.append(self.l[l - 1].y, 1)

                w_correction = alpha[l] * self.l[l].w_correction_ant[j] + eta[l] * self.l[l].delta[j] * input

                self.l[l].w_correction_ant[j] = w_correction
                # w_temp = self.l[l].w[j] + alpha[l] * self.l[l].w_ant[j] + eta[l] * self.l[l].delta[j] * input
                # self.l[l].w_ant[j] = np.copy(self.l[l].w[j])
                self.l[l].w[j] += w_correction

    def fit(self,X,y):
        if self.solver == 'BackPropagation':
            train_neural_network(
                rede=self,
                num_classes=self.m[self.L],
                rnd_seed=self.random_state,
                dataset=dataset,
                test_dataset=test_dataset,
                n_epoch=n_epoch,
                step_plot=step_plot,
                learning_rate=eta,
                momentum=alpha,
                err_min=err_min,
                weight_limit=1.,
                learning_rate_end=0.)
        elif self.solver == 'Genetic':
            pass

    def predict(self, X):
        return self.forward_propagation(X)

def train_neural_network(rede : MLPClassifier, X,y,):

    start_time = dt.datetime.now()
    num_classes = rede.m(rede.L)
    rnd_seed = rede.random_state

    print(f'Start time: {start_time.year:04d}-{start_time.month:02d}-{start_time.day:02d}'
          f'--{start_time.hour:02d}:{start_time.minute:02d}:{start_time.second:02d}')

    # Base de dados de treinamento


    # cria rede neural
    # rede = rede

    n_inst = np.shape(X)[0]

    # parâmetros de treinamento da rede
    n_epoch = rede.n_epoch

    N = n_inst * n_epoch
    step_plot = int(N / (n_epoch * 1))

    n_cont = 0

    eta = np.ones((rede.L, N))
    for l in range(0, rede.L):
        eta[l] = list(np.linspace(rede.learning_rate_init, rede.learning_rate_init, N))

    # for l in range(0,a1.L):
    #   plt.plot(eta[l])
    #   pass
    eta = np.transpose(eta)
    # eta[:, [1, 0]]

    # plt.figure()

    alpha = np.ones((rede.L, N))
    for l in range(0, rede.L):
        alpha[l] = list(np.linspace(rede.momentum, 0., N))
    # alpha[0] *= 0.000000  # camada de entrada
    # alpha[1] *= 0.000000  # camada oculta 1
    # alpha[2] *= 0.000000  # camada de saída
    # alpha[3] *= 0.000000  # camada de saída
    alpha = np.transpose(alpha)

    # Inicializa os pesos com valores aleatórios e o bias como zero
    if rede.weights_initialized == False:
        rede.initialize_weights_random(random_seed=rnd_seed, weight_limit=rede.weight_limit)

    # Vetor de pesos para plotar gráficos de evolução deles.
    a1plt = list()
    acert = list()

    Eav = np.zeros(n_epoch)
    # início do treinamento
    start_time_epoch = dt.datetime.now()
    for ne in range(0, n_epoch):
        dataset_shufle = shufle_dataset(dataset=dataset, rnd_seed=rnd_seed)

        rnd_seed += 1
        e_epoch = 0

        for ni in range(0, n_inst):
            n = ni + ne * (n_inst)
            if n >= (N - 1):
                break
            x = list(dataset_shufle.iloc[ni, 1:(rede.m[0] + 1)])
            output_value = int(dataset_shufle.iloc[ni, 0])
            # d = [dataset_shufle.iloc[ni, 0]]
            d = rede.output_layer_activation(output_value=output_value, num_classes=num_classes)
            rede.forward_propagation(x=x)
            rede.backward_propagation(x=x, d=d, alpha=alpha[n], eta=eta[n])

            if n >= step_plot:
                if n % step_plot == 0:
                    rede.flag_test_acertividade = False
                    teste_acertividade(test_dataset, int(num_classes), rede)
                    acert.append(rede.get_acertividade())
                    elapsed_time = dt.datetime.now() - start_time_epoch
                    start_time_epoch = dt.datetime.now()
                    estimated_time_end = start_time_epoch + elapsed_time * (N // step_plot - n_cont)
                    n_cont += 1
                    print(f'Instância {n}/{N}, Época {ne}/{n_epoch}, Erro médio: {Eav[ne-1]:.7f}'
                          f' Acert.: {acert[-1]:.4f}%, eta[L][n]: {eta[n][rede.L - 1]:.4f}, dt: {elapsed_time.seconds}s'
                          f' t_end: {estimated_time_end.year:04d}-{estimated_time_end.month:02d}-{estimated_time_end.day:02d}'
                          f'--{estimated_time_end.hour:02d}:{estimated_time_end.minute:02d}:{estimated_time_end.second:02d}')
                    temp_rede = rede_neural(rede.L, rede.m, rede.a, rede.b)
                    for l in range(0, rede.L):
                        temp_rede.l[l].w = np.copy(rede.l[l].w)
                    a1plt.append(temp_rede)
                    # a1.save_neural_network('backup_neural_network.xlsx')

            e_epoch += rede.get_sum_eL()
        Eav[ne] = 1 / (n_inst) * e_epoch

        # print(f'Erro Época {ne}/{n_epoch}: {Eav[ne]:.5f}')
        # A linha abaixo calcula a média como escrito no livro, mas
        # não tem muito sentido calcular desse jeito, o erro
        # fica menor se o número de épocas aumenta.
        # Se eu pegar uma rede que foi treinada desse jeito e chegou
        # num erro 0,0001 por exemplo, se eu testá-la novamente
        # com apenas uma época, o erro vai ser maior.
        # Pra mim esse valor deveria ser fíxo, independente do
        # número de épocas. Dessa forma, eu obteria o mesmo erro,
        # seja após 1000 épocas ou após apenas uma.
        # Eav[ne] += Eav[ne] + 1/(2*N) * e_epoch
        if (Eav[ne] < err_min):
            print(f'Erro mínimo: {Eav[ne]}')
            break
    # teste da rede neural

    return rede, a1plt, Eav, n, acert