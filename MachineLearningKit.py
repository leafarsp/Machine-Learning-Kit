import copy
import datetime as dt
import numpy as np
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier as skl
import pickle

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
                 hidden_layer_sizes=((10,)),
                 activation: activation_function_name = activation_function_name.TANH,
                 learning_rate='constant',
                 solver=solver.BACKPROPAGATION,
                 learning_rate_init=0.001,
                 max_iter=200,
                 shuffle=True,
                 random_state=1,
                 momentum=0,
                 n_individuals=10,
                 weight_limit=1,
                 batch_size='auto',
                 tol=0.0001,
                 power_t=0.5,
                 n_iter_no_change = 10,
                 activation_lower_value = -1.
                 ):

        self.activation=activation
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.n_individuals = n_individuals
        self.momentum = momentum
        self.tol=tol
        self.weight_limit=weight_limit
        self.batch_size=batch_size
        self.solver=solver
        self.power_t=power_t
        self.n_iter_no_change=n_iter_no_change

        if type(hidden_layer_sizes) == int:
            self.hidden_layer_sizes = (hidden_layer_sizes,)
            self.L = 2
        else:
            self.hidden_layer_sizes = hidden_layer_sizes
            self.L = len(hidden_layer_sizes)+1

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
        self.flag_teste_acertividade = False
        self.coefs_= None
        self.intercepts_= None
        self.layers_initialized = False
        self.activation_lower_value = activation_lower_value

    def initialize_layers(self, n_input_nodes, n_classes):

        input_nodes = np.array((n_input_nodes))
        output_classes = np.array((n_classes))
        m = np.append(input_nodes, self.hidden_layer_sizes)
        m = np.append(m, output_classes)
        self.m = m

        for i in range(0, self.L):
            self.l.append(layer(m[i + 1], m[i]))
        layers_initialized = True

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
            self.l[l].w = np.random.rand(self.m[l + 1], self.m[l] + 1) * 2. * weight_limit - weight_limit
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
        self.flag_teste_acertividade = True

    def get_acertividade(self):
        return self.acertividade

    def get_flag_teste_acertividade(self):
        return self.flag_teste_acertividade

    def get_generation(self):
        return self.generation

    def set_generation(self, generation):
        self.generation = generation

    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id

    def get_output_class(self, y=None, threshold=0.8):
        num_out = np.nan
        cont_neuronio_ativo = 0

        if y is not None:
            y_l = y
        else:
            y_l = self.l[self.L - 1].y

        for j in range(0, len(y_l)):
            if y_l[j] > (1 * threshold):
                num_out = j
                cont_neuronio_ativo += 1
            if cont_neuronio_ativo > 1:
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

    def output_layer_activation(self, output_value,
                                num_classes):

        d=output_layer_activation(output_value,num_classes,
                                self.activation_lower_value)
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

    def fit(self, X, y):
        Eav= None
        if not self.weights_initialized:
            self.initialize_layers(len(X[0]),len(y[0]))
        if self.solver == solver.BACKPROPAGATION:
            Eav = train_neural_network(rede=self, X=X, y=y)
        elif self.solver == solver.GENETIC_ALGORITHM:
            pass
        return Eav

    def predict(self, X):
        return self.forward_propagation(X[0])

    def save_nn_obj(obj, filename):
        with open(filename, 'wb') as outp:
            # Step 3
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_nn_obj(filename):
    with open(filename, 'rb') as inp:
        clf = pickle.load(inp)

    return clf

def shufle_dataset(X,y):
    n_inst = np.shape(y)[0]
    in_nodes = np.shape(X)[1]
    out_nodes = np.shape(y)[1]
    dataset = np.c_[y,X]
    np.random.shuffle(dataset)
    y=dataset[:,0:out_nodes]
    X=dataset[:,out_nodes:]
    return X,y


def train_neural_network(rede: MLPClassifier, X: list, y: list):

    rnd_seed = rede.random_state

    n_inst = np.shape(X)[0]
    cnt_iter=0
    n_epoch = rede.max_iter

    N = n_inst * n_epoch

    eta = np.ones((rede.L, N))
    if rede.learning_rate == 'constant':
        for l in range(0, rede.L):
            eta[l] = list(np.linspace(rede.learning_rate_init, rede.learning_rate_init, N))

        alpha = np.ones((rede.L, N))
        for l in range(0, rede.L):
            alpha[l] = list(np.linspace(rede.momentum, rede.momentum, N))

    elif rede.learning_rate == 'invscaling':
        t = np.linspace(rede.learning_rate_init,N,N)
        for l in range(0, rede.L):
            eta[l] = rede.learning_rate_init / pow(t, rede.power_t)

        t = np.linspace(rede.momentum, N, N)
        alpha = np.ones((rede.L, N))
        for l in range(0, rede.L):
            alpha[l] = rede.momentum / pow(t, rede.power_t)
    else:
        for l in range(0, rede.L):
            eta[l] = list(np.linspace(rede.learning_rate_init, 0, N))

        alpha = np.ones((rede.L, N))
        for l in range(0, rede.L):
            alpha[l] = list(np.linspace(rede.momentum, 0., N))



    eta = np.transpose(eta)
    alpha = np.transpose(alpha)

    # Inicializa os pesos com valores aleatórios e o bias como zero
    if not rede.weights_initialized:
        rede.initialize_weights_random(random_seed=rede.random_state, weight_limit=rede.weight_limit)

    Eav = np.zeros(n_epoch)
    E_inst = np.zeros(N)
    # início do treinamento
    for ne in range(0, n_epoch):
        X_l, y_l = shufle_dataset(X, y)

        rnd_seed += 1
        e_epoch = 0

        for ni in range(0, n_inst):
            n = ni + ne * (n_inst)
            if n >= (N - 1):
                break
            rede.forward_propagation(x=X_l[ni])
            rede.backward_propagation(x=X_l[ni], d=y_l[ni],
                                      alpha=alpha[n], eta=eta[n])
            e_epoch += rede.get_sum_eL()
            E_inst[cnt_iter] = rede.get_sum_eL()
            cnt_iter += 1

        Eav[ne] = 1 / (n_inst) * e_epoch
        # Faz a avaliação dos últims treinamento depois de x instâncias

        #
        #
        print(f'Epoch: {ne}/{n_epoch}, loss: {Eav[ne]}')
        stop_training = eval_stop_training(cnt_iter=ne,
                                           n_iter_no_change=rede.n_iter_no_change,
                                           err=Eav, tol=rede.tol)
        if stop_training:
            print(f'Training loss did not improve more than '
                  f'tol={rede.tol:.8f} for {rede.n_iter_no_change} '
                  f'consecutive epochs. Stopping.')
            break
        # if Eav[ne] < rede.tol:
        #     break

    return Eav, ne

def eval_stop_training(cnt_iter, n_iter_no_change, err, tol):
    result = False
    cnt_err_no_change = 0
    err_chg = 0

    if cnt_iter > n_iter_no_change:
        # avalia as últimas instancias
        for i in range(cnt_iter - n_iter_no_change,
                       cnt_iter):
            err_chg = abs(err[i]-err[i-1])
            if err_chg < tol:
                cnt_err_no_change += 1
            else:
                cnt_err_no_change = 0

            if cnt_err_no_change >= n_iter_no_change:
                result = True

    # if result:
    #     print(f'result = {result}, cnt_iter={cnt_iter}'
    #           f' n_iter_no_change={n_iter_no_change}')
    #     print(err[cnt_iter - n_iter_no_change:cnt_iter])
    return result

def teste_acertividade(X: list, y: list, rede: MLPClassifier,
                       print_result=False, save_result=False,
                       filename='training_results.xlsx'):
    cont_acert = 0
    wrong_text = ' - wrong'
    n_inst = np.shape(X)[0]
    columns = ['Instance','Real class', 'Predicted class', 'Result']

    columns += list(np.arange(rede.m[-1]))
    df = pd.DataFrame(columns = columns)
    result=0

    if rede.get_flag_teste_acertividade() == False:
        for i in range(0, n_inst):

            num_real = rede.get_output_class(y[i])
            y_l = rede.forward_propagation(X[i])
            num_rede = rede.get_output_class()
            comparision_result = False
            if num_rede != np.nan:

                if (num_real == num_rede):
                    cont_acert += 1
                    comparision_result = True
                    wrong_text = ""

            list_row = [i,num_real, num_rede, comparision_result]

            list_row = list(list_row) + list(y_l)


            df.loc[len(df)] = list_row

            if print_result:
                print(f'Núm. real: {num_real}, núm rede: {num_rede}{wrong_text}') #, neurônios: {y_l}')
            wrong_text = ' - wrong'
        result = 100 * cont_acert / n_inst
        rede.set_acertividade(result)

        list_row = ['', '', 'Accuracy', f'{result:.2f}%']
        list_row += ['']*len(y[0])
        df.loc[len(df)] = list_row
        if save_result==True:
            df.to_excel(filename,sheet_name='Results')
    return result


def get_output_class(y, threshold=0.8):
    num_out = np.nan
    cont_neuronio_ativo = 0
    y_l = y

    for j in range(0, len(y_l)):
        if y_l[j] > (1 * threshold):
            num_out = j
            cont_neuronio_ativo += 1
        if cont_neuronio_ativo > 1:
            num_out = np.nan
            break
    return num_out

def output_layer_activation(output_value,
                            num_classes,
                            activation_lower_value=-1.):
    d = np.ones(num_classes, dtype=np.float64) * \
        activation_lower_value

    d[output_value] = 1.
    return d

def load_neural_network(neural_network_xlsx):
    df = pd.read_excel(open(neural_network_xlsx, 'rb'),
                       sheet_name='weights')
    df2 = pd.read_excel(open(neural_network_xlsx, 'rb'),
                        sheet_name='params')

    L = int(df2['L'][0])
    m = list(df2['m'][0:L + 1])
    a = list(df2['a'][0:L])
    b = list(df2['b'][0:L])

    clf = MLPClassifier(
        hidden_layer_sizes=(tuple(m[1:-1])),
        activation= activation_function_name.TANH,
        learning_rate='invscaling',  # 'constant'
        solver=solver.BACKPROPAGATION,
        learning_rate_init=0.5,  # 0.001 para constant
        max_iter=10,
        shuffle=True,
        random_state=1,
        momentum=0.9,  # 0.01 para constant
        n_individuals=10,
        weight_limit=1,
        batch_size='auto',
        tol=0.01
    )

    clf.initialize_layers(m[0], m[-1])
    cont_neuron = 0

    # Carrega os pesos

    # faz o array para representar as colunas no excel, igual
    # ao que é feito ao salvar a rede

    arrays = np.zeros((2, np.sum(m[1:])))

    end_array = 0
    start_array = 0
    for l in range(0, L):

        if l == 0:
            start_array = 0
            end_array = start_array + m[l + 1]
        else:
            start_array += m[l]
            end_array += m[l + 1]

        arrays[0][start_array:end_array] = int(l + 1)
        arrays[1][start_array:end_array] = np.arange(0, m[l + 1])

    tuples = list(zip(*arrays))

    columns = pd.MultiIndex.from_tuples(tuples, names=['Layer:', 'Neuron:'])

    data = df.to_numpy()
    data = np.delete(data, 0, 1)
    data = data[2:]
    # print(f'{data[0]}')
    # print(f'{data[-1]}')
    # print(f'np.shape(data)={np.shape(data)}')
    df = pd.DataFrame(data=data, columns=columns)

    for l in range(0, L):
        # df[l + 1][0:self.m[l] + 1] = np.transpose(self.l[l].w)
        for j in range(0, m[l + 1]):
            # print(np.transpose(df.loc[l + 1][0:m[l] + 1]))
            # print(f'np.shape(np.transpose(df[l + 1][0:m[l] + 1]))={np.shape(np.transpose(df[l + 1][0:m[l] + 3]))}, np.shape(a1.l[l].w) = {np.shape(a1.l[l].w)}\n')
            clf.l[l].w = np.transpose(df[l + 1][0:m[l] + 1].to_numpy())

    clf.weights_initialized = True
    return clf

def load_scikit_model(model:skl):

    L = len(model.coefs_)
    m = [0] * (L+1)


    for i in range(1,L+1):
        m[i] = len(np.transpose(model.coefs_[i-1]))
    m[0] = len(np.transpose(model.coefs_[0])[0])
    # print(f'L:{L}, m:{m}')


    local_model = MLPClassifier(
        hidden_layer_sizes=tuple(m[1:-1]),
        activation=activation_function_name.TANH,
        learning_rate='invscaling',  # 'constant'
        solver=solver.BACKPROPAGATION,
        learning_rate_init=0.5,  # 0.001 para constant
        max_iter=10,
        shuffle=True,
        random_state=1,
        momentum=0.9,  # 0.01 para constant
        n_individuals=10,
        weight_limit=1,
        batch_size='auto',
        tol=0.01,
        activation_lower_value=0.
    )

    local_model.initialize_layers(m[0],m[-1])
    # print(f'local_model.hidden_layer_sizes={local_model.hidden_layer_sizes}')

    for l in range(0, L):

        for j in range(0, m[l + 1]):
            weights = np.transpose(model.coefs_[l])[j]
            bias = model.intercepts_[l][j]
            local_model.l[l].w[j][0:-1] = weights
            local_model.l[l].w[j][-1] = bias

    local_model.weights_initialized = True
    return local_model