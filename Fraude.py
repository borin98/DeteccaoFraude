import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom as MS
from pylab import plot, show, savefig, pcolor, colorbar

"""

Função que cria o dataset todo processado

    :return -> modelo treinado e valores X, Y, normalizador e o df

"""

def preProcessing() :

    df = pd.read_csv("credit-data.csv")

    # apagando os dados Nan
    df = df.dropna()

    # retirando a idade negativa e coloando a média da idade no lugar
    df.loc[df.age < 0, "age"] = df.age.mean()

    # criando os dados de entrada e predição
    X = df.iloc[:, 0:4].values
    Y = df.iloc[:, 4].values

    # normalização dos dados
    normalizador = MinMaxScaler(feature_range=(0,1))
    X = normalizador.fit_transform(X)

    # criando o Ms
    som = MS(x = 15, y = 15, input_len = 4, random_seed = 0)
    som.random_weights_init(X)
    som.train_random(data = X, num_iteration = 100)

    return som, X, Y, normalizador, df

"""

Vizualização dos dados

"""
def vizuDados(som, X, Y) :

    markers = ["o", "o"]
    colors = ["r", "b"]

    for i, j in enumerate(X) :

        w = som.winner(j)

        plot(w[0] + 0.5, w[1] + 0.5, markers[Y[i]],
             markerfacecolor = "None", markersize = 10,
             markeredgecolor = colors[Y[i]], markeredgewidth = 2)

    show()
    #savefig("grafico.pdf")

"""

Busca das pessoas que podem gerar fráude

    :return -> lista de pessoas que podem gerar fráude

"""
def mapFraude(som, X, normalizador, df):

    mapeamento = som.win_map(X)

    # vetor dos suspeitos
    suspeitos = np.concatenate( (mapeamento[(3,9)], mapeamento[(1,12)]), axis = 0)
    suspeitos = np.concatenate( (suspeitos, mapeamento[(12,9)]), axis = 0)

    # fazendo a normalização inversa dos dados
    suspeitos = normalizador.inverse_transform(suspeitos)

    # separando em classes de fraude e não fraude
    classes = []
    for i in range(len(df)) :

        for j in range(len(suspeitos)) :

            # caso em que o crédito da pessoa não foi aprovado em ambos os casos
            if (df.iloc[i, 0] == int(round(suspeitos[j, 0]))) :

                classes.append(df.iloc[i,4])

    classes = np.asanyarray(classes)

    # concatenando as colunas
    suspeitoFinal = np.column_stack((suspeitos,classes))

    # ordenação dos valores na lista
    suspeitoFinal = suspeitoFinal[suspeitoFinal[:,4]].argsort()

    return suspeitoFinal

def main() :

    som, X, Y, normalizador, df = preProcessing()

    # vizualizando a distância
    pcolor(som.distance_map().T)
    colorbar()
    show()
    #savefig("distance.pdf")

    vizuDados(som = som, X = X, Y = Y)

    suspeitos = mapFraude(som = som, X = X, normalizador = normalizador, df = df)

if __name__ == '__main__':
    main()