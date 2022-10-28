import pandas as pd
arquivo = pd.read_csv("wine_dataset.csv")
arquivo.head()
arquivo['style'] = arquivo['style'].replace('red',0)
arquivo['style'] = arquivo['style'].replace('white',1)
y=arquivo['style']
x=arquivo.drop('style', axis=1)
from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)
print(arquivo.shape, x_treino.shape, x_teste.shape, y_treino.shape, y_teste.shape)
from sklearn.ensemble import ExtraTreesClassifier
modelo = ExtraTreesClassifier()
modelo.fit(x_treino, y_treino)
resultado = modelo.score(x_teste, y_teste)
print("Acurácia:", resultado)
previsoes = modelo.predict(x_teste[400:405])
print(previsoes)
print(y_teste[400:405])