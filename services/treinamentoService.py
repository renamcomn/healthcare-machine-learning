from services.vetorizacaoService import vetorizacao, encode_Y
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# buscar o modelo XGBoost
from xgboost import XGBClassifier



#Passo 1 - 
# Buscar os dados - X, Y

#Passo 2 - 
# Separar os dados em treino (75%) (X_treino, Y_treino) e teste (25%) (X_teste, Y_teste)


#Passo 3 - 
# Treinar o modelo com os dados de treino

#Passo 4 - 
# Avaliar o modelo com os dados de teste (Acurácia)

def buscar_dados():
    # Implementar a lógica para buscar os dados - X, Y
    X = vetorizacao()
    Y = encode_Y()
    return X, Y

def separar_dados():
    
    #buscando os dadps
    X, Y = buscar_dados()

    # Separar os dados em treino (70%) e teste (30%)
    X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.3, random_state=42)
    return X_treino, X_teste, Y_treino, Y_teste

def treinar_modelo():
    """
    Treina o modelo XGBoost e retorna o modelo treinado
    """
    X_treino, X_teste, Y_treino, Y_teste = separar_dados()

    model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=1,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_treino, Y_treino)

    return model


def acuracia_modelo():
    X_treino, X_teste, Y_treino, Y_teste = separar_dados()

    model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=1,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_treino, Y_treino)

    Y_pred = model.predict(X_teste)
    acuracia = accuracy_score(Y_teste, Y_pred)

    return acuracia * 100
