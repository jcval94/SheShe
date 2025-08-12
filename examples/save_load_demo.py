# examples/save_load_demo.py
from pathlib import Path
from sklearn.datasets import load_iris
from sheshe import ModalBoundaryClustering


def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    model = ModalBoundaryClustering(random_state=0).fit(X, y)
    path = Path("sheshe_model.joblib")
    model.save(path)
    loaded = ModalBoundaryClustering.load(path)
    print("Predicciones iguales:", (model.predict(X) == loaded.predict(X)).all())


if __name__ == "__main__":
    main()
