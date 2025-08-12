# examples/save_load_demo.py
"""Demostración de guardado y carga de ModalBoundaryClustering."""
from pathlib import Path
from sklearn.datasets import load_iris
from sheshe import ModalBoundaryClustering


def main() -> None:
    iris = load_iris()
    X, y = iris.data, iris.target
    sh = ModalBoundaryClustering(random_state=0).fit(X, y)
    model_path = Path(__file__).with_suffix(".joblib")
    sh.save(model_path)
    sh2 = ModalBoundaryClustering.load(model_path)
    print("Predicciones iguales:", (sh.predict(X[:5]) == sh2.predict(X[:5])).all())


if __name__ == "__main__":
    main()
