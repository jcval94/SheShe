# examples/demo_classification.py
from sklearn.datasets import load_iris, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sheshe import ModalBoundaryClustering

def main():
    print("=== Iris ===")
    iris = load_iris()
    X, y = iris.data, iris.target

    models = {
        "LogReg": LogisticRegression(max_iter=1000),
        "RF": RandomForestClassifier(n_estimators=250, random_state=0),
        "SVC": SVC(probability=True, gamma="scale"),
    }

    for name, est in models.items():
        print(f"\n-- {name} --")
        sh = ModalBoundaryClustering(
            base_estimator=est,
            task="classification",
            base_2d_rays=8,
            random_state=0,
            drop_fraction=0.5,
        ).fit(X, y)
        y_hat = sh.predict(X)
        print("Accuracy:", accuracy_score(y, y_hat))
        print(sh.interpretability_summary(iris.feature_names).head())
        sh.plot_pairs(X, y, max_pairs=3)

    print("\n=== Wine ===")
    wine = load_wine()
    Xw, yw = wine.data, wine.target
    sh = ModalBoundaryClustering(
        base_estimator=RandomForestClassifier(n_estimators=350, random_state=1),
        task="classification",
        base_2d_rays=8,
        random_state=1,
        drop_fraction=0.5,
    ).fit(Xw, yw)
    print("Accuracy Wine:", accuracy_score(yw, sh.predict(Xw)))
    print(sh.interpretability_summary(wine.feature_names).head())
    sh.plot_pairs(Xw, yw, max_pairs=3)

if __name__ == "__main__":
    main()
