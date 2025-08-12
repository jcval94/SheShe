# examples/titanic_demo.py
# Requiere: pip install seaborn
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sheshe import ModalBoundaryClustering

def main():
    tit = sns.load_dataset("titanic")
    df = tit[["survived", "pclass", "age", "sibsp", "parch", "fare"]].dropna()
    y = df["survived"].astype(int).values
    X = df.drop(columns=["survived"]).values

    sh = ModalBoundaryClustering(
        base_estimator=LogisticRegression(max_iter=1000),
        task="classification", base_2d_rays=8, random_state=2
    ).fit(X, y)
    print("Accuracy Titanic:", accuracy_score(y, sh.predict(X)))
    print(sh.interpretability_summary(df.drop(columns=["survived"]).columns.tolist()).head())
    sh.plot_pairs(X, y, max_pairs=3)

if __name__ == "__main__":
    main()
