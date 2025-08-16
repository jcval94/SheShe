# examples/demo_regression.py
from sklearn.datasets import load_diabetes, make_friedman1
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sheshe import ModalBoundaryClustering

def main():
    print("=== Diabetes ===")
    diab = load_diabetes()
    Xd, yd = diab.data, diab.target

    models = {
        "RFReg": RandomForestRegressor(n_estimators=350, random_state=0),
        "GBR": GradientBoostingRegressor(random_state=0),
    }
    for name, est in models.items():
        print(f"\n-- {name} --")
        sh = ModalBoundaryClustering(
            base_estimator=est,
            task="regression",
            base_2d_rays=8,
            random_state=0,
            drop_fraction=0.5,
        ).fit(Xd, yd)
        seg = sh.predict(Xd).mean()
        yhat = sh.pipeline_.predict(Xd)
        print(f"Proporción en zona alta: {seg:.3f} | R2 base: {r2_score(yd, yhat):.3f}")
        print(sh.interpretability_summary(diab.feature_names).head())
        sh.plot_pairs(Xd, max_pairs=3)

    print("\n=== Friedman1 ===")
    Xf, yf = make_friedman1(n_samples=800, n_features=8, noise=0.5, random_state=0)
    est = GradientBoostingRegressor(random_state=0)
    sh = ModalBoundaryClustering(
        est,
        task="regression",
        base_2d_rays=8,
        random_state=1,
        drop_fraction=0.5,
    ).fit(Xf, yf)
    print("Zona alta (Friedman1):", sh.predict(Xf).mean())
    sh.plot_pairs(Xf, max_pairs=3)

if __name__ == "__main__":
    main()
