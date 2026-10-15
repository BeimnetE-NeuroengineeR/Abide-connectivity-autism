import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, roc_auc_score

N_SUBJECTS = None   # Full dataset run  

# --- 1. Load preprocessed ABIDE (CC200 atlas) ---
abide = datasets.fetch_abide_pcp(derivatives=["rois_cc200"], pipeline="cpac",
                                 band_pass_filtering=True, global_signal_regression=False,
                                 quality_checked=True, n_subjects=N_SUBJECTS)
pheno = pd.DataFrame(abide.phenotypic)
y = (pheno["DX_GROUP"].values == 1).astype(int)          # 1 = autism, 0 = control
sites = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in pheno["SITE_ID"]])
print("subjects:", len(y), "| autism:", y.sum(), "| sites:", len(set(sites)))

# --- 2. Functional connectivity -> one feature vector per person ---
conn = ConnectivityMeasure(kind="correlation", vectorize=True, discard_diagonal=True)
X = np.nan_to_num(conn.fit_transform(abide.rois_cc200))

clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.001, max_iter=2000))

# --- 3. Random 5-fold (can leak site/scanner identity) ---
proba = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          method="predict_proba")[:, 1]
rand_acc = ((proba > 0.5) == y).mean(); rand_auc = roc_auc_score(y, proba)

# --- 4. Leave-one-site-out (honest cross-site test) ---
rows = []
for tr, te in LeaveOneGroupOut().split(X, y, sites):
    clf.fit(X[tr], y[tr]); p = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y[te], p) if len(set(y[te])) > 1 else np.nan
    rows.append((sites[te][0], len(te), ((p > 0.5) == y[te]).mean(), auc))
df = pd.DataFrame(rows, columns=["site", "n", "accuracy", "auc"])
df.to_csv("reports/leave_site_out.csv", index=False)
print(df.round(3))
print(f"Random 5-fold: acc={rand_acc:.3f} auc={rand_auc:.3f}")
print(f"Leave-site-out (mean over sites): acc={df.accuracy.mean():.3f} auc={df.auc.mean():.3f}")

ax = df.plot(x="site", y="accuracy", kind="bar", figsize=(9, 4), legend=False)
ax.axhline(rand_acc, ls="--", c="red", label="random 5-fold")
ax.axhline(y.mean(), ls=":", c="gray", label="autism share")
plt.ylabel("Accuracy"); plt.title("Leave-one-site-out accuracy"); plt.legend()
plt.tight_layout(); plt.savefig("reports/site_accuracy.png", dpi=150)

# --- 5. Exploratory: are 2 autism subgroups stable across sites? ---
Xa, sa = StandardScaler().fit_transform(X[y == 1]), sites[y == 1]
def fit_km(Z):
    pca = PCA(10, random_state=0).fit(Z)
    km = KMeans(2, n_init=10, random_state=0).fit(pca.transform(Z))
    return pca, km

uniq = np.unique(sa)
if len(uniq) >= 4:
    rng, aris = np.random.default_rng(0), []
    for _ in range(30):
        half = rng.permutation(uniq)[: len(uniq) // 2]
        A, B = np.isin(sa, half), ~np.isin(sa, half)
        pA, kA = fit_km(Xa[A]); pB, kB = fit_km(Xa[B])
        aris.append(adjusted_rand_score(kA.predict(pA.transform(Xa[B])), kB.labels_))
    print(f"Subgroup stability across site halves (ARI, 0 = chance): mean={np.mean(aris):.3f}")
    pca, km = fit_km(Xa)
    chi_p = chi2_contingency(pd.crosstab(sa, km.labels_))[1]
    print(f"Do the 2 clusters just track site? chi-square p = {chi_p:.2e}")
else:
    print("Too few autism sites in this sample for the stability test (fine for a test run).")