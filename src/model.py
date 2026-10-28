import numpy as np
from src.preprocessing import fit_preprocess, apply_fitted_preprocess

def stratified_kfold_indices(y, k=5, seed=0):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for c in (0, 1):
        idx_c = rng.permutation(np.where(y == c)[0])
        for i, chunk in enumerate(np.array_split(idx_c, k)):
            folds[i].extend(chunk.tolist())
    return [np.array(sorted(f)) for f in folds]

def auc_score(y, s):
    ranks = np.empty(len(s))
    ranks[np.argsort(s)] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def f1_score(y_true, y_proba, threshold=0.5):
    # Predictions 0 or 1
    y_pred = (y_proba >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    denominator = 2 * tp + fp + fn

    if denominator == 0:
        return 0.0

    return 2 * tp / denominator

def f1_curve(y, p, ths=np.linspace(0.10, 0.50, 81)):
    P = y.sum()
    out = []
    for t in ths:
        pred = p >= t
        tp = np.sum(pred & (y == 1))
        out.append(2 * tp / (pred.sum() + P))
    return ths, np.array(out)

def avg_precision(y, p):
    o = np.argsort(-p)
    yy = y[o]
    prec = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    return np.sum(prec * yy) / yy.sum()

def report(y, p, nom=""):
    ths, f1 = f1_curve(y, p)
    sm = np.convolve(f1, np.ones(5) / 5, mode="same")
    k = 2 + np.argmax(sm[2:-2])
    print(f"{nom:12s} F1 max lisse = {sm[k]:.4f} au seuil {ths[k]:.3f} "
          f"| AP = {avg_precision(y, p):.4f}")
    return ths[k]

########################################### Log Reg ###################################################
def sigmoid(t):
    return 0.5 * (1.0 + np.tanh(0.5 * t)) # stable numériquement

def log_loss(X, y, w):
    z = X @ w
    return np.mean(np.logaddexp(0, z) - y * z) # cross-entropy stable

def log_loss_from_p(p, y):
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def score(oof, y):
    p = np.clip(oof, 1e-9, 1 - 1e-9)
    return auc_score(y, oof), log_loss_from_p(p, y)

### Gradient Descent ###
def train_logreg_GD(X, y, lambda_=1e-2, lr=0.2, n_iter=500, x_te = None, y_te = None, every=10):
    w = np.zeros(X.shape[1])
    n = len(y)
    hist = {"it": [], "train": [], "test": []}

    for t in range(n_iter+1):
        if t % every == 0:
            hist["it"].append(t)
            loss_tr = log_loss(X, y, w)
            hist["train"].append(loss_tr)
            if x_te is not None:
                loss_te = log_loss(x_te, y_te, w)
                hist["test"].append(loss_te)
        if t == n_iter:
            break

        grad = X.T @ (sigmoid(X @ w) - y) / n
        grad[1:] += lambda_ * w[1:] # no penalty on bias
        w -= lr * grad
    return w, hist

def train_logreg_mini_batch(
    X, y, lambda_=1e-2, lr=0.01, n_epoch=20,
    x_te=None, y_te=None, every=10,
    batch_size=256, random_state=None
):
    w = np.zeros(X.shape[1])
    n = len(y)
    hist = {"it": [], "train": [], "test": []}
    rng = np.random.default_rng(random_state)

    for t in range(n_epoch + 1):
        if t % every == 0:
            hist["it"].append(t)
            hist["train"].append(log_loss(X, y, w))
            if x_te is not None:
                hist["test"].append(log_loss(x_te, y_te, w))
        if t == n_epoch:
            break

        indices = rng.permutation(n)

        # Mini-batchs loop
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]

            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            m = len(y_batch)

            grad = X_batch.T @ (
                sigmoid(X_batch @ w) - y_batch
            ) / m

            # L2 regularization, no penalty on bias
            grad[1:] += lambda_ * w[1:]

            w -= lr * grad

    return w, hist

def cv_logreg_GD(x, y, lambda_, header, k=5, seed=0, n_iter=500, every=10, **kw):
    folds = stratified_kfold_indices(y, k, seed)
    hists = []
    oof = np.zeros(len(y))
    
    for i in range(k):
        va = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])

        # Preprocessing at fold level
        p = fit_preprocess(x[tr])
        x_tr, _ = apply_fitted_preprocess(x[tr], p, header)
        x_te, _ = apply_fitted_preprocess(x[va], p, header)
        y_tr = y[tr]
        y_te = y[va]

        # Training
        w, h = train_logreg_GD(x_tr, y_tr, lambda_, x_te=x_te, y_te=y_te, n_iter=n_iter, every=every, **kw)
        hists.append(h)
        print(f"Fold {i+1}/{k} | Tr loss: {h["train"][-1]:.3f} | Min Te loss (at {np.argmin(h["test"])*every}/{n_iter}): {min(h["test"]):.3f} | Final Te loss: {h["test"][-1]:.3f}")

        # Prediction
        y_pred_sig = sigmoid(x_te @ w)
        oof[va] = y_pred_sig

        # Validation
        # acc05 = ((y_pred_sig >= 0.5).astype(int) == y_te).mean()
        # th, acc_best = best_threshold(y_pred_sig, y_te)
        # print(f"lambda={lambda_:<6} acc@0.5={acc05:.4f} | meilleur seuil={th:.2f} acc={acc_best:.4f}")

    return oof, hists

def cv_logreg_mini_batch(
    x, y, lambda_, header, k=5, seed=0,
    n_epoch=500, every=10, batch_size=256, **kw
):
    folds = stratified_kfold_indices(y, k, seed)
    hists = []
    oof = np.zeros(len(y))

    for i in range(k):
        va = folds[i]
        tr = np.concatenate([
            folds[j] for j in range(k) if j != i
        ])

        # Preprocessing at fold level
        p = fit_preprocess(x[tr])
        x_tr, _ = apply_fitted_preprocess(x[tr], p, header)
        x_te, _ = apply_fitted_preprocess(x[va], p, header)

        y_tr = y[tr]
        y_te = y[va]

        # Mini-batch Gradient Descent Training
        w, h = train_logreg_mini_batch(
            x_tr, y_tr,
            lambda_=lambda_,
            x_te=x_te,
            y_te=y_te,
            n_epoch=n_epoch,
            every=every,
            batch_size=batch_size,
            random_state=seed + i,
            **kw
        )

        hists.append(h)

        best_idx = np.argmin(h["test"])
        best_it = h["it"][best_idx]

        print(
            f"Fold {i+1}/{k} | "
            f"Tr loss: {h['train'][-1]:.3f} | "
            f"Min Te loss (at {best_it}/{n_epoch}): "
            f"{h['test'][best_idx]:.3f} | "
            f"Final Te loss: {h['test'][-1]:.3f}"
        )

        # Out-of-fold predictions (probabilities)
        y_pred_sig = sigmoid(x_te @ w)
        oof[va] = y_pred_sig

    return oof, hists

### Newton ###
def pass_over_data(X, y, w, need_H=True, chunk=50000):
    """Only one pass over data for loss, gradient, hessian (no penalty)."""
    n, d = X.shape
    g = np.zeros(d)
    H = np.zeros((d, d)) if need_H else None
    loss = 0.0
    for a in range(0, n, chunk):
        Xc = np.asarray(X[a:a + chunk], dtype=np.float64)
        yc = y[a:a + chunk]
        z = Xc @ w
        p = sigmoid(z)
        loss += np.sum(np.logaddexp(0, z) - yc * z)
        g += Xc.T @ (p - yc)
        if need_H:
            H += Xc.T @ (Xc * (p * (1 - p))[:, None])
    return loss / n, g / n, (H / n if need_H else None)

def newton_fit(X, y, lam, w0=None, n_iter=8, tol=1e-5, hess_every=1):
    n, d = X.shape
    w = np.zeros(d) if w0 is None else w0.copy()
    H = None
    R = np.full(d, lam)

    R[0] = 0.0 # no penalty on bias
    
    for it in range(n_iter):
        need_H = H is None or it % hess_every == 0
        _, g, Hn = pass_over_data(X, y, w, need_H)
        if need_H:
            H = Hn
        step = np.linalg.solve(H + np.diag(R) + 1e-10 * np.eye(d), g + R * w)
        w = w - step
        if np.max(np.abs(step)) < tol:
            break
    return w

########################################### MLP #######################################################
def init_mlp(d, h=(128, 64), seed=0):
    """
    One hidden layer.
    """
    rng = np.random.default_rng(seed)
    sizes = [d, *h, 1]
    return [(rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32), np.zeros(b, np.float32))
            for a, b in zip(sizes[:-1], sizes[1:])]

def forward(P, X, drop=0.0, rng=None):
    acts = [X]
    for k, (W, b) in enumerate(P):
        z = acts[-1] @ W + b
        if k == len(P) - 1:
            acts.append(z)
        else:
            a = np.maximum(z, 0) # ReLU
            if drop > 0:
                a = a * (rng.random(a.shape) > drop) / (1 - drop)      # inversed dropout
            acts.append(a.astype(np.float32))
    return acts

def predict(P, X):
    return sigmoid(forward(P, X)[-1][:, 0]).astype(np.float64)

def train_mlp(Xtr, ytr, Xva=None, yva=None, h=(128, 64), epochs=30, bs=256, lr=1e-3,
              wd=1e-3, drop=0.2, pos_weight=1.0, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    P = init_mlp(Xtr.shape[1], h, seed)
    m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in P]
    v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in P]
    b1, b2, eps, t, n = 0.9, 0.999, 1e-8, 0, len(ytr)

    total = epochs * int(np.ceil(n / bs))
    best = (np.inf, None, 0)

    for ep in range(epochs):
        perm = rng.permutation(n)
        for a in range(0, n, bs):
            i = perm[a:a + bs]
            xb = Xtr[i]
            yb = ytr[i].astype(np.float32)[:, None]
            acts = forward(P, xb, drop, rng)
            wgt = np.where(yb == 1, pos_weight, 1.0).astype(np.float32)
            delta = wgt * (sigmoid(acts[-1]) - yb) / len(i)
            t += 1
            lr_t = lr * 0.5 * (1 + np.cos(np.pi * t / total))       # cosine decay
            for k in range(len(P) - 1, -1, -1):
                W, b = P[k]
                grads = (acts[k].T @ delta + wd * W, delta.sum(0))
                if k > 0:
                    delta = (delta @ W.T) * (acts[k] > 0) / (1 - drop)  # before W update
                for j, g in enumerate(grads):
                    m[k][j][...] = b1 * m[k][j] + (1 - b1) * g
                    v[k][j][...] = b2 * v[k][j] + (1 - b2) * g * g
                    P[k][j][...] -= lr_t * (m[k][j] / (1 - b1 ** t)) / (np.sqrt(v[k][j] / (1 - b2 ** t)) + eps)
        if Xva is not None:
            pv = predict(P, Xva)
            pc = np.clip(pv, 1e-9, 1 - 1e-9)
            ll = -np.mean(yva * np.log(pc) + (1 - yva) * np.log(1 - pc))
            if ll < best[0]:
                best = (ll, [(W.copy(), b.copy()) for W, b in P], ep + 1)
            if verbose:
                _, f1 = f1_curve(yva, pv)
                print(f"époque {ep + 1:2d} | log-loss val = {ll:.5f} | F1 max = {f1.max():.4f} "
                      f"| AP = {avg_precision(yva, pv):.4f}", flush=True)
    return (best[1] if best[1] is not None else P), best[2]