import numpy as np

############################### Initial preprocessing steps #####################################
def preprocess_initial(data, header):
    to_drop = {"CCLGHOUS", "COLGHOUS",  # ~100% NaN
               "STATERES", "CTELNUM1", "CELLFON2"}  # same as CTELENUM
    x = data.copy()
    j = list(header).index("CTELENUM")
    x[:, j] = (~np.isnan(x[:, j])).astype(float) # CTELENUM becomes 0/1 indicator
    keep = np.array([n not in to_drop for n in header])

    return x[:, keep], header[keep]

############################### Special Encoding Correction ##########################################

# Replace 7s and 9s with NaN
BINARY_7_9 = ["TOLDHI2", "CVDSTRK3", "CHCSCNCR", "CHCOCNCR", "CHCCOPD1", "HAVARTH3",
             "ADDEPEV2", "CHCKIDNY", "ASTHMA3", "BLOODCHO", "HLTHPLN1", "MEDCOST",
             "PNEUVAC3", "SMOKE100", "QLACTLM2", "LMTJOIN3", "EXERANY2", "BPMEDS",
             "PERSDOC2", "INTERNET", "ASTHNOW", "USEEQUIP", "BLIND",
             "DECIDE", "DIFFWALK", "DIFFDRES", "DIFFALON", "SMOKDAY2", "STOPSMK2",
             "USENOW3", "ARTHDIS2", "ARTHSOCL", "WTCHSALT", "DRADVISE", "ARTTODAY",
             "ARTHWGT", "ARTHEXER", "ARTHEDU", "FLUSHOT6", "PDIABTST", "PREDIAB1",
             "INSULIN", "EYEEXAM", "DIABEYE", "DIABEDU", "SHINGLE2", "HADMAM",
             "HADPAP2", "HADHYST2", "BLDSTOOL", "HADSIGM3", "HADSGCO1", "CAREGIV1",
             "CRGVLNG1", "CRGVHRS1", "CRGVMST2", "CRGVEXPT", "CIMEMLOS", "EMTSUPRT",
             "LSATISFY", "MISTMNT", "SXORIENT", "TRNSGNDR", "RCSGENDR", "RCSRLTN2",
             "CASTHDX2", "CHECKUP1", "RENTHOM1", "NUMHHOL2", "CPDEMO1", "VETERAN3",
             "PREGNANT", "HIVTST6", "TETANUS", "HPVTEST", "DRNKANY5",
             "GENHLTH", "BPHIGH4", "CHOLCHK", "DIABETE3", "EDUCA", "MARITAL", "LANDLINE"]

# Replace 9s with NaN
VAR_9 = ["EMPLOY1", "_RACEG21", "_RACEGR3", "HOWLONG", "LASTPAP2", "LSTBLDS3", "LASTSIG3",
         "SCNTMNY1", "SCNTMEL1", "ADANXEV", "_CHISPNC", "_RFHLTH", "_HCVU651", "_RFHYPE5",
         "_CHOLCHK", "_RFCHOL", "_LTASTH1", "_CASTHM1", "_ASTHMS1", "_HISPANC", "_RACE",
         "_RFBMI5", "_CHLDCNT", "_EDUCAG", "_INCOMG", "_SMOKER3", "_RFSMOK3", "_RFBING5",
         "_RFDRHV5", "_FRTLT1", "_VEGLT1", "_TOTINDA", "PAMISS1_", "_PACAT1", "_PAINDX1",
         "_PA150R2", "_PA300R2", "_PA30021", "_PASTRNG", "_PAREC1", "_PASTAE1", "_LMTACT1",
         "_LMTWRK1", "_LMTSCL1", "_RFSEAT2", "_RFSEAT3", "_FLSHOT6", "_PNEUMO2", "_AIDTST3"]

# Replace 77s and 99s with NaN
VAR_77_99 = ["INCOME2", "LASTSMK2", "WHRTST10", "IMFVPLAC", "_CRACE1", "_MRACE1", "AVEDRNK2",
             "MAXDRNKS", "JOINPAIN", "HHADULT", "CRGVPRB1"]
# Replace 88s, 77s and 99s with NaN
VAR_88_77_99      = ["CHILDREN", "PHYSHLTH", "MENTHLTH", "POORHLTH", "ADDOWN", "ADTHINK", "ADMOVE", "DRNK3GE5"]
# Replace the list with NaN
SPECIAL_OTHER = {"_AGEG5YR": [14], "SEATBELT" : [7, 8, 9], "DROCDY3_": [900], "_DRNKWEK": [99900],
                 "MAXVO2_": [999], "FC60_": [999]}

# Drop this feature
DROP = ["_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_CLLCPWT", "_DUALUSE", "_DUALCOR",
        "_CPRACE", "_PRACE1", "_LLCPWT", "WEIGHT2", "HEIGHT3", "ALCDAY5", "FRUITJU1",
        "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1", "EXRACT11", "EXRACT21",
        "EXEROFT1", "EXEROFT2", "EXERHMM1", "EXERHMM2", "STRENGTH", "FLSHTMY2", "HIVTSTD3",
        "FMONTH", "IMONTH", "IDATE", "IDAY", "IYEAR", "SEQNO", "_PSU"]

def remap_special(x, names):
    """
    Assigns NaNs where needed.
    """
    x = x.copy()
    names = list(names)

    for n in BINARY_7_9:
        if n in names:
            j = names.index(n)
            x[np.isin(x[:, j], [7, 9]), j] = np.nan

    for n in VAR_9:
        if n in names:
            j = names.index(n)
            x[x[:, j] == 9, j] = np.nan

    for n in VAR_77_99:
        if n in names:
            j = names.index(n)
            x[np.isin(x[:, j], [77, 99]), j] = np.nan

    for n in VAR_88_77_99:
        if n in names:
            j = names.index(n)
            x[x[:, j] == 88, j] = 0
            x[np.isin(x[:, j], [77, 99]), j] = np.nan

    for n, codes in SPECIAL_OTHER.items():
        if n in names:
            j = names.index(n); x[np.isin(x[:, j], codes), j] = np.nan

    return x

def custom_encode(x, names):
    """
    Additional corrections
    """
    x = x.copy()
    names = list(names)
    for n in ["CHECKUP1", "EYEEXAM"]:
        if n in names:
            j = names.index(n)
            x[x[:, j] == 8, j] = 5
    return x, np.array(names)

def drop_by_name(x, names):
    keep = np.array([n not in DROP for n in names])
    return x[:, keep], np.array(names)[keep]

############################### Convertion to Log Scale ##########################################
def skewness(v):
    s = v.std()
    return 0.0 if s == 0 else np.mean(((v - v.mean()) / s) ** 3)

def pick_log_cols(x, names, min_unique=13, skew_thresh=1.8):
    """
    Identifies the features that need a log feature, based on a skewness threshold and if the data is not categorical.
    """
    out = []
    for j, n in enumerate(names):
        v = x[:, j]
        v = v[~np.isnan(v)]
        if len(v) < 200 or v.min() < 0 or len(np.unique(v)) < min_unique:
            continue
        sk = skewness(v)
        if sk > skew_thresh:
            out.append((n, j, sk))
    return out

def get_log_names(x, names):
    cands = pick_log_cols(x, names)
    log_js = [j for _, j, _ in cands]
    log_names = [names[j] for j in log_js]
    return log_names

def apply_log(x, names, js, replace=True):
    """
    Compute log features and add them.
    """
    x = x.copy()
    logs = np.log1p(x[:, js])
    if replace:
        x[:, js] = logs
        return x, np.array(names)
    return np.hstack([x, logs]), np.array(list(names) + ["log_" + names[j] for j in js])

############################### Interactions and One-Hot #########################################
NOMINAL = ["_STATE", "MARITAL", "EMPLOY1", "_RACE", "RENTHOM1",
           "_SMOKER3", "DIABETE3", "BPHIGH4", "MSCODE", "QSTVER"]

def fit_onehot(x, names, min_count=500):
    """
    For predefined categorical features, identifies what special codes need a custom indicator feature.
    Also adds a NaN indicator feature if relevant (min_count reached).
    """
    names = list(names)
    cats = {}
    for n in NOMINAL:
        if n not in names:
            continue
        v = x[:, names.index(n)]
        vals, cnt = np.unique(v[~np.isnan(v)], return_counts=True)
        cats[n] = {"vals": [float(a) for a, b in zip(vals, cnt) if b >= min_count],
                   "nan": bool(np.isnan(v).sum() >= min_count)}
    return cats

def apply_onehot(x, names, cats):
    names = list(names)
    cols, new_names, drop = [], [], []
    for n, info in cats.items():
        j = names.index(n)
        v = x[:, j]
        drop.append(j)
        for c in info["vals"]:
            cols.append((v == c).astype(np.float32))
            new_names.append(f"{n}={c:g}")
        if info["nan"]:
            cols.append(np.isnan(v).astype(np.float32))
            new_names.append(f"{n}=NaN")
    keep = [k for k in range(x.shape[1]) if k not in set(drop)]
    return (np.hstack([x[:, keep], np.column_stack(cols)]),
            np.array([names[k] for k in keep] + new_names))

def add_interactions(x, names):
    """
    Adds interactions between _AGE80 and other features.
    Adds interactions between FEM and other features.
    """
    names = list(names)
    col = lambda n: x[:, names.index(n)]
    age = col("_AGE80")
    age = np.where(np.isnan(age), 55, age)
    age_c = (age - 55) / 20
    flags = {"HTN": col("BPHIGH4") == 1, "CHOL": col("TOLDHI2") == 1, "DIAB": col("DIABETE3") == 1,
             "STRK": col("CVDSTRK3") == 1, "FEM": col("SEX") == 2, "SMOK": col("SMOKE100") == 1,
             "SANTE": col("GENHLTH") >= 4}
    new, nn = [age_c ** 2], ["age2"]
    for k, f in flags.items():
        new.append(age_c * f)
        nn.append(f"age_x_{k}")
    for k in ("HTN", "CHOL", "DIAB", "SMOK"):
        new.append(flags["FEM"] * flags[k]); nn.append(f"FEM_x_{k}")
    return np.hstack([x, np.column_stack(new).astype(float)]), np.array(names + nn)

###################################### Helper ####################################################
def prepare(x_raw, header, log_names=None, onehot_cols=None):
    """
    Apply all preprocessing steps to raw data.
    Call first on train data to get fitted log_names and onehot_cols parameters.
    """
    x, header = preprocess_initial(x_raw, header)

    # Special Encoding
    x = remap_special(x, header)
    x, header = custom_encode(x, header)
    x, header = drop_by_name(x, header)

    # Log scale
    if(log_names is None):
            log_names = get_log_names(x, header)
    if len(log_names):
        js = [list(header).index(n) for n in log_names]
        x, header = apply_log(x, header, js, replace=True)

    # Interactions & One-Hot
    if(onehot_cols is None):
        onehot_cols = fit_onehot(x, header)

    x, header = add_interactions(x, header)

    if(len(onehot_cols)):
        x, header = apply_onehot(x, header, onehot_cols)

    return x, np.array(header), log_names, onehot_cols

#################### Fold Specific Fitted preprocessing (indicators, normalization) ##############
def fit_preprocess(train_data, cat_thresh=12, hi_nan=0.95, min_ind=500):
    x = train_data
    n, d = x.shape
    nan_m = np.isnan(x)
    nf = nan_m.mean(axis=0)

    # Replace NaNs: median for continuous feature, mode for categorical feature
    fill = np.zeros(d)
    for j in range(d):
        v = x[~nan_m[:, j], j]
        if len(v) == 0:
            continue
        u, c = np.unique(v, return_counts=True)
        fill[j] = u[c.argmax()] if len(u) <= cat_thresh else np.median(v)

    # Drop features with too many NaNs
    val_cols = np.where(nf <= hi_nan)[0]

    # Indicators: remove repeated indicators (and complementary ones)
    seen, ind_cols = set(), []
    for j in np.where(nf > 0)[0]:
        k = nan_m[:, j].sum()
        if min(k, n - k) < min_ind:
            continue
        m = nan_m[:, j]
        if m[0]:
            m = ~m                       # canonical form (A and ~A = same knowledge)
        key = np.packbits(m).tobytes()
        if key in seen:
            continue
        seen.add(key)
        ind_cols.append(j)
    ind_cols = np.array(ind_cols, dtype=int)

    # Standardization statistics
    raw = _build_raw(x, fill, val_cols, ind_cols)
    mu, sd = raw.mean(axis=0), raw.std(axis=0)
    sd[sd == 0] = 1.0

    return {"fill": fill, "val_cols": val_cols, "ind_cols": ind_cols,
            "mu": mu, "sd": sd, "n_val": len(val_cols)}

def _build_raw(x, fill, val_cols, ind_cols):
    x_imp = np.where(np.isnan(x), fill, x)[:, val_cols]
    ind = np.isnan(x[:, ind_cols]).astype(float)
    return np.hstack([x_imp, ind])

def apply_fitted_preprocess(x, p, header, clip=5.0):
    raw = _build_raw(x, p["fill"], p["val_cols"], p["ind_cols"])
    z = (raw - p["mu"]) / p["sd"]
    z[:, :p["n_val"]] = np.clip(z[:, :p["n_val"]], -clip, clip)   # for values only, clip it
    new_header = (["bias"]
               + list(header[p["val_cols"]])
               + ["NaN_ind_" + n for n in header[p["ind_cols"]]])
    return np.hstack([np.ones((len(x), 1)), z]), new_header # had ones for bias

