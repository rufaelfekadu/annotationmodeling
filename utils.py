from itertools import combinations
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import pickle

dem_semisup = """
data {
    // observed data
    int<lower=0> NDATA;
    int<lower=0> NITEMS;
    int<lower=0> NUSERS;
    int<lower=1, upper=NITEMS> items[NDATA];
    int<lower=1, upper=NUSERS> u1s[NDATA];
    int<lower=1, upper=NUSERS> u2s[NDATA];
    int<lower=0, upper=NUSERS> n_gold_users; // first n_gold_users assumed to be gold
    real<lower=0> gold_uerr[NUSERS];
    real distances[NDATA];

    // hyperparameters
    real<lower=0> uerr_prior_scale;
    real<lower=0> diff_prior_scale;

    real<lower=0> uerr_prior_loc_scale;
    real<lower=0> diff_prior_loc_scale;
}
parameters {
    real<lower=0> uerr_prior_loc;
    // real<lower=0> diff_prior_loc;
    // real<lower=0> uerr_prior_scale;
    // real<lower=0> diff_prior_scale;
    vector<lower=0>[NUSERS] uerr;
    vector<lower=0>[NITEMS] diff;
}
transformed parameters {
    matrix[NITEMS, NUSERS] label_logprobs = rep_matrix(0, NITEMS, NUSERS);
    matrix[NITEMS, NUSERS] label_probabilities = rep_matrix(0, NITEMS, NUSERS);
    vector[NITEMS] item_logprobs = rep_vector(0, NITEMS);
    vector<lower=0>[NUSERS] uerr_g = uerr;
    if (n_gold_users > 0) {
        for (u in 1:NUSERS) {
            uerr_g[u] = gold_uerr[u];
        }
    }

    for (n in 1:NDATA) {
        int i = items[n];
        int u1 = u1s[n];
        int u2 = u2s[n];
        real dist = distances[n];
        real scale = (uerr_g[u1] + uerr_g[u2]) * diff[i] + 0.01;

        label_logprobs[i, u1] += normal_lpdf(dist | 0, scale);
        label_logprobs[i, u2] += normal_lpdf(dist | 0, scale);
    }
    {
        int n_itemlabels[NITEMS];
        int active_users[NITEMS, NUSERS];
        for (i in 1:NITEMS) {
            int nlabels = 0;
            for (u in 1:NUSERS) {
                if (label_logprobs[i, u] != 0) {
                    nlabels += 1;
                    active_users[i, nlabels] = u;
                }
            }
            n_itemlabels[i] = nlabels;
            if (nlabels > 0) {
                row_vector[nlabels] label_logprobs_v = label_logprobs[i, active_users[i, 1:nlabels]];
                vector[nlabels] probs = softmax(label_logprobs_v');
                for (n in 1:nlabels) {
                    int u = active_users[i, n];
                    label_probabilities[i, u] = probs[n];
                }
                item_logprobs[i] = log_sum_exp(label_logprobs_v);
            }
        }
    }
}
model {
    for (u in 1:NUSERS) {
        uerr[u] ~ normal(uerr_prior_loc, uerr_prior_scale);
    }
    // uerr ~ normal(uerr_prior_loc, uerr_prior_scale);
    // diff ~ normal(diff_prior_loc, diff_prior_scale);
    // uerr_prior_loc ~ normal(0, uerr_prior_loc_scale);
    // diff_prior_loc ~ normal(0, diff_prior_loc_scale);

    for (i in 1:NITEMS) {
        target += item_logprobs[i];
    }
}
generated quantities {
    matrix<lower=0>[NITEMS, NUSERS] dist_from_truth = 1 - label_probabilities;
}
"""

mas2_semisup = """
functions {
    // no built-in vector norm in Stan?
    real norm(vector x) {
        return sqrt(sum(square(x)));
    }
}

data {
    # observed data
    int<lower=0> NDATA;
    int<lower=0> NITEMS;
    int<lower=0> NUSERS;
    int<lower=1, upper=NITEMS> items[NDATA];
    int<lower=1, upper=NUSERS> u1s[NDATA];
    int<lower=1, upper=NUSERS> u2s[NDATA];
    int<lower=0, upper=NUSERS> n_gold_users; // first n_gold_users assumed to be gold
    real<lower=0> gold_uerr[NUSERS];
    real distances[NDATA];

    # hyperparameters
    int<lower=1> DIM_SIZE;
    int<lower=0> eps_limit;
    real<lower=0> uerr_prior_scale;
    real<lower=0> diff_prior_scale;
}
parameters {
    vector<lower=0>[NUSERS] uerr;
    vector<lower=0>[NITEMS] diff;
    real<lower=0> sigma;
    real<lower=0> sigma2;

    vector<lower=-eps_limit, upper=eps_limit>[DIM_SIZE] item_user_errors_Z[NITEMS, NUSERS];
}
transformed parameters {
    real pred_distances[NDATA];
    matrix<lower=0>[NITEMS, NUSERS] dist_from_truth = rep_matrix(666, NITEMS, NUSERS);
    vector<lower=0>[NUSERS] uerr_g = uerr;
    if (n_gold_users > 0) {
        for (u in 1:NUSERS) {
            uerr_g[u] = gold_uerr[u];
        }
    }

    for (n in 1:NDATA) {
        int u1 = u1s[n];
        int u2 = u2s[n];
        int item = items[n];
        
        vector[DIM_SIZE] iueZ1 = item_user_errors_Z[item, u1];
        vector[DIM_SIZE] iueZ2 = item_user_errors_Z[item, u2];

        dist_from_truth[item, u1] = norm(iueZ1);
        dist_from_truth[item, u2] = norm(iueZ2);

        pred_distances[n] = norm(iueZ1 - iueZ2);
    }
}
model {
    sigma ~ exponential(1);
    uerr ~ normal(1, uerr_prior_scale);
    diff ~ normal(1, diff_prior_scale);

    for (i in 1:NITEMS) {
        for (u in 1:NUSERS) {
            if (dist_from_truth[i, u] != 666) {
                item_user_errors_Z[i, u] ~ normal(0, diff[i] * uerr_g[u]);
            }
        }
    }

    // likelihood
    distances ~ normal(pred_distances, sigma);
}
generated quantities {
    vector[DIM_SIZE] item_user_errors[NITEMS, NUSERS] = item_user_errors_Z;
}

"""

masX = """
functions {
    // no built-in vector norm in Stan?
    real norm(vector x) {
        return sqrt(sum(square(x)));
    }
}

data {
    # observed data
    int<lower=0> NDATA;
    int<lower=0> NITEMS;
    int<lower=0> NUSERS;
    int<lower=1> DIM_SIZE;

    int<lower=0> item_users[NITEMS, NUSERS];
    vector[DIM_SIZE] embeddings[NITEMS, NUSERS];
    int<lower=0, upper=NUSERS> n_gold_users; // first n_gold_users assumed to be gold
    real<lower=0> gold_uerr[NUSERS];

    # hyperparameters
    real<lower=0> uerr_prior_scale;
    real<lower=0> uerr_prior_loc_scale;
    real<lower=0> diff_prior_scale;
}
parameters {
    real<lower=0> uerr_center;
    vector<lower=0>[NUSERS] uerr;
    vector<lower=0>[NITEMS] diff;
    vector[DIM_SIZE] center[NITEMS];
}
transformed parameters {
    vector<lower=0>[NUSERS] uerr_g = uerr;
    if (n_gold_users > 0) {
        for (u in 1:NUSERS) {
            uerr_g[u] = gold_uerr[u];
        }
    }
}
model {
    uerr_center ~ normal(0, uerr_prior_loc_scale);
    diff ~ normal(0, diff_prior_scale);
    uerr ~ normal(uerr_center, uerr_prior_scale);
    // uerr ~ gamma(2, uerr_prior_scale);
    // diff ~ gamma(2, diff_prior_scale);

    for (i in 1:NITEMS) {
        for (u in 1:NUSERS) {
            real dist_from_center;
            int uid = item_users[i, u];
            if (uid == 0) break;
            dist_from_center = norm(embeddings[i, u] - center[i]);
            dist_from_center ~ normal(0, diff[i] + uerr_g[uid]);
        }
    }
}
generated quantities {
    vector[DIM_SIZE] item_user_errors[NITEMS, NUSERS] = embeddings;
    matrix<lower=0>[NITEMS, NUSERS] dist_from_truth = rep_matrix(666, NITEMS, NUSERS);

    for (i in 1:NITEMS) {
        for (u in 1:NUSERS) {
            real dist_from_center;
            int uid = item_users[i, u];
            if (uid == 0) break;
            dist_from_center = norm(embeddings[i, u] - center[i]);
            dist_from_truth[i, uid] = dist_from_center;
            // item_user_errors[i, uid] = embeddings[i, u] - center[i];
        }
    }
}
"""

def stanmodel(modelname, overwrite):
    import pystan
    picklefile = modelname + ".pkl"
    try:
        if overwrite:
            raise IOError("Overwriting picked files")
        stan_model = pickle.load(open(picklefile, 'rb'))
        print("Pickled model loaded")
    except (OSError, IOError):
        print("Pickled model not found")
        print("Compiling model")
        # program_code = globals()[modelname]
        print(modelname)
        stan_model = pystan.StanModel(file=modelname+'.stan')
        with open(picklefile, 'wb') as f:
            print("Pickling model")
            pickle.dump(stan_model, f)
    return stan_model

def make_categorical(df, colname, overwrite=True):
    orig = list(df[colname].values)
    if overwrite:
        df[colname] = pd.Categorical(df[colname]).codes
    return dict(zip(orig, df[colname]))

def flatten(listoflists):
    return [item for sublist in listoflists for item in sublist]

def translate_categorical(df, colname, coldict, drop_missing=True):
    df[colname] = np.array([coldict.get(i) for i in df[colname].dropna().values])
    result = df[df[colname] >= 0].copy()
    result[colname] = result[colname].astype(int)
    return result

def groups_of(df, colname, colvals=None):
    if colvals is None:
        colvals = df[colname].unique()
    gdf = df.groupby(colname)
    for colval in colvals:
        yield colval, gdf.get_group(colval)

# def calc_distances_foritem(idf, compare_fn, label_colname, item_colname, uid_colname):
#     users = idf[uid_colname].unique()
#     items = []
#     u1s = []
#     u2s = []
#     distances = []
#     for u1, u2 in combinations(users, 2):
#         p1 = idf[idf[uid_colname]==u1][label_colname].values[0]
#         p2 = idf[idf[uid_colname]==u2][label_colname].values[0]
#         distance = compare_fn(p1, p2)
#         items.append(idf[item_colname].values[0])
#         u1s.append(u1)
#         u2s.append(u2)
#         distances.append(distance)
#     distances /= 2
#     distances = np.array(distances) + (.1 - np.min(distances))
#     return {
#         "items":np.array(items) + 1,
#         "u1s":np.array(u1s) + 1,
#         "u2s":np.array(u2s) + 1,
#         "distances":distances
#     }
    
# def calc_distances_parallel(df, compare_fn, label_colname, item_colname, uid_colname="uid"):
#     items = df[item_colname].unique()
#     args = tuple([(df[df[item_colname]==i], compare_fn, label_colname, item_colname, uid_colname) for i in items])
#     with Pool() as p:
#         r = list(p.starmap(calc_distances_foritem, args))
#         return pd.concat([pd.DataFrame(d) for d in r]).to_dict(orient="list")

def calc_distances(df, compare_fn, label_colname, item_colname, uid_colname="uid", bound=True):
    items = []
    u1s = []
    u2s = []
    a1s = []
    a2s = []
    distances = []
    n_labels = 0
    for item in tqdm(sorted(df[item_colname].unique())):
        idf = df[df[item_colname] == item]
        users = idf[uid_colname].unique()
        u_i = range(len(users))
        user_i_lookup = dict(zip(users, u_i))
        for u1, u2 in combinations(users, 2):
            p1 = idf[idf[uid_colname]==u1][label_colname].values[0]
            p2 = idf[idf[uid_colname]==u2][label_colname].values[0]
            distance = compare_fn(p1, p2)
            if np.isnan(distance):
                print(f"WARNING: NAN DISTANCE BETWEEN {p1} and {p2}")
            items.append(item)
            u1s.append(u1)
            u2s.append(u2)
            distances.append(distance)
            a1s.append(user_i_lookup.get(u1) + n_labels)
            a2s.append(user_i_lookup.get(u2) + n_labels)
        if len(users) > 1: # labels not used when no redundancy
            n_labels += len(users)
    if bound:
        distances = np.array(distances) + (.1 - np.min(distances))
        # distances = (np.array(distances) + 0.01 - np.min(distances)) / (np.max(distances) + 0.02 - np.min(distances))
    numnans = np.sum(np.isnan(distances))
    if numnans > 0:
        print(f"WARNING! FOUND {numnans} NAN DISTANCES!!")
    stan_data = {
        "items":np.array(items) + 1,
        "u1s":np.array(u1s) + 1,
        "u2s":np.array(u2s) + 1,
        "a1s":np.array(a1s) + 1,
        "a2s":np.array(a2s) + 1,
        "distances":distances,
    }
    stan_data["NDATA"] = len(stan_data["distances"])
    stan_data["NITEMS"] = np.max(np.unique(stan_data["items"]))
    stan_data["NUSERS"] = len(df[uid_colname].unique())
    stan_data["NLABELS"] = n_labels
    stan_data["n_gold_users"] = 0
    stan_data["gold_user_err"] = 0

    sdf = pd.DataFrame(stan_data)
    all_as = set(sdf["a1s"]).union(set(sdf["a2s"]))
    expected = set(range(1, sdf["NLABELS"].unique()[0]))
    empty = expected - all_as
    assert len(empty) == 0, F"we found some missing labels: {sorted(list(empty))}"

    return stan_data

def nancorr(v1, v2):
    return pd.DataFrame({"v1":v1, "v2":v2}).corr().values[0,1]

def rotate_via_numpy(xy, radians):
    """Use numpy to build a rotation matrix and take the dot product."""
    x, y = xy
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, [x, y])
    return np.array(m.T)

def bounded_cauchy(scale, shape, abs_bound):
    return np.maximum(np.minimum(np.random.standard_cauchy(shape) * scale, abs_bound), -abs_bound)

def proper_score(model_scores, gold_scores, score_fn=np.square):
    map_ps = model_scores / np.sum(model_scores)
    max_i = np.argmax(gold_scores)
    map_r = 1 - map_ps
    map_r[max_i] = map_ps[max_i]
    return np.mean(score_fn(map_r))
