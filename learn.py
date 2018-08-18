# General imports
import sys
import os
import shutil
import math
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
from pytz import utc
#import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Keras imports
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.utils.generic_utils import get_custom_objects
import keras.backend as K

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

# Custom imports
import load_model_config
import keras_helpers
import define_model

def get_timestamp():
    return datetime.utcnow().replace(tzinfo = utc).strftime("%Y-%m-%dT%H,%M,%S.%f")[:(2 - 6)] + "Z"

keras_helpers.define_custom_activations()

config = load_model_config.Config()

np.random.seed(config.rand_seed)

# Load data
client = MongoClient("mongodb://manager:toric@129.10.135.170:27017/MLEARN")
MLEARN = client.MLEARN
OVERCOUNT = MLEARN.OVERCOUNT

curs = OVERCOUNT.find({"NFSRT": {"$exists": True}}, {"_id": 0, "NVERTS": 0, "LNNFRTPREDICTSUM": 0}).hint([("H11", 1)])
raw_df = pd.DataFrame(list(map(keras_helpers.flatten_doc, curs))).set_index("POLYID").astype(float).sample(frac = 1, random_state = config.rand_seed)

client.close()

# raw_df = raw_df.drop(columns = [x for x in raw_df.columns if "STDEV" in x])
# y_df = pd.DataFrame((X_df["NFSRTPREDICT"] - X_df["NFSRT"]).rename("NFSRTDIFF"), index = X_df.index)
y_df = raw_df[["LNNFRTSUM", "NFSRT"]].apply(lambda x: x[1] / np.exp(x[0]), axis = 1).to_frame(name = "NFSRTRAT")
# y_df = X_df[["NFSRTPREDICT", "NFSRT"]].apply(lambda x: math.log10(x[0] / x[1]), axis = 1).to_frame(name = "NFSRTLOGRAT")
X_df = raw_df.drop(columns = ["LNNFRTSUM", "NFSRT"])

# X_df["NFSRTPREDICT"] = X_df["NFSRTPREDICT"].apply(math.log10)
# y_df = X_df["NFSRT"].apply(math.log10).to_frame(name = "NFSRT")
# y_df = X_df["NFSRT"].to_frame(name = "NFSRT")
# X_df = X_df.drop(columns = ["NFSRT"])

# train_inds = ((X_df.H11 <= 4) | (X_df.H11 == 6)).nonzero()[0]
train_poly_ids = pd.concat([X_df[X_df.H11 == h11].sample(frac = 0.7, random_state = config.rand_seed) for h11 in range(4, 6 + 1)], axis = 0)
train_inds = X_df.index.isin(train_poly_ids.index).nonzero()[0]

train_bound = int(config.train_split * train_inds.size)

raw_df_train = raw_df.iloc[train_inds[:train_bound]]
X_df_train = X_df.iloc[train_inds[:train_bound]]
y_df_train = y_df.iloc[train_inds[:train_bound]]

raw_df_test = raw_df.drop(raw_df_train.index)
X_df_test = X_df.drop(X_df_train.index)
y_df_test = y_df.drop(y_df_train.index)

X_train = X_df_train.as_matrix()
y_train = y_df_train.as_matrix()

X_test = X_df_test.as_matrix()
y_test = y_df_test.as_matrix()

# Define scaler
scalerX = StandardScaler().fit(X_train)
X_train_scaled = scalerX.transform(X_train)

# Create model
model = define_model.create_model(X_train.shape[1], y_train.shape[1])

# Select the job scheduler
scheduler = lambda x: keras_helpers.lr_drop(x, model, config.scheduler.crit_epochs, config.scheduler.drop_factors)
change_lr = keras.callbacks.LearningRateScheduler(scheduler)

# Train
history = model.fit(X_train_scaled, y_train, batch_size = config.batch_size,
                                             epochs = config.epochs,
                                             verbose = config.verbosity,
                                             validation_split = config.val_split,
                                             shuffle = config.shuffle,
                                             callbacks = [change_lr])

# Save the model
timestamp = get_timestamp()
os.mkdir(timestamp)
# with open(timestamp + "/model.json", "w") as json_stream:
#     json_stream.write(model.to_json())
model.save_weights(timestamp + "/model.h5")
shutil.copy2("model.config", timestamp + "/model.config")
joblib.dump(scalerX, timestamp + "/scaler.pkl")
raw_df_train.to_csv(timestamp + "/raw_train.csv")
X_df_train.to_csv(timestamp + "/X_train.csv")
y_df_train.to_csv(timestamp + "/y_train.csv")
raw_df_test.to_csv(timestamp + "/raw_test.csv")
X_df_test.to_csv(timestamp + "/X_test.csv")
y_df_test.to_csv(timestamp + "/y_test.csv")

# # Do the Gaussian fits
# gaussian_file = "GaussianFits_%s.txt" % modelname
# gaussian_file_all = "GaussianFits_All_%s.txt" % modelname

# h11min = 7
# h11max = 15
# for h11 in range(h11min, h11max+1):
#     print(h11)
#     if h11 <= MAX_TRAIN:
#         if even:
#             filename = sci_header + "TrainEven/SUBCONE_INFO_%d_TEST_EVEN.csv" % h11
#         else:
#             filename = sci_header + "Train%dp/SUBCONE_INFO_%d_TEST_%dp.csv" % (pct, h11, pct)
#     else:
#         filename = sci_header + "SUBCONE_INFO_%d.csv" % h11
#     exraw = pd.read_csv(filename, delimiter=dlm, header=header, names=namesNFV)

#     exdata = pd.DataFrame()
#     for key in tokeep:
#         exdata[key] = exraw[key]

#     # The target (output) data
#     extarget = exraw[targetkey]
#     extarget = np.log(extarget)
#     extarget = extarget.as_matrix()

#     expred = model.predict(exdata, batch_size=config.batch_size)

#     mean_pred = np.mean(expred)
#     mean_real = np.mean(extarget)

#     outfile = "PREDICTION_RESULTS_%d_%s.txt" % (h11, modelname)

#     wf2 = 0

#     with open(outfile, 'w') as f:
#         f.write('\n')
#         for i in range(len(extarget)):
#             diff = expred[i] - extarget[i]
#             if (keras_helpers.within_factor_of_n(extarget[i], expred[i], 2)):
#                 wf2 += 1
#             diff2 = diff**2
#             f.write("%f\t%f\t%f\t%f\n" % (expred[i], extarget[i], diff, diff2))


#     # Get the frequency values
#     p_r = [expred[i] - extarget[i] for i in range(len(extarget))] # Predicted minus real values
#     min_pr = min(p_r)
#     max_pr = max(p_r)
#     hist_lb = math.floor(10*min_pr)/10
#     hist_ub = math.ceil(10*max_pr)/10
#     hist_bins = np.arange(hist_lb, hist_ub+0.1,0.1)
#     hist_vals, bin_edges = np.histogram(p_r, bins=hist_bins)
#     hist_vals = list(hist_vals)
#     bin_edges = list(bin_edges)[:len(hist_vals)]
    
#     x = bin_edges
#     y = hist_vals

#     n = len(x)
#     mean = sum([x[i]*y[i] for i in range(n)])/sum(y)
#     sigma = math.sqrt(sum([y[i]*(x[i]-mean)**2 for i in range(n)])/n)


#     # Do the Gaussian fit
#     popt, pcov = curve_fit(keras_helpers.gaus,x,y,p0=[1,mean,sigma])
#     # plt.plot(x,y,'b+:',label='data')
#     # plt.plot(x,gaus(x,*popt),'ro:',label='fit')
#     # plt.legend()
#     # plt.title('Gaussian Fit')
#     # plt.xlabel('P-R')
#     # plt.ylabel('Amount')
#     # plt.show()

#     print("A: %f" % popt[0])
#     print("mu: %f" % popt[1])
#     print("sigma: %f" % abs(popt[2]))

#     residuals = [y[i] - keras_helpers.gaus(x[i],*popt) for i in range(n)]
#     ss_res = sum([w*w for w in residuals])

#     y_mean = sum(y)/len(y)
#     ss_tot = sum([(w-y_mean)**2 for w in y])

#     r_sq = 1 - (ss_res / ss_tot)
#     print("R^2: %f" % r_sq)

#     mape = keras_helpers.MAPE(list(extarget), list(expred))
#     mse = keras_helpers.MSE(list(extarget), list(expred))

#     print("%d\t%f\t%f\t%f\t%f\t%d\t%f\t%f\t%f\t%f" % (h11, popt[0], popt[1], abs(popt[2]), r_sq, wf2, mape, mse, mean_real, mean_pred))

#     pcov_s = np.array_str(pcov, max_line_width=10000)
#     pcov_s = pcov_s.replace("\n", " ")

#     with open(gaussian_file, 'a') as f:
#         f.write("%d\t%f\t%f\t%f\t%f\t%s\t%d\t%f\t%f\t%f\t%f\n" % (h11, popt[0], popt[1], abs(popt[2]), r_sq, pcov_s, wf2, mape, mse, mean_real, mean_pred))


# for h11 in range(h11min, h11max+1):
#     print(h11)
#     filename = sci_header + "SUBCONE_INFO_%d.csv" % h11
#     exraw = pd.read_csv(filename, delimiter=dlm, header=header, names=namesNFV)

#     exdata = pd.DataFrame()
#     for key in tokeep:
#         exdata[key] = exraw[key]

#     # The target (output) data
#     extarget = exraw[targetkey]
#     extarget = np.log(extarget)
#     extarget = extarget.as_matrix()

#     expred = model.predict(exdata, config.batch_size=config.batch_size)

#     mean_pred = np.mean(expred)
#     mean_real = np.mean(extarget)

#     outfile = "PREDICTION_RESULTS_ALL_%d_%s.txt" % (h11, modelname)

#     wf2 = 0

#     with open(outfile, 'w') as f:
#         f.write('\n')
#         for i in range(len(extarget)):
#             diff = expred[i] - extarget[i]
#             if (within_factor_of_n(extarget[i], expred[i], 2)):
#                 wf2 += 1
#             diff2 = diff**2
#             f.write("%f\t%f\t%f\t%f\n" % (expred[i], extarget[i], diff, diff2))


#     # Get the frequency values
#     p_r = [expred[i] - extarget[i] for i in range(len(extarget))] # Predicted minus real values
#     min_pr = min(p_r)
#     max_pr = max(p_r)
#     hist_lb = math.floor(10*min_pr)/10
#     hist_ub = math.ceil(10*max_pr)/10
#     hist_bins = np.arange(hist_lb, hist_ub+0.1,0.1)
#     hist_vals, bin_edges = np.histogram(p_r, bins=hist_bins)
#     hist_vals = list(hist_vals)
#     bin_edges = list(bin_edges)[:len(hist_vals)]
    
#     x = bin_edges
#     y = hist_vals

#     n = len(x)
#     mean = sum([x[i]*y[i] for i in range(n)])/sum(y)
#     sigma = math.sqrt(sum([y[i]*(x[i]-mean)**2 for i in range(n)])/n)


#     # Do the Gaussian fit
#     popt, pcov = curve_fit(keras_helpers.gaus,x,y,p0=[1,mean,sigma])
#     # plt.plot(x,y,'b+:',label='data')
#     # plt.plot(x,gaus(x,*popt),'ro:',label='fit')
#     # plt.legend()
#     # plt.title('Gaussian Fit')
#     # plt.xlabel('P-R')
#     # plt.ylabel('Amount')
#     # plt.show()

#     print("A: %f" % popt[0])
#     print("mu: %f" % popt[1])
#     print("sigma: %f" % abs(popt[2]))

#     residuals = [y[i] - keras_helpers.gaus(x[i],*popt) for i in range(n)]
#     ss_res = sum([w*w for w in residuals])

#     y_mean = sum(y)/len(y)
#     ss_tot = sum([(w-y_mean)**2 for w in y])

#     r_sq = 1 - (ss_res / ss_tot)
#     print("R^2: %f" % r_sq)

#     mape = keras_helpers.MAPE(list(extarget), list(expred))
#     mse = keras_helpers.MSE(list(extarget), list(expred))

#     print("%d\t%f\t%f\t%f\t%f\t%d\t%f\t%f\t%f\t%f" % (h11, popt[0], popt[1], abs(popt[2]), r_sq, wf2, mape, mse, mean_real, mean_pred))

#     pcov_s = np.array_str(pcov, max_line_width=10000)
#     pcov_s = pcov_s.replace("\n", " ")

#     with open(gaussian_file_all, 'a') as f:
#         f.write("%d\t%f\t%f\t%f\t%f\t%s\t%d\t%f\t%f\t%f\t%f\n" % (h11, popt[0], popt[1], abs(popt[2]), r_sq, pcov_s, wf2, mape, mse, mean_real, mean_pred))

#     print(modelname)