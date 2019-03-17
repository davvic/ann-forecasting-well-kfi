
# coding: utf-8

# # IMPORT ALL DEPENDENCIES 

# In[1]:

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from matplotlib import rcParams
import pandas as pd

from scipy import stats
from scipy.optimize import least_squares
random.seed(12)


# %matplotlib inline
# rcParams['figure.figsize'] = 15, 9

np.random.seed(12)


# # IMPORT INPUT DATA SET

# In[2]:

address = 'C:\\Users\\davidnnamdi\\Downloads\\Documents\\' + input('what is file name? ') + '.xlsx'
column_heads= ['Date', 'Oil', 'WC', 'FTHP', 'Beansize', 'GOR']
df = pd.read_excel(address, names=column_heads)
ur = np.float(input('what is UR in MMSTB?')) * 1000000
df.head()


# # DATA ANALYSIS

# In[3]:

import seaborn as sns
#boxplot
sns.boxplot(x=df['Oil'])



plt.title('FULL DATASET OUTLIER DETECTION')
plt.xlabel('Oil Rate (stb/d)')
plt.show()

#ORIGINAL DATA
uncleanoil = df['Oil']
plt.plot(np.arange(1,len(uncleanoil)+1),df['Oil'],'r')
plt.ylabel('Oil Rate (stb/d)')
plt.xlabel('Days')
plt.title('ORIGINAL DATASET')
plt.show()

uncleanfthp = df['FTHP']

#IQR outlier remover
droplist = []
nn = 30
for i in np.arange(nn,len(df),nn):
    df_o1 = df['Oil'][i-nn:i]
    Q1 = df_o1.quantile(0.25)
    Q3 = df_o1.quantile(0.75)
    IQR = Q3-Q1
    droplist = np.append(droplist, df_o1[df_o1>Q3+1.5*IQR].index)
    droplist = np.append(droplist, df_o1[df_o1<Q1-1.3*IQR].index)
    df_f1 = df['FTHP'][i-nn:i]
    Q1f = df_f1.quantile(0.25)
    Q3f = df_f1.quantile(0.75)
    IQRf = Q3f-Q1f
    droplist = np.append(droplist, df_f1[df_f1>Q3f+1.5*IQRf].index)
    droplist = np.append(droplist, df_f1[df_f1<Q1f-1.5*IQRf].index)

    

print('No. of dropped rows: ', len(droplist))
print()
df_o2 = df.drop(droplist)
clean_oil2 = df_o2['Oil']
clean_fthp = df_o2['FTHP']
c_time2 = np.arange(0,len(clean_oil2))
plt.plot(c_time2,clean_oil2, 'r')
plt.ylabel('Oil Rate (stb/d)')
plt.xlabel('Days')
plt.title('FILTERED DATASET')
plt.show()


plt.plot(np.arange(1,len(uncleanfthp)+1),uncleanfthp,'b')
plt.ylabel('FTHP (psia)')
plt.xlabel('Days')
plt.title('ORIGINAL DATASET')
plt.show()


plt.plot(c_time2,clean_fthp,'b')
plt.ylabel('FTHP (psia)')
plt.xlabel('Days')
plt.title('FILTERED DATASET')
plt.show()




#SET ORIGINAL DATASET = CLEAN DATA
df = df_o2
df.index = np.arange(0,len(df))
# df['Days']=np.arange(1, len(df['Oil'])+1)


# # GENERATE PERFORMANCE PLOT

# In[4]:

#Performance plot
length = np.arange(1, len(df['Oil'])+1)
fig, ax1 = plt.subplots()
ax1.plot(length, df['Oil'], 'r', label='Oil', marker='+', mew=1)
ax1.plot(length, df['GOR'], 'g', label='GOR')
ax1.plot(length, df['FTHP'], 'black', label='FTHP')
ax1.set_title('Well-2 Performance Plot')
ax1.set_xlabel('Days')
ax1.set_ylabel('Oil Rate (stb/d)  /  GOR (scf/bbl)  /  FTHP (psia)')
ax1.legend(loc=6)
ax2 = ax1.twinx()
ax2.plot(length, df['Beansize'], 'magenta', label='Beansize')
ax2.plot(length, df['WC']*100, 'blue', label='Water-cut')
ax2.set_ylabel('Beansize (n/64)  /  Water-Cut(percent)')
ax2.legend(loc=5)
fig.tight_layout()
plt.show()


# # COMPUTE APPROX RUR FOR FILTERED DATASET AND
# #  GENERATE KFI CORRELATION PANEL

# In[5]:

rur = DataFrame((ur - df['Oil'].cumsum()).values/1000000, columns=['RUR'])
# rur = DataFrame([ur/1000000]*len(df), columns=['RUR'])
df = pd.concat([df,rur], axis=1)

#SHOW CORR PANEL FOR WELL DECLINE ANALYSIS
sns.heatmap(df.corr())
plt.title('WELL KFI CORRELATION PANEL')
plt.show()


# # ALGORITHM FOR HYPER-PARAMETER SELECTION

# In[6]:

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint, Callback
from tqdm import tqdm
from keras.optimizers import Adam, RMSprop, SGD, Adagrad
from keras.callbacks import EarlyStopping
import os
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from keras import backend as K
import numpy as np
import tensorflow as tf
import random as rn
from keras import regularizers


class OptimizedAnnNonLinearRegression:

    def __init__(self, data_x, data_y, validation_data_start_index, hidden_layer_range=(1, 2),
                 neuron_count_range=(8, 32), learning_rate_range=(0, 0), batch_size_list=[32],
                 dropout_list=[0.3], activation='relu', optimizer='adam'):

        np.random.seed(12)
        rn.seed(12345)
        tf.set_random_seed(1234)
        self.data_x = data_x
        self.data_y = data_y
        self.learning_rate_range = learning_rate_range
        self.learning_rates = []
        self.hidden_layer_counts = []
        self.neurons_per_layer = []
        self.batch_size_list = []
        self.combinations = []
        self.activation = activation
        self.epochs = 5000
        self.optimizer_name = optimizer
        self.losses_and_filepaths = {}
        self.losses_and_combination = {}
        self.feature_count = len(self.data_x[0])
        self.hist = None
        self.model = None

        if len(hidden_layer_range) == 2:
            self.hidden_layer_min = hidden_layer_range[0]
            self.hidden_layer_max = hidden_layer_range[1]
        else:
            self.hidden_layer_min = hidden_layer_range
            self.hidden_layer_max = hidden_layer_range
        i = self.hidden_layer_min
        while i <= self.hidden_layer_max:
            self.hidden_layer_counts.append(i)
            i += 1

        if len(neuron_count_range) == 2:
            self.neuron_count_min = neuron_count_range[0]
            self.neuron_count_max = neuron_count_range[1]
        else:
            self.neuron_count_min = neuron_count_range
            self.neuron_count_max = neuron_count_range
        i = self.neuron_count_min
        while i <= self.neuron_count_max:
            self.neurons_per_layer.append(i)
            i *= 2

        if learning_rate_range != (0, 0):
            if len(learning_rate_range) == 2:
                self.learning_rate_min = learning_rate_range[0]
                self.learning_rate_max = learning_rate_range[1]
            else:
                self.learning_rate_min = learning_rate_range
                self.learning_rate_max = learning_rate_range
            i = self.learning_rate_min
            while i <= self.learning_rate_max:
                self.learning_rates.append(i)
                i *= 10
        else:
            self.learning_rate_min = 0
            self.learning_rate_max = 0
            self.learning_rates.append(0)

        self.batch_size_list = batch_size_list

        self.scalers_x = []
        self.scaler_x = StandardScaler()
        self.data_x = self.scaler_x.fit_transform(self.data_x)

        self.train_x = self.data_x[:validation_data_start_index]
        self.valid_x = self.data_x[validation_data_start_index:]

        self.scaler_y = StandardScaler()
        self.data_y = self.scaler_y.fit_transform(self.data_y.reshape(-1, 1))

        self.train_y = self.data_y[:validation_data_start_index]
        self.valid_y = self.data_y[validation_data_start_index:]

        for i in range(len(self.learning_rates)):
            for j in range(len(self.hidden_layer_counts)):
                for k in range(len(self.neurons_per_layer)):
                    for l in range(len(self.batch_size_list)):
                        for m in range(len(dropout_list)):
                            combination = []
                            combination.append(self.learning_rates[i])
                            combination.append(self.hidden_layer_counts[j])
                            combination.append(self.neurons_per_layer[k])
                            combination.append(self.batch_size_list[l])
                            combination.append(dropout_list[m])
                            self.combinations.append(combination)

        self.optimizer = optimizer

    def getPathFromCombination(self, combination):
        path = ""
        for val in combination:
            path += str(val)
        return path

    def getModelFromCombination(self, combination):
        model = Sequential()
        
        learn_rate = combination[0]
        layers = combination[1]
        neurons = combination[2]
        batchsize = combination[3]
        dropout = combination[4]
        
        model.add(Dense(neurons, activation=self.activation, input_dim=len(self.train_x[0]), 
                        kernel_regularizer=regularizers.l2(l = 0.001)))
        if dropout != 0 and dropout is not None:
            model.add(Dropout(dropout))
        for i in range(layers - 1):
            model.add(Dense(neurons, activation=self.activation, kernel_regularizer=regularizers.l2(l = 0.001)))
            if dropout != 0 and dropout is not None:
                model.add(Dropout(dropout))
        model.add(Dense(1))

        if self.learning_rate_range != (0, 0) and self.learning_rate_range is not None:
            if self.optimizer_name == 'adam':
                self.optimizer = Adam(lr=learn_rate)
            elif self.optimizer_name == 'rmsprop':
                self.optimizer = RMSprop(lr=learn_rate)
            elif self.optimizer_name == 'adagrad':
                self.optimizer = Adagrad(lr=learn_rate)
            else:
                self.optimizer = SGD(lr=learn_rate)
        else:
            if self.optimizer_name == 'adam':
                self.optimizer = Adam()
            elif self.optimizer_name == 'rmsprop':
                self.optimizer = RMSprop()
            elif self.optimizer_name == 'adagrad':
                self.optimizer = Adagrad()
            else:
                self.optimizer = SGD()

        model.compile(loss='mean_squared_error', optimizer="Adam",
                      metrics=['mae', 'acc'])
        return model

    def beginOptimization(self):

        os.makedirs("NLRWeights/", exist_ok=True)

        for i in tqdm(range(len(self.combinations))):
            del self.model
            K.clear_session()
            combination = self.combinations[i]

            self.model = self.getModelFromCombination(combination)
            path = self.getPathFromCombination(combination)

            filepath = "NLRWeights/" + path + "-{val_loss:.4f}.hdf5"
            checkpointer = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, save_weights_only=False)

            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=300,
                                           verbose=0, mode='auto')
            self.hist = self.model.fit(self.train_x, self.train_y, epochs=self.epochs,
                                  validation_data=(self.valid_x, self.valid_y),
                                  batch_size=combination[3], verbose=0, callbacks=[checkpointer, early_stopping])
            val_loss = "{0:.4f}".format(min(self.hist.history['val_loss']))
            self.losses_and_combination[val_loss] = combination
            for loss in self.hist.history['val_loss']:
                loss = "{0:.4f}".format(loss)
                del_path = "NLRWeights/" + path + "-" + loss + ".hdf5"
                if loss != val_loss and os.path.exists(del_path):
                    os.remove(del_path)
                elif loss == val_loss:
                    self.losses_and_filepaths[loss] = del_path
    
        min_loss = min(self.losses_and_combination.keys())
        min_config = self.losses_and_combination[min_loss]
        for loss in self.losses_and_combination.keys():
            del_path = self.losses_and_filepaths[loss]
            if loss != min_loss and os.path.exists(del_path):
                os.remove(del_path)
            elif loss == min_loss:
                new_path = "NLRWeights/best-model.hdf5"
                if os.path.exists(new_path):
                    os.remove(new_path)
                os.rename(del_path, new_path)
                
        del self.model
        K.clear_session()
        
        self.model = self.getModelFromCombination(min_config)
        self.model.load_weights("NLRWeights/best-model.hdf5")
        self.model.save("NLRWeights/best-model.hdf5")
        self.hist = None
        self.losses_and_combination = None
        print("Optimization Completed Successfully! Best Model Loaded.")
        print("Best Configuration")
        print(self.model.summary())
        print()
        print("Loss Value: " + str(min_loss))
        if self.learning_rate_range != (0, 0) and self.learning_rate_range is not None:
            print("Learning Rate: " + str(min_config[0]))
        print("Batch Size: " + str(min_config[3]))
        print("Dropout: " + str(min_config[4]))
    
    def load_model(self, filepath):
        self.model = load_model(filepath)

    def predict(self, pred_x):
        if len(pred_x[0]) == self.feature_count:
            pred_x = self.scaler_x.transform(pred_x)
            predictions = self.model.predict(pred_x)
            pred_values = self.scaler_y.inverse_transform(predictions.reshape(-1, 1))
            return pred_values
        else:
            print("Feature count mismatch. Train: " + str(self.feature_count) + " Pred: " + str(len(pred_x[0])))
            


# # SPLIT DATASET INTO TRAIN/TEST

# In[14]:

x = df.iloc[:, 2:]
y = df.iloc[:, 1]
split = int(len(x)*0.85)
x_train, x_test, y_train, y_test = x[:split], x[split:], y[:split], y[split:]

plt.plot(np.arange(0, len(x_train)), y_train)
plt.show()

len(x_test)


# # SET MODEL PARAMETER RANGE AND RUN MODEL
# 
# -> SHOW MATCH WITH TEST DATA

# In[24]:

mdl = OptimizedAnnNonLinearRegression(x_train.values, y_train.values, int(len(x_train)*0.9), hidden_layer_range=(2, 2),
                 neuron_count_range=(40,40), learning_rate_range=(0, 0), batch_size_list=[300],
                 dropout_list=[0.3], activation='relu', optimizer='adam')

mdl.beginOptimization()


newy = mdl.predict(x_test.values)
nxxx = np.arange(0,len(newy))
plt.plot(nxxx, newy, 'r')
plt.plot(nxxx, y_test)
plt.show()

x_val = x_train[int(len(x_train)*0.9):]
y_val = y_train[int(len(x_train)*0.9):]
y_pval = mdl.predict(x_val.values)
n_val = np.arange(0,len(x_val))
plt.plot(n_val, y_pval, 'r')
plt.plot(n_val, y_val)
plt.show()


# # MODEL PREDICTION ON TEST AND TRAIN DATASET

# In[25]:

y_predd = mdl.predict(x_test.values)
y_pp =mdl.predict(x_train.values)
amen = np.arange(1, len(y)+1)
nd = pd.DataFrame(amen[split:])[0]
nd2 = pd.DataFrame(amen[:split])[0]


# # COMPARISON OF TRAINED ANN WITH ARPS DCA ON TEST DATA

# In[26]:

d_oil = y_train
d_time = nd2
logoil = np.log(d_oil)
lt = 0
f_time = nd
t_time = np.append(nd2.values,nd.values)

MSE = []
RMSE = []
MAPE = []
PE = []

# y_predd[34]=y_predd[33]

#ARPS EXPONENTIAL
slope, intercept, r_value, p_value, std_err = stats.linregress(d_time[lt:],logoil[lt:])

ss1 = slope
int1 = intercept
exp_q = np.exp(ss1*t_time + int1)
exp_q1 = np.exp(ss1*f_time + int1)
MSE_exp = sum((y_test.values-exp_q1.values)**2)/len(y_test)
RMSE_exp = MSE_exp**0.5
MAPE_exp = (sum(abs((y_test.values-exp_q1.values)/y_test.values))/len(y_test)) * 100
PE_exp = sum((exp_q1.values-y_test.values)/y_test.values)/len(y_test)* 100
MSE = np.append(MSE, MSE_exp)
RMSE = np.append(RMSE, RMSE_exp)
MAPE = np.append(MAPE, MAPE_exp)
PE = np.append(PE, PE_exp)
cum_exp = exp_q1.cumsum()/1000000


#ARPS HYPERBOLIC
def hyperb(x,r,t):
    return (x[0] * (1/((1+ x[1]*x[2]*t)**(1/x[1])))-r)
bounds = (0,[np.inf, 1., np.inf])
x0 = np.array([1, 0.5, 1])
res_lsq1 = least_squares(hyperb, x0, args=(d_oil[lt:], d_time[lt:]), bounds=bounds)
exp1 = res_lsq1.x[0]
exp2 = res_lsq1.x[1]
exp3 = res_lsq1.x[2]

hyp_q = exp1 * (1/((1+ exp2*exp3*t_time)**(1/exp2)))
hyp_q1 = exp1 * (1/((1+ exp2*exp3*f_time)**(1/exp2)))
MSE_hyp = sum((y_test-hyp_q1)**2)/len(y_test)
RMSE_hyp = MSE_hyp**0.5
MAPE_hyp = (sum(abs((y_test-hyp_q1)/y_test))/len(y_test)) * 100
PE_hyp = sum((hyp_q1.values-y_test.values)/y_test.values)/len(y_test)* 100
MSE = np.append(MSE, MSE_hyp)
RMSE = np.append(RMSE, RMSE_hyp)
MAPE = np.append(MAPE, MAPE_hyp)
PE = np.append(PE, PE_hyp)
cum_hyp = hyp_q1.cumsum()/1000000

#ANN
y_predd1 = pd.DataFrame(y_predd)[0]
y_predd1.index = y_test.index
y_predd1
MSE_ann = sum((y_test-y_predd1)**2)/len(y_test)
RMSE_ann = MSE_ann**0.5
MAPE_ann = (sum(abs((y_test-y_predd1)/y_test))/len(y_test)) * 100
PE_ann = sum((y_predd1.values-y_test.values)/y_test.values)/len(y_test)* 100
MSE = np.append(MSE, MSE_ann)
RMSE = np.append(RMSE, RMSE_ann)
MAPE = np.append(MAPE, MAPE_ann)
PE = np.append(PE, PE_ann)
cum_ann = y_predd.cumsum()/1000000

cum_actual = y_test.cumsum()/1000000


plt.plot(d_time, d_oil, 'black', label='History')
plt.plot(nd, y_test, 'orange', label='Test actual production')
plt.plot(t_time[lt:],  exp_q[lt:], 'blue',label='Exp forecast')
plt.plot(t_time[lt:], hyp_q[lt:], 'g',  label='Hyp forecast')
plt.plot(nd, y_predd, 'r', label ='ANN forecast')
plt.xlabel('Days')
plt.ylabel('Oil Rate (stb/d)')
plt.title('COMPARISON OF ARPS TO ANN')
plt.legend(loc='best')
plt.show()



plt.plot(nd, y_test, 'black',label='Test actual production')
plt.plot(f_time, exp_q1, 'blue',label='Exp forecast')
plt.plot(f_time, hyp_q1, 'g',label='Hyp forecast')
plt.plot(nd, y_predd, 'r',label ='ANN forecast')
plt.xlabel('Days')
plt.ylabel('Oil Rate (stb/d)')
plt.title('ZOOM IN OF COMPARISON PLOT TO TEST DATA SECTION')
plt.legend(loc='best')
plt.show()


plt.plot(f_time, cum_exp, 'blue',label='Exp forecast',marker='+', mew=1)
plt.plot(f_time, cum_hyp, 'g',label='Hyp forecast',marker='+', mew=1)
plt.plot(nd, cum_ann, 'r', label ='ANN forecast',marker='+', mew=1)
plt.plot(nd, cum_actual, 'black',label ='Test actual production',marker='+', mew=1)
plt.xlabel('Days')
plt.ylabel('Oil Prodn (MMSTB)')
plt.title('TEST PERIOD CUMM OIL PRODUCTION COMPARISON')
plt.legend(loc='best')
plt.show()


print('MSE: ', np.round(MSE,2))
print('RMSE: ', np.round(RMSE,2))
print('MAPE: ', np.round(MAPE,2))
print('PE: ', np.round(PE,2))


# # DEFINE TIME STEPS FOR GRAPHING PURPOSES
# PLEASE NOTE THAT THESE TIMESTEPS ARE NOT IN ANYWAY USED IN TRAINING OR FORECASTING WITH ANN

# In[27]:

x['Days']=pd.DataFrame(amen)[0]


# # RUN FORECAST AND TEST FOR DIFFERENT SCENARIOS

# In[43]:

plt.plot(nd, y_predd,'r', label='ANN Test data Prediction')
plt.plot(nd, y_test, label='Test data')
plt.plot(nd2, y_train, label='Historical/Training data')
plt.xlabel('Days')
plt.ylabel('Oil Rate (stb/d)')
plt.title('HISTORICAL+ANN')
plt.legend(loc='best')
plt.show()

plt.plot(nd2, y_pp, label='ANN Training data prediction')
plt.plot(nd2, y_train, label='Historical/Training data')
plt.xlabel('Days')
plt.ylabel('Oil Rate (stb/d)')
plt.title('TRAINING DATASET')
plt.legend(loc='best')
plt.show()

plt.plot(nd, y_predd,'r', label='ANN Test data Prediction')
plt.plot(nd, y_test, label='Test data')
plt.xlabel('Days')
plt.ylabel('Oil Rate (stb/d)')
plt.title('TEST DATASET')
plt.legend(loc='best')
plt.show()


bsw = df['WC'].values
cummoil = df['Oil'].cumsum().values

lt=int(len(bsw)*0.2)
lt2 = int(len(bsw)*0.9)

slope, intercept, r_value, p_value, std_err = stats.linregress(cummoil[lt:],bsw[lt:])
s1 = slope
c1 = intercept

cutoff = 1
EUR_oil = (cutoff-c1)/s1
RR_oil = EUR_oil - cummoil[-1]

fit_bsw = cummoil*s1 + c1


plt.plot(cummoil, bsw, 'b', label='Historical Water-Cut')
plt.plot(cummoil[lt:], fit_bsw[lt:], 'r', label='Water-Cut trendline')
plt.xlabel('Cumm. Oil prod. (STB)')
plt.ylabel('Water-Cut (fraction)')
plt.title('BS&W vs Cumm Oil prod. Ratio Plot')
plt.legend(loc='best')
plt.show()





k = x['Days'].values[-1]
# create dataframes for result output
forecast_time = 120
d = np.arange(0, int(forecast_time*30))

method = 'c'
slopem = (50-x['FTHP'][len(x['Days'])-1])/(d[-1]-d[0])
wc = []
fthp = []
bn = []
rur = []
pred = []
for values in d:
    aa = x['WC'][len(x['Days'])-1]
    bb = x['FTHP'][len(x['Days'])-1]
    cc = x['Beansize'][len(x['Days'])-1]
    dd = x['RUR'][len(x['Days'])-1]
    if x['GOR'][len(x['Days'])-1] == x['GOR'][0]:
        ee = x['GOR'][len(x['Days'])-1]
    else:
        ee = x['GOR'][lt2:].mean()
#         ee = x['GOR'].values[-1]
    if values == 0:
        wc1 = aa
        fthp1 = bb
        bn1 = cc
        ak = dd
        inp = np.array([ wc1, fthp1, bn1, ee, ak])
#         p = sc.transform(inp.reshape(1,6))
        p = inp.reshape(1,5)
        pp = mdl.predict(p)
        pred = np.append(pred,pp)
        rur = np.append(rur,ak)
        wc = np.append(wc, wc1)
        bn = np.append(bn, bn1)
        fthp = np.append(fthp, fthp1)
    else:
        if method == 'c':
            decln = np.log(bb/50)/len(d)
#             fthp1 = slopem*(values-d[0]) + bb
            fthp1 = bb*np.exp(-decln*values)
        else:
            fthp1 = bb
        bn1 = np.append(wc,cc)[-1]
        if pred[-1]<50:
            break
        else:
            tpred = (pred.cumsum()[-1]/1000000)
            Np = tpred + (df['Oil'].cumsum().values[-1]/1000000)
            ak = ur/1000000 - Np
            wc1 = (Np*1000000)*(s1) + c1
#             values = float(values)
#             wc1 = (-3e-16*(values**5))+(6e-13*(values**4))-(1e-10*(values**2))-(6e-8*(values**2))+(4e-5*values)+0.0003
#             wc1 = (2e-10*(values**2))+(1e-5*values)+ 0.3701
#             wc1 = aa
#             declnw = np.log(aa/0.99)/len(d)
#             wc1 = aa*np.exp(-declnw*values)
    
        
            if wc1 > 1:
                break
            else:
                inp = np.array([ wc1, fthp1, bn1, ee, ak])
#                 p = sc.transform(inp.reshape(1,6))
#                 pp = dc.inverse_transform(classifier.predict(p))
                p = inp.reshape(1,5)
                pp = mdl.predict(p)
                pred = np.append(pred,pp)
                if rur[-1]<0:
                    break
                else:
                    rur = np.append(rur,ak)
            wc = np.append(wc, wc1)
            bn = np.append(bn, bn1)
            fthp = np.append(fthp, fthp1)

newx = np.arange(0, len(pred))
newxx = newx +x['Days'][len(x['Days'])-1]

Hist = np.append(y_pp, y_predd)

for_t = d+x['Days'][len(x['Days'])-1]
exp_pred = np.exp(ss1*for_t + int1)
hyp_pred = exp1 * (1/((1+ exp2*exp3*for_t)**(1/exp2)))



plt.plot(x['Days'], y, 'r', label='Production data')
plt.plot(x['Days'], Hist, 'black', label='History match')
plt.plot(newxx,pred, 'g', label='Forecast')
plt.plot(for_t,exp_pred, 'orange', label='EXP Forecast')
plt.plot(for_t,hyp_pred, 'y', label='HYP Forecast')
plt.ylabel('Oil Rate (bbl/d)')
plt.xlabel('Time(Days)')
plt.title('Rate-Time plot of ANN Forecast and Production data')
plt.legend(loc='best')
plt.show()

print('GOR: ',round(ee), 'SCF/STB')
print('cum oil prod: ', round(sum(pred)/1000000,3), 'MMSTB')
print('No. of days predicted: ',len(pred), 'days')
if method == 'c':
    print('Linear changing FTHP forecast')
else:
    print('Constant FTHP forecast')


# # EXTRACT MONTHLY FORECASTS FROM DAILY

# In[42]:

mndn = np.arange(30, len(pred), 30)
dmn = []
mnavg = []
gasavg = []
watavg = []
wcavg = []
for val in mndn:
#     if val == 0:
#         dmn = np.append(dmn,val)
#     else:
    avg = pred[val-30:val].mean()
    gavg = avg*ee/1000
    wcm = wc[val-30:val].mean()
    wcavg = np.append(wcavg, wcm)
    wavg = (avg*wcm)/(1-wcm)
    mnavg = np.append(mnavg,avg)
    gasavg = np.append(gasavg, gavg)
    watavg = np.append(watavg, wavg)
    dmn = np.append(dmn, val)
loil = pred[int(dmn[-1]):].mean()
lwat = wc[int(dmn[-1]):].mean()
mnavg = pd.DataFrame(np.append(mnavg, loil), columns=['Oil Avg rate (bbl/d)']).dropna()
gasavg = pd.DataFrame(np.append(gasavg, (loil*ee/1000)), columns=['Gas Avg rate (Mscf/d)']).dropna()
watavg =  pd.DataFrame(np.append(wcavg, lwat), columns=['Water Cut Avg']).dropna()



kxx = str(df['Date'].values[-1])
yrr = int(kxx[0]+kxx[1]+kxx[2]+kxx[3])
mnnth = int(kxx[5]+kxx[6])
admn = (len(mnavg)-(12-mnnth))
adyr = int(admn/12)
fnmn = admn - adyr*12
if fnmn == 0:
    fnyr = yrr+adyr
    fnmn = 12
else:
    fnyr = yrr+adyr+1
fmonths = pd.date_range(str(yrr)+'-'+str(mnnth+1)+'-'+str(1),
              str(fnyr)+'-'+str(fnmn)+'-'+str(28), 
              freq='MS').strftime("%Y-%b")

pred_mnth = pd.DataFrame(fmonths, columns=['Forecast Months'])



m_oup = pd.concat([pred_mnth, mnavg, gasavg, watavg], axis=1)


plt.plot(np.arange(0,len(mnavg)), mnavg.values, 'g', label='Forecast', marker='+', mew=2.5)
plt.title('ANN OIL MONTHLY FORECAST')
plt.xlabel('Months')
plt.ylabel('Oil Rate (stb/d)')
plt.show()

m_oup.head()


# # SAVE FORECAST RESULTS

# In[85]:

#save results as excel file to the path C:\\Users\\davidnnamdi
name = input('save results as: ')
writer = pd.ExcelWriter(name + ' DAILY ANN FORECAST.xlsx', engine='xlsxwriter')
pd.DataFrame(pred).to_excel(writer, sheet_name='Sheet1')
writer.save()

