#!/usr/bin/env python
# coding: utf-8

# In[1]:

 
def reg_results(y_true, y_pred):
    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4)) 
    
def features(data,columns):
    feat = data.drop(columns=columns)
    cor = feat.corr().abs().stack().reset_index().sort_values(0, ascending=False)
    
    cor['col_pairs'] = list(zip(cor.level_0,cor.level_1))
    cor['same'] = cor['col_pairs'].map(lambda x: (x[0] in x[1]) or (x[1] in x[0]))
    cor['col_pairs'] = cor['col_pairs'].map(lambda x:sorted(list(x)))
    cor.set_index(['col_pairs'], inplace=True)
    cor = cor[cor['same'] == False]
    cor.drop(columns=['level_0','level_1','same'], inplace=True)
    cor.columns = ['C']
    cor.drop_duplicates(inplace=True)
    return cor

def encode(frame, column):
    import numpy as np
    import pandas as pd
    y = frame[[column]]
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(categories="auto", handle_unknown="error", sparse=False)
    ohe.fit(y)
    encod = ohe.transform(y)
    encod = pd.DataFrame(encod,
    columns=ohe.categories_[0],
    index= frame.index)
    encod.drop(columns = encod.columns[0], axis=1, inplace=True)
    return encod

def evaluate(y_tr, y_te, y_tr_pr, y_te_pr, log=True):
    import numpy as np
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    import matplotlib.pyplot as plt
    from matplotlib import style
    import statsmodels.api as sm
    style.use('ggplot')
    
    if log == True:
        y_tr = np.exp(y_tr)
        y_te = np.exp(y_te)
        y_tr_pr = np.exp(y_tr_pr)
        y_te_pr = np.exp(y_te_pr)
        
    # residuals
    train_res = y_tr - y_tr_pr
    test_res = y_te - y_te_pr
    
    print(f'Train R2 score: {r2_score(y_tr, y_tr_pr)} ')
    print(f'Test R2 score: {r2_score(y_te, y_te_pr)} ')
    print('')
    print(f'Train RMSE: ${mean_squared_error(y_tr, y_tr_pr, squared=False):,.2f} ')
    print(f'Test RMSE: ${mean_squared_error(y_te, y_te_pr, squared=False):,.2f} ')
    print('')
    print(f'Train MAE: ${mean_absolute_error(y_tr, y_tr_pr):,.2f} ')
    print(f'Test MAE: ${mean_absolute_error(y_te, y_te_pr):,.2f} ')
    
    # scatter plot of residuals
    print("\nScatter of residuals:")
    plt.scatter(y_tr_pr, train_res, label='Train')
    plt.scatter(y_te_pr, test_res, label='Test')
    plt.axhline(y=0, color='purple', label='0')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual Price")
    plt.legend()
    plt.show()
    
    print("QQ Plot of residuals:")
    fig, ax = plt.subplots()
    sm.qqplot(train_res, ax=ax, marker='.', color='r', label='Train', alpha=0.3, line='s')
    sm.qqplot(test_res, ax=ax,  marker='.', color='g', label='Test', alpha=0.3)
    plt.legend()

def linpreds(X_tr_scaled, y_tr, X_te_scaled):
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_tr_scaled, y_tr)
    return lr.predict(X_tr_scaled), lr.predict(X_te_scaled)
    

def results(x_tr, x_te, y_tr, y_te, cols):
    import statsmodels.api as sm
    from sklearn.linear_model import LinearRegression
    xcol = x_tr[cols]
    shmod = sm.OLS(endog=y_tr, exog=sm.add_constant(xcol)).fit()
    
    lr = LinearRegression()
    lr.fit(x_tr, y_tr)
    trp, tep = lr.predict(x_tr), lr.predict(x_te)
    
    xr, xf = x_tr[cols], x_te[cols]
    tr, tp = linpreds(xr, y_tr, xf)
    a = evaluate(y_tr, y_te, tr, tp)
    return shmod, a

