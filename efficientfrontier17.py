import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as solver
from functools import reduce


df = pd.read_csv("D:\Desktop\櫃買中心比賽資料\ef14.csv",
                  index_col=0,parse_dates=True)
total_stock = len(df.columns)
returns = df.pct_change()
returns = returns[1:]
returns.head()

covariance_matrix = returns.cov() * 252
stocks_expected_return = returns.mean() * 252
stocks_weights = np.array([1/14, 1/14, 1/14, 1/14,1/14,1/14,1/14,1/14,1/14,1/14,1/14,1/14,1/14,1/14])
portfolio_return = sum(stocks_weights * stocks_expected_return)
portfolio_risk = np.sqrt(reduce(np.dot, [stocks_weights, covariance_matrix, stocks_weights.T]))
print('投資組合預期報酬率為: '+ str(round(portfolio_return,4)))
print('投資組合風險為: ' + str(round(portfolio_risk,4)))

risk_list = []
return_list = []

stop = 0

while stop < 1000:

    try:
        stop += 1
        weight = np.random.rand(total_stock)
        weight = weight / sum(weight)
        return_list.append(sum(stocks_expected_return * weight))
        risk_list.append(np.sqrt(reduce(np.dot, [weight, covariance_matrix, weight.T])))
    except:
        pass

fig = plt.figure(figsize = (10,6))
fig.suptitle('Stochastic simulation results', fontsize=18, fontweight='bold')
ax = fig.add_subplot()
ax.plot(risk_list, return_list, 'o')
ax.set_title('n=1,000,000', fontsize=16)
fig.savefig('result3.png',dpi=300)

def standard_deviation(weights):
    return np.sqrt(reduce(np.dot, [weights, covariance_matrix, weights.T]))


x0 = stocks_weights
bounds = tuple((0, 1) for x in range(total_stock))
constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
minimize_variance = solver.minimize(standard_deviation, x0=x0, constraints=constraints, bounds=bounds)
mvp_risk = minimize_variance.fun
mvp_return = sum(minimize_variance.x * stocks_expected_return)

print('風險最小化投資組合預期報酬率為:' + str(round(mvp_return,2)))
print('風險最小化投資組合風險為:' + str(round(mvp_risk,2)))


for i in range(total_stock):
    print(str(df.columns[i])+' 佔投資組合權重 : ' + str(format(minimize_variance.x[i], '.4f')))


x0 = stocks_weights
bounds = tuple((0, 1) for x in range(total_stock))

efficient_fronter_return_range = np.arange(0.05, 0.35, .005)
efficient_fronter_risk_list = []

for i in efficient_fronter_return_range:
    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1},
                   {'type': 'eq', 'fun': lambda x: sum(x * stocks_expected_return) - i}]
    efficient_fronter = solver.minimize(standard_deviation, x0=x0, constraints=constraints, bounds=bounds)
    efficient_fronter_risk_list.append(efficient_fronter.fun)


risk_free = 0.00405

fig = plt.figure(figsize = (12,6))
fig.subplots_adjust(top=0.85)
ax = fig.add_subplot()

fig.subplots_adjust(top=0.85)
ax0 = ax.scatter(risk_list, return_list,
                c=(np.array(return_list)-risk_free)/np.array(risk_list),
                marker = 'o')
ax.plot(efficient_fronter_risk_list, efficient_fronter_return_range, linewidth=1, color='#251f6b', marker='o',
         markerfacecolor='#251f6b', markersize=5)
ax.plot(mvp_risk, mvp_return,'*',color='r', markerfacecolor='#ed1313',  markersize=25)


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title('Efficient Frontier', fontsize=22, fontweight='bold')
ax.set_xlabel('Risk')
ax.set_ylabel('Return')
fig.colorbar(ax0, ax=ax, label = 'Sharpe Ratio')
plt.savefig('Efficient_Frontier3.png',dpi=300)