import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as solver
from functools import reduce


tickers = ['1268', '3088', '3388', '4105', '4126', '4506', '4972', '5209', '5371', '5536', '6147', '6263', '6613', '6803'] 
q = len(tickers)

df = pd.read_csv("D:\Desktop\櫃買中心比賽資料\ef14.csv",
                  index_col=0,parse_dates=True)
total_stock = len(df.columns)
returns = df.pct_change()
returns = returns[1:]
print(returns.head())

mean = returns.mean() * 252
cov = returns.cov() * 252

w = np.array([1/14, 1/14, 1/14, 1/14,1/14,1/14,1/14,1/14,1/14,1/14,1/14,1/14,1/14,1/14])

r = sum(w*mean)
var = np.dot(w, cov)
#var = np.dot(sds, w.T)
s = np.sqrt(var) # s = var ** .5

r = sum(w * mean) # multiply
s = np.sqrt(reduce(np.dot, [w, cov, w.T])) # dot multiply

plt.plot(r, s, 'y*')
plt.savefig('test33-1.png',dpi=300)


sds = []
rtn = []

for _ in range(1000000):
    w = np.random.rand(q)
    w /= sum(w)
    rtn.append(sum(mean * w))
    sds.append(np.sqrt(reduce(np.dot, [w, cov, w.T])))

# stop = 0
# while stop < 1000:

#     try:
#         stop += 1
#         w = np.random.rand(q)
#         w = w / sum(w)
#         rtn.append(sum(mean * w))
#         sds.append(np.sqrt(reduce(np.dot, [w, cov, w.T])))
#     except:
#         pass

plt.plot(sds, rtn, 'ro') # ro for red dot
plt.savefig('test33-2.png',dpi=300)

def sd(w):
    return np.sqrt(reduce(np.dot, [w, cov, w.T]))

x0 = np.array([1.0 / q for x in range(q)])

bounds = tuple((0, 1) for x in range(q))

constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1},
               {'type': 'eq', 'fun': lambda x: sum(x * mean) - 0}]


outcome = solver.minimize(sd, x0=x0, constraints=constraints, bounds=bounds)
print(outcome)
print (sum(outcome.x * mean))

for i in range(14):
    print(str(df.columns[i])+' 佔投資組合權重 : ' + str(format(outcome.x[i], '.4f')))

mvp_risk = outcome.fun
mvp_return = sum(outcome.x * mean)

def sd(w):
    return np.sqrt(reduce(np.dot, [w, cov, w.T]))

x0 = np.array([1.0 / q for x in range(q)])
bounds = tuple((0, 1) for x in range(q))

given_r = np.arange(0, .2, .005)
risk = []

for i in given_r:
    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1},
                   {'type': 'eq', 'fun': lambda x: sum(x * mean) - i}]
    outcome = solver.minimize(sd, x0=x0, constraints=constraints, bounds=bounds)
    risk.append(outcome.fun)
plt.plot(risk, given_r, 'x')
plt.savefig('test33-3.png',dpi=300)


constraints = {'type': 'eq', 'fun': lambda x: sum(x) - 1}
minv = solver.minimize(sd, x0=x0, constraints=constraints, bounds=bounds).fun
minvr = sum(solver.minimize(sd, x0=x0, constraints=constraints, bounds=bounds).x * mean)
plt.plot(minv, minvr, 'w*') # w* for white star
plt.grid(True)
plt.title('Efficient Frontier')
plt.xlabel('portfolio volatility')
plt.ylabel('portfolio return')
plt.savefig('test33-4.png',dpi=300)

risk_free = 0.00405

fig = plt.figure(figsize = (12,6))
fig.subplots_adjust(top=0.85)
ax = fig.add_subplot()

fig.subplots_adjust(top=0.85)
ax0 = ax.scatter(sds, rtn,
                c=(np.array(rtn)-risk_free)/np.array(sds),
                marker = 'o')
ax.plot(risk, given_r, linewidth=1, color='#251f6b', marker='o',
         markerfacecolor='#251f6b', markersize=5)
ax.plot(minv, minvr,'*',color='r', markerfacecolor='#ed1313',  markersize=25)


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title('Efficient Frontier', fontsize=22, fontweight='bold')
ax.set_xlabel('Risk')
ax.set_ylabel('Return')
fig.colorbar(ax0, ax=ax, label = 'Sharpe Ratio')
plt.savefig('test33-5.png',dpi=300)