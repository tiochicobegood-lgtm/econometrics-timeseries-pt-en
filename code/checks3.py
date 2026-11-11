import numpy as np, json, pandas as pd, warnings, sys, os
LANG = sys.argv[1] if len(sys.argv)>1 else 'pt'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..',LANG,'tex','fig')
os.makedirs(OUT,exist_ok=True)
TR = {'pt':dict(rf='floresta aleatória',lr='regressão linear',serie='série',coint='Cointegradas: andam juntas',comb='Combinação X2−2·X1: estacionária'),'en':dict(rf='random forest',lr='linear regression',serie='series',coint='Cointegrated: they move together',comb='Combination X2−2·X1: stationary')}[LANG]

warnings.filterwarnings("ignore")
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
rng=np.random.default_rng(11); R={}
# OVB n grande
n=2_000_000; z=rng.normal(size=n); x=0.6*z+rng.normal(size=n); y=1+2*x+1.5*z+rng.normal(size=n)
R["ovb_grande"]=float(np.polyfit(x,y,1)[0]); R["ovb_pop"]=2+1.5*0.6/1.36
# GARCH rho1 de e^2 teórico vs simulação longa
om,al,be=0.1,0.1,0.85; N=400000
e=np.zeros(N); s2=np.zeros(N); s2[0]=om/(1-al-be)
u=rng.normal(size=N)
for t in range(1,N):
    s2[t]=om+al*e[t-1]**2+be*s2[t-1]; e[t]=np.sqrt(s2[t])*u[t]
from statsmodels.tsa.stattools import acf
R["garch_rho1_e2_sim"]=float(acf(e**2,nlags=1)[1])
R["garch_rho1_e2_formula"]=al*(1-al*be-be**2)/(1-2*al*be-be**2)
R["garch_var_sim"]=float(e.var()); R["garch_var_formula"]=om/(1-al-be)
k=((e-e.mean())**4).mean()/e.var()**2; R["garch_kurt_sim"]=float(k)
R["garch_kurt_formula"]=3*(1-(al+be)**2)/(1-(al+be)**2-2*al**2)
# IV minúsculo
zz=np.array([0,1,0,1,1.]); xx=np.array([1,3,2,4,5.]); yy=np.array([2,5,3,7,9.])
b_iv=((zz-zz.mean())*(yy-yy.mean())).sum()/((zz-zz.mean())*(xx-xx.mean())).sum()
b_ols=((xx-xx.mean())*(yy-yy.mean())).sum()/((xx-xx.mean())**2).sum()
R["iv_mini"]=dict(iv=float(b_iv),ols=float(b_ols),num=float(((zz-zz.mean())*(yy-yy.mean())).sum()),den=float(((zz-zz.mean())*(xx-xx.mean())).sum()))
# Cointegração Engle-Granger
from statsmodels.tsa.stattools import coint, adfuller
T=400; w=np.cumsum(rng.normal(size=T)); x1=w+rng.normal(size=T)*0.5; x2=2*w+rng.normal(size=T)*0.5
R["coint_p"]=float(coint(x1,x2)[1])
a=np.cumsum(rng.normal(size=T)); b=np.cumsum(rng.normal(size=T)); R["coint_indep_p"]=float(coint(a,b)[1])
# VAR(1) simulação
A=np.array([[0.5,0.2],[0.0,0.4]]); T=1500; Y=np.zeros((T,2))
for t in range(1,T): Y[t]=A@Y[t-1]+rng.normal(size=2)
from statsmodels.tsa.api import VAR
res=VAR(Y).fit(1,trend="n"); R["var_A"]=res.coefs[0].round(3).tolist()
R["var_eig"]=np.abs(np.linalg.eigvals(A)).round(3).tolist()
g1=res.test_causality(0,[1],kind="f"); g2=res.test_causality(1,[0],kind="f")
R["granger_y2_causa_y1_p"]=float(g1.pvalue); R["granger_y1_causa_y2_p"]=float(g2.pvalue)
# ML: CV aleatório vs ordenado; RF e extrapolação
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
T=1200; e_=rng.normal(size=T); s=np.zeros(T)
for t in range(1,T): s[t]=0.98*s[t-1]+e_[t]
L=5; X=np.column_stack([s[L-i-1:T-i-1] for i in range(L)]); yv=s[L:]
def cv(splitter):
    errs=[]
    for tr,te in splitter.split(X):
        m=RandomForestRegressor(100,random_state=0,n_jobs=-1,min_samples_leaf=3).fit(X[tr],yv[tr])
        errs.append(mean_squared_error(yv[te],m.predict(X[te])))
    return float(np.sqrt(np.mean(errs)))
R["cv_aleatorio_rmse"]=cv(KFold(5,shuffle=True,random_state=0))
R["cv_temporal_rmse"]=cv(TimeSeriesSplit(5))
R["ruido_branco_sd"]=1.0
# RF não extrapola tendência
tt=np.arange(300); ytr=0.05*tt+rng.normal(size=300)*0.5
Xtr=tt[:200].reshape(-1,1); rf=RandomForestRegressor(200,random_state=0).fit(Xtr,ytr[:200])
from sklearn.linear_model import LinearRegression
lr=LinearRegression().fit(Xtr,ytr[:200])
fig,ax=plt.subplots(figsize=(6.4,3)); ax.plot(tt,ytr,lw=.7,c="gray",label=TR["serie"])
ax.plot(tt[200:],rf.predict(tt[200:].reshape(-1,1)),label=TR["rf"]); ax.plot(tt[200:],lr.predict(tt[200:].reshape(-1,1)),label=TR["lr"])
ax.axvline(200,ls="--",c="k",lw=.6); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(OUT,"rf_extrapola.png")); plt.close()
R["rf_ultimo_pred"]=float(rf.predict([[299]])[0]); R["valor_real_tendencia"]=float(0.05*299)
# Cointegração figura
fig,ax=plt.subplots(1,2,figsize=(8,2.8)); ax[0].plot(x1,lw=.7,label="X1"); ax[0].plot(x2,lw=.7,label="X2"); ax[0].legend(fontsize=8); ax[0].set_title(TR["coint"],fontsize=9)
ax[1].plot(x2-2*x1,lw=.7); ax[1].set_title(TR["comb"],fontsize=9); plt.tight_layout(); plt.savefig(os.path.join(OUT,"cointegracao.png")); plt.close()
json.dump(R,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"checks3.json"),"w"),indent=1,default=float); print(json.dumps(R,indent=1,default=float))
