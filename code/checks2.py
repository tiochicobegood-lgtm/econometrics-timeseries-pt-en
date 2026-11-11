import os
import numpy as np, json, pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess, arma_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf
rng=np.random.default_rng(7)
R={}
# A) convenção de sinal MA(1): X_t = e_t + 0.5 e_{t-1}
th=0.5
R["ma_sinal"]=dict(rho1_formula_mais=th/(1+th**2),
    rho1_numerico=float(arma_acf([1],[1,th],lags=3)[1]),
    rho1_formula_menos_CrChan=-(-th)/(1+th**2))   # com X=e-θ' e, θ'=-0.5 => rho1=-θ'/(1+θ'^2)
# B) AR(2): ACF por Yule-Walker vs numérico
p1,p2=0.5,0.3
r1=p1/(1-p2); r2=p1*r1+p2
R["ar2"]=dict(rho1_yw=r1,rho2_yw=r2,rho1_num=float(arma_acf([1,-p1,-p2],[1],lags=4)[1]),rho2_num=float(arma_acf([1,-p1,-p2],[1],lags=4)[2]))
# C) triângulo de estacionariedade AR(2): raízes fora do círculo
def est(a1,a2): 
    r=np.roots([1,-a1,-a2]); return bool(np.all(np.abs(r)<1))
R["tri"]=[(a1,a2,est(a1,a2)) for a1,a2 in [(0.5,0.3),(1.2,-0.2),(0.5,0.6),(1.0,0.0),(-0.5,0.5)]]
# D) cobertura do IC de previsão AR(1) h=3
phi,mu,s=0.8,0,1; h=3; cov=0; M=4000; T=300
for _ in range(M):
    e=rng.normal(size=T+h); x=np.zeros(T+h)
    for t in range(1,T+h): x[t]=phi*x[t-1]+e[t]
    f=phi**h*x[T-1]; sd=np.sqrt(s**2*(1-phi**(2*h))/(1-phi**2))
    cov += (abs(x[T+h-1]-f)<1.96*sd)
R["cobertura_h3"]=cov/M
R["var_erro_h3_formula"]=s**2*(1-phi**(2*h))/(1-phi**2)
# E) MQ2E manual == linearmodels
n=3000; z=rng.normal(size=(n,2)); u=rng.normal(size=n)
x=z@np.array([0.6,0.4])+0.7*u+rng.normal(size=n); y=1+2*x+u
X=np.c_[np.ones(n),x]; Z=np.c_[np.ones(n),z]
Pz=Z@np.linalg.solve(Z.T@Z,Z.T); b2sls=np.linalg.solve(X.T@Pz@X,X.T@Pz@y)
from linearmodels.iv import IV2SLS
d=pd.DataFrame(dict(y=y,x=x,z1=z[:,0],z2=z[:,1],c=1.))
r=IV2SLS(d.y,d[["c"]],d[["x"]],d[["z1","z2"]]).fit()
R["mq2e"]=dict(manual=b2sls.tolist(),pacote=[float(r.params["c"]),float(r.params["x"])])
# F) Hausman (versão Mundlak/regressão) em painel com efeito correlacionado vs não correlacionado
from linearmodels.panel import PanelOLS, RandomEffects
def painel(corr):
    N,T=400,6; a=rng.normal(size=N)
    X=(0.7*a[:,None] if corr else 0)+rng.normal(size=(N,T))
    Y=1+X+a[:,None]+rng.normal(size=(N,T))
    idx=pd.MultiIndex.from_product([range(N),range(T)],names=["i","t"])
    dp=pd.DataFrame(dict(y=Y.ravel(),x=X.ravel()),index=idx)
    dp["xbar"]=dp.groupby(level=0)["x"].transform("mean")
    m=sm.OLS(dp.y,sm.add_constant(dp[["x","xbar"]])).fit(cov_type="cluster",cov_kwds={"groups":dp.index.get_level_values(0)})
    # Mundlak: coef de xbar mede (RE-FE); teste de xbar=0
    return float(m.params["xbar"]),float(m.pvalues["xbar"])
R["mundlak_corr"]=painel(True); R["mundlak_semcorr"]=painel(False)
# G) viés do IV fraco ~ OLS
json.dump(R,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"checks2.json"),"w"),indent=1,default=float); print(json.dumps(R,indent=1,default=float))
