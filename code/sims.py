import numpy as np, json, sys, os
LANG = sys.argv[1] if len(sys.argv)>1 else 'pt'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..',LANG,'tex','fig')
os.makedirs(OUT,exist_ok=True)
TR = {
 'pt':dict(ar2='AR(2): φ1=0,5, φ2=−0,3',ma1='MA(1): θ=0,8',acf='FAC amostral',pacf='FACP amostral',five='Cinco passeios aleatórios independentes',two='Duas séries independentes que parecem relacionadas',sA='série A',sB='série B',pred='previsão pontual',ic='IC 95%',h='horizonte h',garch='Retornos simulados GARCH(1,1): calmaria e turbulência'),
 'en':dict(ar2='AR(2): φ1=0.5, φ2=−0.3',ma1='MA(1): θ=0.8',acf='Sample ACF',pacf='Sample PACF',five='Five independent random walks',two='Two independent series that look related',sA='series A',sB='series B',pred='point forecast',ic='95% PI',h='horizon h',garch='Simulated GARCH(1,1) returns: calm and turbulence'),
}[LANG]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import acf, pacf, adfuller
from linearmodels.iv import IV2SLS
from linearmodels.panel import PanelOLS, RandomEffects, FirstDifferenceOLS
import pandas as pd
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
rng = np.random.default_rng(42)
R = {}

# 1) Variável omitida: viés de MQO
n=5000
z = rng.normal(size=n); x = 0.6*z + rng.normal(size=n)
y = 1 + 2*x + 1.5*z + rng.normal(size=n)
b_short = sm.OLS(y, sm.add_constant(x)).fit().params[1]
b_long  = sm.OLS(y, sm.add_constant(np.c_[x,z])).fit().params[1]
# viés teórico: 1.5 * cov(x,z)/var(x)
vies = 1.5*np.cov(x,z)[0,1]/np.var(x,ddof=1)
R["ovb"]=dict(curto=b_short, longo=b_long, previsto=2+vies)

# 2) Erro de medida: atenuação
x_true = rng.normal(size=n); y = 1+2*x_true+rng.normal(size=n)
x_obs = x_true + rng.normal(scale=1.0,size=n)   # var(u)=var(x*) -> fator 0.5
b_obs = sm.OLS(y, sm.add_constant(x_obs)).fit().params[1]
R["att"]=dict(obs=b_obs, previsto=2*0.5)

# 3) Heterocedasticidade: EP usual vs robusto
x = rng.normal(size=n); y = 1+2*x + np.abs(x)*rng.normal(size=n)
m = sm.OLS(y, sm.add_constant(x)).fit()
mr = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC1")
R["het"]=dict(ep_usual=m.bse[1], ep_robusto=mr.bse[1])

# 4) VI: endogeneidade, MQO vs MQ2E
z = rng.normal(size=n); u = rng.normal(size=n)
x = 0.8*z + 0.7*u + rng.normal(size=n)
y = 1 + 2*x + u
ols = sm.OLS(y, sm.add_constant(x)).fit().params[1]
df = pd.DataFrame(dict(y=y,x=x,z=z,c=1.0))
iv = IV2SLS(df.y, df[["c"]], df[["x"]], df[["z"]]).fit()
# instrumento fraco
zw = rng.normal(size=n); xw = 0.02*zw + 0.7*u + rng.normal(size=n); yw = 1+2*xw+u
dfw = pd.DataFrame(dict(y=yw,x=xw,z=zw,c=1.0))
ivw = IV2SLS(dfw.y, dfw[["c"]], dfw[["x"]], dfw[["z"]]).fit()
R["iv"]=dict(ols=ols, iv=float(iv.params["x"]), iv_fraco=float(ivw.params["x"]),
             F1=float(iv.first_stage.diagnostics["f.stat"].iloc[0]) if hasattr(iv.first_stage,"diagnostics") else None)

# 5) Painel: EF vs EA vs POLS com efeito correlacionado
N,T=500,6
a = rng.normal(size=N)
X = 0.7*a[:,None] + rng.normal(size=(N,T))
Y = 1 + 1.0*X + a[:,None] + rng.normal(size=(N,T))
idx = pd.MultiIndex.from_product([range(N),range(T)],names=["i","t"])
dp = pd.DataFrame(dict(y=Y.ravel(),x=X.ravel()),index=idx)
pols = sm.OLS(dp.y, sm.add_constant(dp.x)).fit().params["x"]
fe = PanelOLS(dp.y, dp[["x"]], entity_effects=True).fit().params["x"]
re = RandomEffects(dp.y, sm.add_constant(dp[["x"]])).fit().params["x"]
fd = FirstDifferenceOLS(dp.y, dp[["x"]]).fit().params["x"]
R["painel"]=dict(pols=pols, fe=fe, re=re, fd=fd)

# 6) Séries: ACF/PACF teóricas de AR(2) e MA(1) + amostral
fig,ax = plt.subplots(2,3,figsize=(9,4.8))
ar2 = ArmaProcess([1,-0.5,0.3],[1]); ma1 = ArmaProcess([1],[1,0.8])
for r,(pr,nm) in enumerate([(ar2,TR["ar2"]),(ma1,TR["ma1"])]):
    s = pr.generate_sample(500, distrvs=rng.standard_normal)
    ax[r,0].plot(s,lw=.7); ax[r,0].set_title(nm,fontsize=9)
    ax[r,1].stem(acf(s,nlags=15)[1:],basefmt=" "); ax[r,1].set_title(TR["acf"],fontsize=9)
    ax[r,2].stem(pacf(s,nlags=15)[1:],basefmt=" "); ax[r,2].set_title(TR["pacf"],fontsize=9)
    for k in (1,2):
        ax[r,k].axhline(1.96/np.sqrt(500),ls="--",c="gray",lw=.6); ax[r,k].axhline(-1.96/np.sqrt(500),ls="--",c="gray",lw=.6)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"acf_pacf.png")); plt.close()

# 7) Regressão espúria e raiz unitária
rej=0; M=1000; rej_stat=0
for _ in range(M):
    a_ = np.cumsum(rng.normal(size=200)); b_ = np.cumsum(rng.normal(size=200))
    f = sm.OLS(a_, sm.add_constant(b_)).fit()
    rej += f.pvalues[1]<0.05
R["espuria"]=dict(rejeicao_5pct=rej/M)
rw = np.cumsum(rng.normal(size=300)); wn = rng.normal(size=300)
R["adf"]=dict(passeio_p=adfuller(rw,regression="c",autolag="AIC")[1], ruido_p=adfuller(wn,regression="c",autolag="AIC")[1])
fig,ax = plt.subplots(1,2,figsize=(8,3))
for _ in range(5): ax[0].plot(np.cumsum(rng.normal(size=200)),lw=.8)
ax[0].set_title(TR["five"],fontsize=9)
a_ = np.cumsum(rng.normal(size=200)); b_ = np.cumsum(rng.normal(size=200))
ax[1].plot(a_,label=TR["sA"]); ax[1].plot(b_,label=TR["sB"]); ax[1].legend(fontsize=8)
ax[1].set_title(TR["two"],fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"espuria.png")); plt.close()

# 8) Previsão AR(1): decaimento para a média e intervalo crescente
phi=0.8; mu=10; sig=1
h = np.arange(1,21); pred = mu + phi**h*(5)  # último desvio = 5
var = sig**2*(1-phi**(2*h))/(1-phi**2)
fig,ax=plt.subplots(figsize=(6,3))
ax.plot(h,pred,label=TR["pred"]); ax.fill_between(h,pred-1.96*np.sqrt(var),pred+1.96*np.sqrt(var),alpha=.2,label=TR["ic"])
ax.axhline(mu,ls="--",c="gray",lw=.8); ax.set_xlabel(TR["h"]); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"previsao_ar1.png")); plt.close()

# 9) GARCH: agrupamento de volatilidade
from arch import arch_model
om,al,be = 0.1,0.1,0.85
e = np.zeros(2000); s2=np.zeros(2000); s2[0]=om/(1-al-be)
for t in range(1,2000):
    s2[t]=om+al*e[t-1]**2+be*s2[t-1]; e[t]=np.sqrt(s2[t])*rng.normal()
am = arch_model(e,mean="Zero",vol="GARCH",p=1,q=1).fit(disp="off")
R["garch"]=dict(omega=am.params["omega"],alpha=am.params["alpha[1]"],beta=am.params["beta[1]"],
                acf_e1=float(acf(e,nlags=1)[1]), acf_e2_1=float(acf(e**2,nlags=1)[1]))
fig,ax=plt.subplots(figsize=(7,2.6)); ax.plot(e,lw=.5); ax.set_title(TR["garch"],fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"garch.png")); plt.close()

json.dump(R,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"resultados.json"),"w"),indent=1,default=float)
print(json.dumps(R,indent=1,default=float))
