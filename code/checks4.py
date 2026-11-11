import os
import numpy as np, json, warnings; warnings.filterwarnings("ignore")
from statsmodels.tsa.stattools import coint
rng=np.random.default_rng(5); R={}
# exemplo mínimo de VI em que VI != MQO
zz=np.array([0,0,1,1,1,0.]); xx=np.array([1,2,2,4,5,3.]); yy=np.array([2,3,6,8,9,3.])
num=((zz-zz.mean())*(yy-yy.mean())).sum(); den=((zz-zz.mean())*(xx-xx.mean())).sum()
ols=((xx-xx.mean())*(yy-yy.mean())).sum()/((xx-xx.mean())**2).sum()
R["iv_mini"]=dict(z=zz.tolist(),x=xx.tolist(),y=yy.tolist(),zbar=zz.mean(),xbar=xx.mean(),ybar=yy.mean(),num=num,den=den,iv=num/den,ols=ols)
# taxa de rejeição EG para independentes e cointegradas
rej_i=0; rej_c=0; M=500; T=200
for _ in range(M):
    a=np.cumsum(rng.normal(size=T)); b=np.cumsum(rng.normal(size=T)); rej_i+=coint(a,b)[1]<0.05
    w=np.cumsum(rng.normal(size=T)); c1=w+rng.normal(size=T)*.5; c2=2*w+rng.normal(size=T)*.5; rej_c+=coint(c1,c2)[1]<0.05
R["eg_rej_indep"]=rej_i/M; R["eg_rej_coint"]=rej_c/M
json.dump(R,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"checks4.json"),"w"),indent=1,default=float); print(json.dumps(R,indent=1,default=float))
