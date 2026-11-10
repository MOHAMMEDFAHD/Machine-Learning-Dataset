from pathlib import Path
import pandas as pd,numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.dummy import DummyClassifier,DummyRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import roc_auc_score,accuracy_score,mean_absolute_error,mean_squared_error
from sklearn.cluster import KMeans
ROOT=Path(__file__).resolve().parents[1]
def pipeline(X,model):
 # dtype-agnostic split: works with pandas 2 (object) and pandas 3 (str) text columns
 num=[c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])];cat=[c for c in X.columns if c not in num]; pre=ColumnTransformer([('numeric',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),num),('categorical',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore')),cat)]);return make_pipeline(pre,model)
def healthcare():
 df=pd.read_csv(ROOT/'data/clean/healthcare_clean.csv'); X=df.drop(columns=['patient_id','cancer_status']);y=df.cancer_status; ids=np.arange(len(df));train,temp=train_test_split(ids,test_size=.30,stratify=y,random_state=42);val,test=train_test_split(temp,test_size=.5,stratify=y.iloc[temp],random_state=42); results={};
 for label,model in [('dummy',DummyClassifier(strategy='prior')),('logistic',LogisticRegression(max_iter=2000)),('forest',RandomForestClassifier(n_estimators=80,min_samples_leaf=5,random_state=42))]:
  fit=pipeline(X.iloc[train],model).fit(X.iloc[train],y.iloc[train]);pred=fit.predict_proba(X.iloc[val])[:,1];results[label]={'validation_auc':round(float(roc_auc_score(y.iloc[val],pred)),4),'validation_accuracy':round(float(accuracy_score(y.iloc[val],pred>=.5)),4)}
 return results,{'train':len(train),'validation':len(val),'test':len(test),'train_positive':int(y.iloc[train].sum()),'validation_positive':int(y.iloc[val].sum()),'test_positive':int(y.iloc[test].sum())}
def retail():
 df=pd.read_csv(ROOT/'data/clean/retail_clean.csv').sort_values(['date','product_id']).reset_index(drop=True);df['date']=pd.to_datetime(df.date);df['day_index']=(df.date-df.date.min()).dt.days;df['weekday']=df.date.dt.dayofweek
 # One-day-ahead forecast: lag1 is available after yesterday closes; no future realized features.
 df['lag1']=df.groupby('product_id').units_sold.shift(1);df['rolling7']=df.groupby('product_id').units_sold.transform(lambda x:x.shift(1).rolling(7,min_periods=1).mean());df=df.dropna(subset=['lag1']);features=['product_id','day_index','weekday','lag1','rolling7'];X=df[features];y=df.units_sold;tr=df.date<df.date.min()+pd.Timedelta(days=69);va=(df.date>=df.date.min()+pd.Timedelta(days=69))&(df.date<df.date.min()+pd.Timedelta(days=84));te=~(tr|va);results={'naive_lag1_mae':round(float(mean_absolute_error(y[va],df.loc[va,'lag1'])),4)}
 for label,model in [('dummy',DummyRegressor()),('ridge',Ridge(alpha=10)),('forest',RandomForestRegressor(n_estimators=80,min_samples_leaf=4,random_state=42,n_jobs=1))]:
  fit=pipeline(X[tr],model).fit(X[tr],y[tr]);results[label+'_validation_mae']=round(float(mean_absolute_error(y[va],fit.predict(X[va]))),4)
 # Scenario B, same-day planning forecast: the day's plan is committed before the store opens,
 # so opening inventory, list price, discount, advertising budget and promotion flag are known.
 # units_sold is capped by inventory on most days, so inventory is the strongest legitimate signal.
 planned=features+['inventory_available','unit_price','discount_pct','advertising_spend','promotion_period','is_weekend']
 for label,model in [('ridge',Ridge(alpha=10)),('forest',RandomForestRegressor(n_estimators=80,min_samples_leaf=4,random_state=42,n_jobs=1))]:
  fit=pipeline(df.loc[tr,planned],model).fit(df.loc[tr,planned],y[tr]);results['planned_'+label+'_validation_mae']=round(float(mean_absolute_error(y[va],fit.predict(df.loc[va,planned]))),4)
 # Scenario C, INVALID on purpose: net_sales_revenue is computed FROM units_sold after the day ends.
 # Teaching use only: shows how post-outcome leakage produces an impossibly good score.
 leaky=planned+['net_sales_revenue']
 fit=pipeline(df.loc[tr,leaky],RandomForestRegressor(n_estimators=80,min_samples_leaf=4,random_state=42,n_jobs=1)).fit(df.loc[tr,leaky],y[tr])
 results['LEAKAGE_DEMO_forest_validation_mae']=round(float(mean_absolute_error(y[va],fit.predict(df.loc[va,leaky]))),4)
 return results,{'train':int(tr.sum()),'validation':int(va.sum()),'test':int(te.sum()),'first_date':str(df.date.min().date())}
def unsupervised():
 df=pd.read_csv(ROOT/'data/clean/healthcare_clean.csv');cols=['age','bmi','biomarker_level','imaging_score','inflammation_marker'];z=StandardScaler().fit_transform(df[cols]);return {'clusters':np.bincount(KMeans(n_clusters=3,n_init=10,random_state=42).fit_predict(z)).tolist()}
if __name__=='__main__':print('classification',healthcare());print('retail',retail());print('unsupervised',unsupervised())
