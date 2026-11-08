import sys,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def inject_data_quality_issues(df,name,seed=20261110):
 r=np.random.default_rng(seed+(name=='retail')); x=df.copy().astype(object); manifest=[]; ids='patient_id' if name=='healthcare' else 'sale_id'
 def edit(i,c,v,kind,exercise,recover=True):
  old=x.at[i,c]; x.at[i,c]=v; manifest.append(dict(dataset=name,row_position=int(i),original_row_identity=df.at[i,ids],record_identifier=str(x.at[i,ids]),column=c,original_value=str(old),corrupted_value=str(v),defect_category=kind,exercise=exercise,exact_recovery_possible=recover))
 numeric='bmi' if name=='healthcare' else 'advertising_spend'; mar='biomarker_level' if name=='healthcare' else 'customer_rating'; mnar='inflammation_marker' if name=='healthcare' else 'units_sold'; category='smoking_status' if name=='healthcare' else 'category'; invalid='age' if name=='healthcare' else 'inventory_available'; target='cancer_status' if name=='healthcare' else 'units_sold'
 for i in r.choice(len(x),30,replace=False): edit(int(i),numeric,np.nan,'MCAR missing','Random missingness',False)
 eligible=np.flatnonzero((df.age>60).to_numpy() if name=='healthcare' else (df.promotion_period==1).to_numpy());
 for i in r.choice(eligible,min(35,len(eligible)),replace=False):edit(int(i),mar,np.nan,'MAR missing','Observed covariate predicts missingness',False)
 eligible=np.flatnonzero((df[mnar]>df[mnar].quantile(.65)).to_numpy());
 for i in r.choice(eligible,30,replace=False):edit(int(i),mnar,np.nan,'MNAR missing','Higher hidden value more likely missing',False) if mnar!=target else None
 if mnar==target:
  mnar='advertising_spend'; eligible=np.flatnonzero((df[mnar]>df[mnar].quantile(.65)).to_numpy());
  for i in r.choice(eligible,30,replace=False):edit(int(i),mnar,np.nan,'MNAR missing','High hidden spend more likely missing',False)
 for i in r.choice(len(x),12,replace=False):edit(int(i),category,' '+str(x.at[i,category]).upper()+' ','category formatting','Normalize categories')
 for i in r.choice(len(x),5,replace=False):edit(int(i),category,'unknown_typo','invalid category','Validate vocabulary')
 for i in r.choice(len(x),7,replace=False):edit(int(i),invalid,-999,'invalid numeric','Apply domain constraints')
 for i in r.choice(len(x),8,replace=False):edit(int(i),numeric,'not recorded','type mismatch','Coerce numeric',False)
 for i in r.choice(len(x),5,replace=False):edit(int(i),numeric,9999,'corrupted outlier','Investigate extreme values')
 for i in r.choice(len(x),4,replace=False):edit(int(i),ids,str(x.at[0,ids]),'duplicate identifier','Resolve ID collisions')
 if name=='retail':
  for i in r.choice(len(x),20,replace=False):edit(int(i),'date',pd.Timestamp(x.at[i,'date']).strftime('%d/%m/%Y'),'date format','Parse explicit day-first dates')
  for i in r.choice(len(x),5,replace=False):edit(int(i),'return_count',int(df.at[i,'units_sold'])+10,'logical inconsistency','Check returns versus sales')
 else:
  for i in r.choice(len(x),5,replace=False):edit(int(i),'symptom_count',-3,'logical inconsistency','Validate count nonnegative')
 # Replace rows rather than append; displaced clean identity preserved in instructor manifest.
 for dest,source in [(998,1),(999,2)]:
  original=df.loc[dest].to_dict(); x.loc[dest]=x.loc[source].copy(); manifest.append(dict(dataset=name,row_position=dest,original_row_identity=df.at[dest,ids],record_identifier=str(x.at[dest,ids]),column='__whole_row__',original_value=json.dumps(original,default=str),corrupted_value=f'copy of row {source}',defect_category='exact duplicate replacement',exercise='Identify duplicates and displaced original rows',exact_recovery_possible=True))
 return x,pd.DataFrame(manifest)
def main():
 allm=[]
 for name in ['healthcare','retail']:
  df=pd.read_csv(ROOT/f'data/clean/{name}_clean.csv'); raw,m=inject_data_quality_issues(df,name); raw.to_csv(ROOT/f'data/raw/{name}_raw.csv',index=False);allm.append(m)
 pd.concat(allm,ignore_index=True).to_csv(ROOT/'data/metadata/defect_manifest.csv',index=False)
if __name__=='__main__':main()
