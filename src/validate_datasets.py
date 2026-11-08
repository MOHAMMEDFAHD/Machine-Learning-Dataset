from pathlib import Path
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1]
def validate_datasets():
 h=pd.read_csv(ROOT/'data/clean/healthcare_clean.csv'); r=pd.read_csv(ROOT/'data/clean/retail_clean.csv'); m=pd.read_csv(ROOT/'data/metadata/defect_manifest.csv')
 for name,df,idcol in [('healthcare',h,'patient_id'),('retail',r,'sale_id')]:
  raw=pd.read_csv(ROOT/f'data/raw/{name}_raw.csv');assert len(df)==len(raw)==1000;assert df[idcol].nunique()==1000;assert raw.duplicated().sum()>=2; assert set(m.loc[m.dataset==name,'row_position'])<=set(range(1000));assert {'MCAR missing','MAR missing','MNAR missing','exact duplicate replacement'}<=set(m.loc[m.dataset==name,'defect_category'])
 assert h.cancer_status.isin([0,1]).all() and .1<h.cancer_status.mean()<.8
 assert h.age.between(18,90).all() and h.bmi.between(16,48).all()
 assert len(r.date.unique())==100 and r.groupby('date').size().eq(10).all()
 assert (r.units_sold<=r.inventory_available).all() and (r.return_count<=r.units_sold).all()
 assert np.allclose(r.gross_revenue,r.units_sold*r.unit_price,atol=.011)
 assert np.allclose(r.net_sales_revenue,r.gross_revenue-r.discount_amount-r.return_count*r.unit_price*(1-r.discount_pct),atol=.025)
 assert h.groupby('cancer_status').biomarker_level.mean().loc[1]>h.groupby('cancer_status').biomarker_level.mean().loc[0]
 return {'healthcare_prevalence':float(h.cancer_status.mean()),'retail_mean_units':float(r.units_sold.mean()),'manifest_entries':len(m)}
if __name__=='__main__':print(validate_datasets())
