import unittest
import pandas as pd
from src.data import features
class TestFeatures(unittest.TestCase):
    def test_first_row_uses_past_only(self):
        df=pd.DataFrame({'date':pd.date_range('2020-01-01',periods=24,freq='MS'),'production_mn_kg':range(24)})
        f,_=features(df)
        self.assertEqual(f.iloc[0].lag_1,11)
        self.assertEqual(f.iloc[0].lag_12,0)
if __name__=='__main__':unittest.main()
