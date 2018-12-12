# -*- coding: utf-8 -*-

# ライブラリをロード
import pandas as pd

# DataFrameを作成
employee_data = {'employee_id': ['1', '2', '3', '4'],
                 'name': ['Amy Jones', 'Allen Keys', 'Alice Bees',
                 'Tim Horton']}
dataframe_employees = pd.DataFrame(employee_data, columns = ['employee_id',
                                                              'name'])

# DataFrameを作成
sales_data = {'employee_id': ['3', '4', '5', '6'],
              'total_sales': [23456, 2512, 2345, 1455]}
dataframe_sales = pd.DataFrame(sales_data, columns = ['employee_id',
                                                      'total_sales'])

# DataFrameをマージ
pd.merge(dataframe_employees, dataframe_sales, on='employee_id')

##########

# DataFrameをマージ
pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='outer')

##########

# DataFrameをマージ
pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='left')

##########

# DataFrameをマージ
pd.merge(dataframe_employees,
         dataframe_sales,
         left_on='employee_id',
         right_on='employee_id')

