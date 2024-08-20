# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 04:13:53 2024

@author: Armando
"""

from pulp import *
import pandas as pd


def solve_model(O, w1, w2, w3, w4):
    ### Define variables ###
    #Lamda
    l1 = [LpVariable(f"l1_{i}", lowBound=0) for i in range(r)]
    l2 = [LpVariable(f"l2_{i}", lowBound=0) for i in range(r)]
    l3 = [LpVariable(f"l3_{i}", lowBound=0) for i in range(r)]
    l4 = [LpVariable(f"l4_{i}", lowBound=0) for i in range(r)]

    #Positive Slacks
    sp = [[LpVariable(f"sp{i}_{j}", lowBound=0) for j in range(2)]for i in range(4)]
    
    #Negative Slacks
    sn = [[LpVariable(f"sn{i}_{j}", lowBound=0) for j in range(2)]for i in range(4)]
    
    '''
    ### Define the Input-Oriented LP problem ###
    model = LpProblem("Input-Oriented NSBM", LpMinimize)
    
    # Input-Oriented objective function
    model += (w1 * (1 - ((1/2) * lpSum(sn[0][i] * (1/X.iloc[O,i]) for i in range(2)))))\
        + (w2 * (1 - ((1/2) * lpSum(sn[1][i] * (1/Z1.iloc[O,i]) for i in range(2)))))\
            + (w3 * (1 - ((1/2) * lpSum(sn[2][i] * (1/Z2.iloc[O,i]) for i in range(2)))))\
                + (w4 * (1 - ((1/2) * lpSum(sn[3][i] * (1/Z3.iloc[O,i]) for i in range(2)))))
    '''            
                
    ### Define the Output-Oriented LP problem ###
    model = LpProblem("Output-Oriented NSBM", LpMaximize)
    
    
    # Weighted Output-Oriented objective function
    model += (w1 * (1 + ((1/2) * lpSum(sp[0][i] * (1/Z1.iloc[O,i]) for i in range(2)))))\
        + (w2 * (1 + ((1/2) * lpSum(sp[1][i] * (1/Z2.iloc[O,i]) for i in range(2)))))\
            + (w3 * (1 + ((1/2) * lpSum(sp[2][i] * (1/Z3.iloc[O,i]) for i in range(2)))))\
                + (w4 * (1 + ((1/2) * lpSum(sp[3][i] * (1/Y.iloc[O,i]) for i in range(2)))))
    
            

    # Define constraints
    #System
    model += lpSum(l1[j] * X.iloc[j,0] for j in range(r)) + sn[0][0] == X.iloc[O,0]
    model += lpSum(l1[j] * X.iloc[j,1] for j in range(r)) + sn[0][1] == X.iloc[O,1]
    model += lpSum(l4[j] * Y.iloc[j,0] for j in range(r)) - sp[3][0] == Y.iloc[O,0]
    model += lpSum(l4[j] * Y.iloc[j,1] for j in range(r)) - sp[3][1] == Y.iloc[O,1]
    
    #lambda for VRS model
    model += lpSum(l1[j] for j in range(r)) == 1
    model += lpSum(l2[j] for j in range(r)) == 1
    model += lpSum(l3[j] for j in range(r)) == 1
    model += lpSum(l4[j] for j in range(r)) == 1
    
    #Stage 1
    model += lpSum(l1[j] * Z1.iloc[j,0] for j in range(r)) - sp[0][0] == Z1.iloc[O,0]
    model += lpSum(l1[j] * Z1.iloc[j,1] for j in range(r)) - sp[0][1] == Z1.iloc[O,1]
    #Stage 2
    model += lpSum(l2[j] * Z2.iloc[j,0] for j in range(r)) - sp[1][0] == Z2.iloc[O,0]
    model += lpSum(l2[j] * Z2.iloc[j,1] for j in range(r)) - sp[1][1] == Z2.iloc[O,1]
    #Stage 3
    model += lpSum(l3[j] * Z3.iloc[j,0] for j in range(r)) - sp[2][0] == Z3.iloc[O,0]
    model += lpSum(l3[j] * Z3.iloc[j,1] for j in range(r)) - sp[2][1] == Z3.iloc[O,1]
            
    #Linking Parameters
    for i in range(2):
        model += lpSum(l1[j] * Z1.iloc[j,i] for j in range(r)) == lpSum(l2[j] * Z1.iloc[j,i] for j in range(r))
        model += lpSum(l2[j] * Z2.iloc[j,i] for j in range(r)) == lpSum(l3[j] * Z2.iloc[j,i] for j in range(r))
        model += lpSum(l3[j] * Z3.iloc[j,i] for j in range(r)) == lpSum(l4[j] * Z3.iloc[j,i] for j in range(r))
    

    # Solve the problem
    model.solve()
    
    
    # Inverse of the objective function for Output-Oriented model
    objective_value = 1 / value(model.objective)
    '''
    objective_value = value(model.objective)
    '''
    # Collect results
    results = [objective_value
               ] + [value(var) for sublist in sp for var in sublist]
    lmda = [value(var) for sublist in [l1, l2, l3, l4] for var in sublist]
    
    return results, lmda

'''
# Input-Oriented divisional efficiency
def IO_div_efficiency(O, SN1, SN2, SN3, SN4):    
    E1 = 1 - ((1/2) * ((SN1[0] / X.iloc[O,0]) + (SN1[1] / X.iloc[O,1])))
    E2 = 1 - ((1/2) * ((SN2[0] / Z1.iloc[O,0]) + (SN2[1] / Z1.iloc[O,1])))
    E3 = 1 - ((1/2) * ((SN3[0] / Z2.iloc[O,0]) + (SN3[1] / Z2.iloc[O,1])))
    E4 = 1 - ((1/2) * ((SN4[0] / Z3.iloc[O,0]) + (SN4[1] / Z3.iloc[O,1])))
    
    return [E1, E2, E3, E4]
'''

# Output-Oriented divisional efficiency
def OO_div_efficiency(O, SP1, SP2, SP3, SP4):    
    E1 = 1 / (1 + ((1/2) * ((SP1[0] / Z1.iloc[O,0]) + (SP1[1] / Z1.iloc[O,1]))))
    E2 = 1 / (1 + ((1/2) * ((SP2[0] / Z2.iloc[O,0]) + (SP2[1] / Z2.iloc[O,1]))))
    E3 = 1 / (1 + ((1/2) * ((SP3[0] / Z3.iloc[O,0]) + (SP3[1] / Z3.iloc[O,1]))))
    E4 = 1 / (1 + ((1/2) * ((SP4[0] / Y.iloc[O,0]) + (SP4[1] / Y.iloc[O,1]))))
    
    return [E1, E2, E3, E4]

# Importing the Data
df = pd.read_excel("Companies Data.xlsx", index_col="شاخص")
df = df.iloc[:31, :10]

r = len(df)




X = df.iloc[:,[0,1]]
Z1 = df.iloc[:,[2,3]]
Z2 = df.iloc[:,[4,5]]
Z3 = df.iloc[:,[6,7]]
Y = df.iloc[:,[8,9]]

w = pd.read_excel("Parameters' Importance.xlsx", index_col=0)

column_names = ['Z*'] + [f'sp{i}_{j}' for i in range(1, 5) for j in range(1, 3)]
sys_results = pd.DataFrame(columns=column_names)

lm_columns = [f'λ1_{i}' for i in range(1,r+1)] + [f'λ2_{i}' for i in range(1,r+1)]\
    + [f'λ3_{i}' for i in range(1,r+1)] + [f'λ4_{i}' for i in range(1,r+1)]

lm = pd.DataFrame(columns=lm_columns)

#io_dv_efficiency = pd.DataFrame(columns=['E(L)', 'E(IP)', 'E(C)', 'E(F)'])
oo_dv_efficiency = pd.DataFrame(columns=['E(L)', 'E(IP)', 'E(C)', 'E(F)'])

index = [f'DMU{i}' for i in range(1,r+1)]

for O in range(r):
    results, lmda = solve_model(O, w.iloc[0,1], w.iloc[1,1], w.iloc[2,1], w.iloc[3,1])
    sys_results.loc[len(sys_results)] = results
    lm.loc[len(lm)] = lmda
    
    
    SP1 = results[1:3]
    SP2 = results[3:5]
    SP3 = results[5:7]
    SP4 = results[7:9]

    
    oo_dv = OO_div_efficiency(O, SP1, SP2, SP3, SP4)
    oo_dv_efficiency.loc[len(oo_dv_efficiency)] = oo_dv
    
    '''
    io_dv = IO_div_efficiency(O, SN1, SN2, SN3, SN4)
    io_dv_efficiency.loc[len(io_dv_efficiency)] = io_dv
    '''

lm.set_index(pd.Index(index), inplace=True)
sys_results.set_index(pd.Index(index), inplace=True)
#io_dv_efficiency.set_index(pd.Index(index), inplace=True)
oo_dv_efficiency.set_index(pd.Index(index), inplace=True)

### Calculating non-zero lambdas
nonzero_values_df = pd.DataFrame(index=lm.index)

# Iterate over each row in lm
for index, row in lm.iterrows():
    # Initialize dictionaries to store the non-zero column headers based on their categories
    div_L = {}
    div_IP = {}
    div_C = {}
    div_F = {}
    
    # Iterate over each column in the row
    for column in lm.columns:
        # Check if the value in the current column is non-zero
        if row[column] != 0:
            # Extract the last number after the '_' character in the column label
            last_number = column.split('_')[-1]
            
            # Check the prefix of the column header and assign the last number to the corresponding dictionary
            if column.startswith('λ1_'):
                div_L[column] = last_number
            elif column.startswith('λ2_'):
                div_IP[column] = last_number
            elif column.startswith('λ3_'):
                div_C[column] = last_number
            elif column.startswith('λ4_'):
                div_F[column] = last_number
    
    # Assign the dictionaries to the corresponding columns in the new DataFrame
    nonzero_values_df.loc[index, 'Div(L)'] = ', '.join(div_L.values())
    nonzero_values_df.loc[index, 'Div(IP)'] = ', '.join(div_IP.values())
    nonzero_values_df.loc[index, 'Div(C)'] = ', '.join(div_C.values())
    nonzero_values_df.loc[index, 'Div(F)'] = ', '.join(div_F.values())
    
    
final_result = pd.concat([sys_results[['Z*']], oo_dv_efficiency, nonzero_values_df, sys_results[[f'sp{i}_{j}' for i in range(1, 5) for j in range(1, 3)]]], axis=1)

    
