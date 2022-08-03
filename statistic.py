# statistic.py
# Alessio Burrello <alessio.burrello@unibo.it>
#
# Copyright (C) 2019-2020 University of Bologna
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import product
import numpy as np 
from scipy.stats import t
from scipy.stats import ttest_ind
import pandas as pd
import scipy.io as io
from random import choices
from scipy.stats import chi2_contingency 
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal 
from scipy.stats import f_oneway 
from scipy.stats import levene
import scikit_posthocs as sp

def provide_stratified_bootstrap_sample_indices(bs_sample):

    strata = bs_sample.loc[:, "STRATIFICATION_VARIABLE"].value_counts()
    bs_index_list_stratified = []

    for idx_stratum_var, n_stratum_var in strata.iteritems():
        data_index_stratum = list(bs_sample[bs_sample["STRATIFICATION_VARIABLE"] == idx_stratum_var].index)
        bs_index_list_stratified.extend(choices(data_index_stratum , k = len(data_index_stratum )))
    return bs_index_list_stratified


def creation_training(sheet = "ALL (updated)", group_variable = 'None', subgroup_selection = 'None', feature = 'None'):
    df = pd.read_excel('Bilateral ADX DB - 2022.07.25_JB.xlsx', sheet_name = sheet)
    groups = df[group_variable].values
    subgroup_selection = df[subgroup_selection].values
    centers = df['Centre'].values
    for i, values in enumerate(centers):
        if values in ["Munich", "Torino", "Taiwan"]:
            centers[i] = 0
        elif values in ["Sendai"]:
            centers[i] = 1
        elif values in ["Brisbane"]:
            centers[i] = 2 
        else:
            centers[i] = 3
    data = {'groups':  groups,
            'subgroup_selection': subgroup_selection,
            'STRATIFICATION_VARIABLE': centers,
            'data':  df[feature].values}
    DATA_original = pd.DataFrame(data)
    DATA_original = DATA_original.dropna()
    return DATA_original

class Mannwhitneyu():
    """docstring for ClassName"""
    def __init__(self, dataframe, confidence):
        self.dataframe = dataframe 
        self.confidence = confidence 

    def execute_test(self):
        possible_values = np.unique(self.dataframe['groups'].values)
        groups = []
        for val in possible_values:
            groups.append(self.dataframe.loc[self.dataframe['groups'].values==val,:]["data"].values)
        result = mannwhitneyu(groups[0], groups[1], use_continuity=True, alternative='two-sided', axis=0, method='auto',nan_policy='propagate')
        return result

    def print_test_bootstrapping(self, statistic_array, p_value, multiple_comparisons):
        CI_t = np.percentile(statistic_array,[100*(1-self.confidence )/2,100*(1-(1-self.confidence )/2)]) 
        result_df = pd.DataFrame(np.array([np.mean(statistic_array),CI_t[0],CI_t[1], np.mean(p_value)]).reshape(1,-1),columns=['U statistic','CI U low bound','CI U high bound', 'p-value'])
        print(result_df)

    def print_test(self, t_value, p_value):
        print(f"\nOne test iteration")
        result_df = pd.DataFrame(np.array([t_value, p_value]).reshape(1,-1),columns=['U statistic','p_value'])
        print(result_df)

class TTEST():
    """docstring for ClassName"""
    def __init__(self, dataframe, confidence):
        self.dataframe = dataframe 
        self.confidence = confidence 

    def execute_test(self):
        possible_values = np.unique(self.dataframe['groups'].values)
        groups = []
        for val in possible_values:
            groups.append(self.dataframe.loc[self.dataframe['groups'].values==val,:]["data"].values)
        _ , p_value = levene(groups[0], groups[1], center='mean', proportiontocut=0.05)
        if p_value < 0.05: 
            var = False
        else:
            var = True
        result = ttest_ind(groups[0], groups[1], axis=0, equal_var=var, nan_policy='propagate', permutations=None, random_state=None, alternative='two-sided', trim=0)
        return result

    def print_test_bootstrapping(self, statistic_array, p_value, multiple_comparisons):
        CI_t = np.percentile(statistic_array,[100*(1-self.confidence )/2,100*(1-(1-self.confidence )/2)]) 
        result_df = pd.DataFrame(np.array([np.mean(statistic_array),CI_t[0],CI_t[1], np.mean(p_value)]).reshape(1,-1),columns=['T statistic','CI T low bound','CI T high bound', 'p-value'])
        print(result_df)

    def print_test(self, t_value, p_value):
        print(f"\nOne test iteration")
        result_df = pd.DataFrame(np.array([t_value, p_value]).reshape(1,-1),columns=['T statistic','p_value'])
        print(result_df)

class Anova():
    """docstring for ClassName"""
    def __init__(self, dataframe, confidence):
        self.dataframe = dataframe 
        self.confidence = confidence 

    def execute_test(self):
        possible_values = np.unique(self.dataframe['groups'].values)
        groups = []
        for val in possible_values:
            groups.append(self.dataframe.loc[self.dataframe['groups'].values==val,:]["data"].values)
        try:
            result = f_oneway(groups[0], groups[1], groups[2])
        except:
            return 0, 0
        result1 = sp.posthoc_ttest([groups[0], groups[1], groups[2]], p_adjust = 'bonferroni', pool_sd  = True)
        val_col  = possible_values    
        df = pd.DataFrame(np.array([result1.values[0,1], result1.values[0,2], result1.values[1,2]]).reshape(1,-1), columns=[f'Group_{val_col[0]}_{val_col[1]}', f'Group_{val_col[0]}_{val_col[2]}', f'Group_{val_col[1]}_{val_col[2]}'])
        return result, df

    def print_test_bootstrapping(self, statistic_array, p_value, multiple_comparisons):
        CI_t = np.percentile(statistic_array,[100*(1-self.confidence )/2,100*(1-(1-self.confidence )/2)]) 
        result_df = pd.DataFrame(np.array([np.mean(statistic_array),CI_t[0],CI_t[1], np.mean(p_value)]).reshape(1,-1),columns=['T statistic','CI T low bound','CI T high bound', 'p-value'])
        print(result_df)
        import copy
        df_new = copy.deepcopy(multiple_comparisons[0])
        c0 = 0
        c1 = 0
        c2 = 0
        for i in np.arange(len(multiple_comparisons)):
            c0 += multiple_comparisons[i][multiple_comparisons[0].keys()[0]].values[0]
            c1 += multiple_comparisons[i][multiple_comparisons[0].keys()[1]].values[0]
            c2 += multiple_comparisons[i][multiple_comparisons[0].keys()[2]].values[0]
        c0 = c0 / len(multiple_comparisons)
        c1 = c1 / len(multiple_comparisons)
        c2 = c2 / len(multiple_comparisons)
        df_new[df_new.keys()[0]] = c0
        df_new[df_new.keys()[1]] = c1
        df_new[df_new.keys()[2]] = c2
        print(df_new)

    def print_test(self, t_value, p_value, multiple_comparisons):
        print(f"\nOne test iteration")
        result_df = pd.DataFrame(np.array([t_value, p_value]).reshape(1,-1),columns=['T statistic','p_value'])
        print(result_df)
        print(multiple_comparisons)

class Kruskal():
    """docstring for ClassName"""
    def __init__(self, dataframe, confidence):
        self.dataframe = dataframe 
        self.confidence = confidence 

    def execute_test(self):
        possible_values = np.unique(self.dataframe['groups'].values)
        groups = []
        for val in possible_values:
            groups.append(self.dataframe.loc[self.dataframe['groups'].values==val,:]["data"].values)
        try:
            result = kruskal(groups[0], groups[1], groups[2])
        except:
            return 0, 0
        result1 = sp.posthoc_dunn([groups[0], groups[1], groups[2]], p_adjust = 'bonferroni')
        val_col  = possible_values    
        df = pd.DataFrame(np.array([result1.values[0,1], result1.values[0,2], result1.values[1,2]]).reshape(1,-1), columns=[f'Group_{val_col[0]}_{val_col[1]}', f'Group_{val_col[0]}_{val_col[2]}', f'Group_{val_col[1]}_{val_col[2]}'])
        return result, df

    def print_test_bootstrapping(self, statistic_array, p_value, multiple_comparisons):
        CI_t = np.percentile(statistic_array,[100*(1-self.confidence )/2,100*(1-(1-self.confidence )/2)]) 
        result_df = pd.DataFrame(np.array([np.mean(statistic_array),CI_t[0],CI_t[1], np.mean(p_value)]).reshape(1,-1),columns=['U statistic','CI U low bound','CI U high bound', 'p-value'])
        print(result_df)
        import copy
        df_new = copy.deepcopy(multiple_comparisons[0])
        c0 = 0
        c1 = 0
        c2 = 0
        for i in np.arange(len(multiple_comparisons)):
            c0 += multiple_comparisons[i][multiple_comparisons[0].keys()[0]].values[0]
            c1 += multiple_comparisons[i][multiple_comparisons[0].keys()[1]].values[0]
            c2 += multiple_comparisons[i][multiple_comparisons[0].keys()[2]].values[0]
        c0 = c0 / len(multiple_comparisons)
        c1 = c1 / len(multiple_comparisons)
        c2 = c2 / len(multiple_comparisons)
        df_new[df_new.keys()[0]] = c0
        df_new[df_new.keys()[1]] = c1
        df_new[df_new.keys()[2]] = c2
        print(df_new)

    def print_test(self, t_value, p_value, multiple_comparisons):
        print(f"\nOne test iteration")
        result_df = pd.DataFrame(np.array([t_value, p_value]).reshape(1,-1),columns=['U statistic','p_value'])
        print(result_df)
        print(multiple_comparisons)

class CHI2_multiple():
    """docstring for ClassName"""
    def __init__(self, dataframe, confidence):
        self.dataframe = dataframe 
        self.confidence = confidence 

    def execute_test(self):
        possible_values = np.unique(self.dataframe['groups'].values)
        groups = []
        if len(possible_values) < 3:
            return 0, 0
        for val in possible_values:
            val_groups = np.unique(self.dataframe.loc[self.dataframe['groups'].values==val,:]["data"].values)
            groups.append(self.dataframe.loc[self.dataframe['groups'].values==val,:]["data"].values)
            if len(val_groups) < 2:
                return 0, 0
        possible_values = np.unique(self.dataframe['data'].values)
        contingency_table = []
        for value in possible_values:
            row = []
            for group in groups:
                row.append(sum(group==value))
            contingency_table.append(row)
        result = chi2_contingency(contingency_table, correction=False)
        contingency_table = []
        for value in possible_values:
            row = []
            for group in [0, 1]:
                row.append(sum(groups[group]==value))
            contingency_table.append(row)
        result1 = chi2_contingency(contingency_table, correction=False)
        contingency_table = []
        for value in possible_values:
            row = []
            for group in [0, 2]:
                row.append(sum(groups[group]==value))
            contingency_table.append(row)
        result2 = chi2_contingency(contingency_table, correction=False)
        contingency_table = []
        for value in possible_values:
            row = []
            for group in [1, 2]:
                row.append(sum(groups[group]==value))
            contingency_table.append(row)
        result3 = chi2_contingency(contingency_table, correction=False)
        val_col = np.unique(self.dataframe['groups'].values)
        df = pd.DataFrame(np.array([result1[1], result2[1], result3[1]]).reshape(1,-1), columns=[f'Group_{val_col[0]}_{val_col[1]}', f'Group_{val_col[0]}_{val_col[2]}', f'Group_{val_col[1]}_{val_col[2]}'])
        return result, df

    def print_test_bootstrapping(self, statistic_array, p_value, multiple_comparisons):
        CI_chi2_val = np.percentile(statistic_array,[100*(1-self.confidence )/2,100*(1-(1-self.confidence )/2)]) 
        result_df = pd.DataFrame(np.array([np.median(statistic_array),CI_chi2_val[0],CI_chi2_val[1], np.mean(p_value)]).reshape(1,-1),columns=['CHI2 statistic','CI CHI2 low bound','CI CHI2 high bound', 'p-value'])
        print(result_df)
        import copy
        df_new = copy.deepcopy(multiple_comparisons[0])
        c0 = 0
        c1 = 0
        c2 = 0
        for i in np.arange(len(multiple_comparisons)):
            c0 += multiple_comparisons[i][multiple_comparisons[0].keys()[0]].values[0]
            c1 += multiple_comparisons[i][multiple_comparisons[0].keys()[1]].values[0]
            c2 += multiple_comparisons[i][multiple_comparisons[0].keys()[2]].values[0]
        c0 = c0 / len(multiple_comparisons)
        c1 = c1 / len(multiple_comparisons)
        c2 = c2 / len(multiple_comparisons)
        df_new[df_new.keys()[0]] = c0
        df_new[df_new.keys()[1]] = c1
        df_new[df_new.keys()[2]] = c2
        print(df_new)

    def print_test(self, t_value, p_value, multiple_comparisons):
        print(f"One test iteration")
        result_df = pd.DataFrame(np.array([t_value, p_value]).reshape(1,-1),columns=['Chi statistic','p_value'])
        print(result_df)
        print(multiple_comparisons)

class CHI2():
    """docstring for ClassName"""
    def __init__(self, dataframe, confidence):
        self.dataframe = dataframe 
        self.confidence = confidence 

    def execute_test(self):
        possible_values = np.unique(self.dataframe['groups'].values)
        groups = []
        for val in possible_values:
            groups.append(self.dataframe.loc[self.dataframe['groups'].values==val,:]["data"].values)
        possible_values = np.unique(self.dataframe['data'].values)
        contingency_table = []
        for value in possible_values:
            row = []
            for group in groups:
                row.append(sum(group==value))
            contingency_table.append(row)
        result = chi2_contingency(contingency_table, correction=False)
        return result

    def print_test_bootstrapping(self, statistic_array, p_value, multiple_comparisons):
        CI_chi2_val = np.percentile(statistic_array,[100*(1-self.confidence )/2,100*(1-(1-self.confidence )/2)]) 
        result_df = pd.DataFrame(np.array([np.median(statistic_array),CI_chi2_val[0],CI_chi2_val[1], np.mean(p_value)]).reshape(1,-1),columns=['CHI2 statistic','CI CHI2 low bound','CI CHI2 high bound', 'p-value'])
        print(result_df)

    def print_test(self, t_value, p_value):
        print(f"One test iteration")
        result_df = pd.DataFrame(np.array([t_value, p_value]).reshape(1,-1),columns=['Chi statistic','p_value'])
        print(result_df)

def bootstrapping(iterations, dataframe, algorithm, confidence = 0.95):
    statistic_array = []
    p_value = []
    multiple_comparisons = []
    for i in range(iterations):
        bs_sample = dataframe.copy()
        bs_index_list_stratified = provide_stratified_bootstrap_sample_indices(bs_sample)
        bs_sample = bs_sample.loc[bs_index_list_stratified , :]
        if algorithm == "ttest":
            test = TTEST(bs_sample, confidence)
            result = test.execute_test()
        if algorithm == "chi2":
            test = CHI2(bs_sample, confidence)
            result = test.execute_test()
        if algorithm == "Mannwhitneyu":
            test = Mannwhitneyu(bs_sample, confidence)
            result = test.execute_test()
        if algorithm == "Anova":
            test = Anova(bs_sample, confidence)
            result, df = test.execute_test()
            if result == 0 and df == 0:
                continue
            multiple_comparisons.append(df)
        if algorithm == "chi2_multiple":
            test = CHI2_multiple(bs_sample, confidence)
            result, df = test.execute_test()
            if result == 0 and df == 0:
                continue
            multiple_comparisons.append(df)
        if algorithm == "KruskallWallis":
            test = Kruskal(bs_sample, confidence)
            result, df = test.execute_test()
            if result == 0 and df == 0:
                continue
            multiple_comparisons.append(df)
        statistic_array.append(result[0])
        p_value.append(result[1])
    print(f"Bootstrapping iterations = {len(p_value)}. Elements in dataset: {len(bs_sample)}")
    test.print_test_bootstrapping(statistic_array, p_value, multiple_comparisons)

if __name__ == '__main__':
    DATA_original = creation_training(group_variable = 'ADX Comparation', subgroup_selection = 'LI Ord (0=</=4; 1=>4)', feature = "Baseline systolic BP (mmHg)")
    k=10000
    import random
    confidence = 0.95
    Print_Table = [5]

    outcomes = ['Clinical outcome Complete=0 Partial=1 Absent=2', 'Biochemical outcome Complete=0 Partial=1 Absent=2', 'Clinical outcome Complete=0 Partial=1 Absent=2 12', 'Biochemical outcome Complete=0 Partial=1 Absent=2 12']
    ### Table 8
    if 8 in Print_Table:
        random.seed(0)
        print("\n#######\nTable 8\n#######")
        for out in outcomes:
            print(f'\nTest on {out}')
            DATA_original = creation_training(group_variable = 'ADX Comparation', subgroup_selection = 'LI Ord (0=</=4; 1=>4)', feature = out)
            bootstrapping(k, DATA_original, "chi2", confidence)
            test = CHI2(DATA_original, confidence)
            statistic, p_value, _, _ = test.execute_test()
            test.print_test(statistic, p_value)

    ### Table 9
    if 9 in Print_Table:
        random.seed(0)
        print("\n#######\nTable 9\n#######")
        subgroup = 'LI Ord (0=</=4; 1=>4)'
        for out in outcomes:
            print(f'\nTest on {out}. Decision {subgroup} = 0')
            DATA_original = creation_training(group_variable = 'ADX Comparation', subgroup_selection = subgroup, feature = out)
            DATA_original = DATA_original.loc[DATA_original['subgroup_selection']==0,:]
            bootstrapping(k, DATA_original, "chi2", confidence)
            test = CHI2(DATA_original, confidence)
            statistic, p_value, _, _ = test.execute_test()
            test.print_test(statistic, p_value)

        for out in outcomes:
            print(f'\nTest on {out}. Decision {subgroup} = 1')
            DATA_original = creation_training(group_variable = 'ADX Comparation', subgroup_selection = subgroup, feature = out)
            DATA_original = DATA_original.loc[DATA_original['subgroup_selection']==1,:]
            bootstrapping(k, DATA_original, "chi2", confidence)
            test = CHI2(DATA_original, confidence)
            statistic, p_value, _, _ = test.execute_test()
            test.print_test(statistic, p_value)

    if 10 in Print_Table:
        random.seed(0)
        ### Table 10
        print("\n#######\nTable 10\n#######")
        subgroup = 'Decision for surgery cAVS=0 sAVS=1'
        for out in outcomes:
            print(f'\nTest on {out}. Decision {subgroup} = 0')
            DATA_original = creation_training(group_variable = 'ADX Comparation', subgroup_selection = subgroup, feature = out)
            DATA_original = DATA_original.loc[DATA_original['subgroup_selection']==0,:]
            bootstrapping(k, DATA_original, "chi2", confidence)
            test = CHI2(DATA_original, confidence)
            statistic, p_value, _, _ = test.execute_test()
            test.print_test(statistic, p_value)

        for out in outcomes:
            print(f'\nTest on {out}. Decision {subgroup} = 1')
            DATA_original = creation_training(group_variable = 'ADX Comparation', subgroup_selection = subgroup, feature = out)
            DATA_original = DATA_original.loc[DATA_original['subgroup_selection']==1,:]
            bootstrapping(k, DATA_original, "chi2", confidence)
            test = CHI2(DATA_original, confidence)
            statistic, p_value, _, _ = test.execute_test()
            test.print_test(statistic, p_value)

    if 11 in Print_Table:
        random.seed(0)
        ### Table 11
        print("\n#######\nTable 11\n#######")
        subgroup = 'Partial vs. Total ADX (0=total; 1=partial) '
        for out in outcomes:
            print(f'\nTest on {out}. Decision {subgroup} = 0')
            DATA_original = creation_training(group_variable = 'ADX Comparation', subgroup_selection = subgroup, feature = out)
            DATA_original = DATA_original.loc[DATA_original['subgroup_selection']==0,:]
            bootstrapping(k, DATA_original, "chi2", confidence)
            test = CHI2(DATA_original, confidence)
            statistic, p_value, _, _ = test.execute_test()
            test.print_test(statistic, p_value)

        for out in outcomes:
            print(f'\nTest on {out}. Decision {subgroup} = 1')
            DATA_original = creation_training(group_variable = 'ADX Comparation', subgroup_selection = subgroup, feature = out)
            DATA_original = DATA_original.loc[DATA_original['subgroup_selection']==1,:]
            bootstrapping(k, DATA_original, "chi2", confidence)
            test = CHI2(DATA_original, confidence)
            statistic, p_value, _, _ = test.execute_test()
            test.print_test(statistic, p_value)

    if 12 in Print_Table:
        random.seed(0)
        ### Table 12
        print("\n#######\nTable 12\n#######")
        subgroup = 'APA (1) vs NON-APA (0)'
        for out in outcomes:
            print(f'\nTest on {out}. Decision {subgroup} = 0')
            DATA_original = creation_training(group_variable = 'ADX Comparation', subgroup_selection = subgroup, feature = out)
            DATA_original = DATA_original.loc[DATA_original['subgroup_selection']==0,:]
            bootstrapping(k, DATA_original, "chi2", confidence)
            test = CHI2(DATA_original, confidence)
            statistic, p_value, _, _ = test.execute_test()
            test.print_test(statistic, p_value)

        for out in outcomes:
            print(f'\nTest on {out}. Decision {subgroup} = 1')
            DATA_original = creation_training(group_variable = 'ADX Comparation', subgroup_selection = subgroup, feature = out)
            DATA_original = DATA_original.loc[DATA_original['subgroup_selection']==1,:]
            bootstrapping(k, DATA_original, "chi2", confidence)
            test = CHI2(DATA_original, confidence)
            statistic, p_value, _, _ = test.execute_test()
            test.print_test(statistic, p_value)

    if 13 in Print_Table:
        random.seed(0)
        ### Table 13
        print("\n#######\nTable 13\n#######")
        for out in outcomes:
            print(f'\nTest on {out}')
            DATA_original = creation_training(group_variable = 'APA (1) vs NON-APA (0)', subgroup_selection = 'LI Ord (0=</=4; 1=>4)', feature = out)
            bootstrapping(k, DATA_original, "chi2", confidence)
            test = CHI2(DATA_original, confidence)
            statistic, p_value, _, _ = test.execute_test()
            test.print_test(statistic, p_value)
   
    outcomes = []
    df = pd.read_excel('Bilateral ADX DB - 2022.07.25_JB.xlsx', sheet_name = "ALL (updated)")
    for value in [15,14,16,23,24,27,17,18,19,20,21,22,34,35,33,28,9,41,42,43,46,48,49,50,51,52,47,53,54,55,60,61,62,65,67,68,69,70,71,66,72,73,74]:
        outcomes.append(df.keys()[value])
    tests = ['chi2', 'ttest', 'ttest', 'ttest', 'ttest', 'Mannwhitneyu', 'Mannwhitneyu', 'Mannwhitneyu', 'Mannwhitneyu', 'Mannwhitneyu', 'Mannwhitneyu', 'ttest', 'chi2', 'Mannwhitneyu', 'Mannwhitneyu', 'Mannwhitneyu', 'chi2', 'Mannwhitneyu', 'ttest', 'ttest', 'Mannwhitneyu', 'Mannwhitneyu', 'Mannwhitneyu', 'Mannwhitneyu', 'Mannwhitneyu', 'Mannwhitneyu', 'ttest', 'ttest', 'ttest', 'ttest',  'Mannwhitneyu', 'ttest', 'ttest', 'Mannwhitneyu', 'Mannwhitneyu', 'Mannwhitneyu', 'Mannwhitneyu', 'Mannwhitneyu', 'Mannwhitneyu', 'ttest', 'ttest', 'ttest', 'ttest' ]
    

    ### Table 5
    if 5 in Print_Table:
        random.seed(0)
        print("\n#######\nTable 5\n#######")
        for i, out in enumerate(outcomes):
            print(f'\nTest on {out}')
            DATA_original = creation_training(group_variable = 'ADX Comparation', subgroup_selection = 'LI Ord (0=</=4; 1=>4)', feature = out)
            bootstrapping(k, DATA_original, tests[i], confidence)
            if tests[i] == "ttest":
                test = TTEST(DATA_original, confidence)
                statistic, p_value = test.execute_test()
            if tests[i] == "chi2":
                test = CHI2(DATA_original, confidence)
                statistic, p_value, _, _ = test.execute_test()
            if tests[i] == "Mannwhitneyu":
                test = Mannwhitneyu(DATA_original, confidence)
                statistic, p_value = test.execute_test()
            test.print_test(statistic, p_value)

    tests = ['chi2_multiple', 'Anova', 'Anova', 'Anova', 'Anova', 'KruskallWallis', 'KruskallWallis', 'KruskallWallis', 'KruskallWallis', 'KruskallWallis', 'KruskallWallis', 'Anova', 'chi2_multiple', 'KruskallWallis', 'KruskallWallis', 'KruskallWallis', 'chi2_multiple', 'KruskallWallis', 'Anova', 'Anova', 'KruskallWallis', 'KruskallWallis', 'KruskallWallis', 'KruskallWallis', 'KruskallWallis', 'KruskallWallis', 'Anova', 'Anova', 'Anova', 'Anova',  'KruskallWallis', 'Anova', 'Anova', 'KruskallWallis', 'KruskallWallis', 'KruskallWallis', 'KruskallWallis', 'KruskallWallis', 'KruskallWallis', 'Anova', 'Anova', 'Anova', 'Anova' ]
    ### Table 6
    if 6 in Print_Table:
        random.seed(0)
        print("\n#######\nTable 6\n#######")
        for i, out in enumerate(outcomes):
            print(f'\nTest on {out}')
            DATA_original = creation_training(group_variable = 'Clinical outcome Complete=0 Partial=1 Absent=2 Strat', subgroup_selection = 'LI Ord (0=</=4; 1=>4)', feature = out)
            bootstrapping(k, DATA_original, tests[i], confidence)
            if tests[i] == "Anova":
                test = Anova(DATA_original, confidence)
                result, df = test.execute_test()
                test.print_test(result[0], result[1], df)
            if tests[i] == "chi2_multiple":
                test = CHI2_multiple(DATA_original, confidence)
                result, df = test.execute_test()
                test.print_test(result[0], result[1], df)
            if tests[i] == "KruskallWallis":
                test = Kruskal(DATA_original, confidence)
                result, df = test.execute_test()
                test.print_test(result[0], result[1], df)

    ### Table 7
    if 7 in Print_Table:
        random.seed(0)
        print("\n#######\nTable 7\n#######")
        for i, out in enumerate(outcomes):
            print(f'\nTest on {out}')
            DATA_original = creation_training(group_variable = 'Biochemical outcome Complete=0 Partial=1 Absent=2 Strat', subgroup_selection = 'LI Ord (0=</=4; 1=>4)', feature = out)
            bootstrapping(k, DATA_original, tests[i], confidence)
            if tests[i] == "Anova":
                test = Anova(DATA_original, confidence)
                result, df = test.execute_test()
                test.print_test(result[0], result[1], df)
            if tests[i] == "chi2_multiple":
                test = CHI2_multiple(DATA_original, confidence)
                result, df = test.execute_test()
                test.print_test(result[0], result[1], df)
            if tests[i] == "KruskallWallis":
                test = Kruskal(DATA_original, confidence)
                result, df = test.execute_test()
                test.print_test(result[0], result[1], df)
