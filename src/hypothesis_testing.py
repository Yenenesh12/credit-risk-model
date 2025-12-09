"""
Hypothesis testing module for insurance risk analysis.
Contains statistical tests for validating risk drivers.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency, f_oneway, ttest_ind
import statsmodels.stats.proportion as smprop
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from typing import Dict, Tuple, Optional


class InsuranceHypothesisTester:
    """Class for performing hypothesis testing on insurance data."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the hypothesis tester.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Insurance data with risk metrics
        """
        self.data = data.copy()
        self.results = {}
        
    def calculate_risk_metrics(self):
        """Calculate necessary risk metrics if not present."""
        if 'has_claim' not in self.data.columns:
            self.data['has_claim'] = (self.data['TotalClaims'] > 0).astype(int)
        
        if 'claim_severity' not in self.data.columns:
            self.data['claim_severity'] = np.where(
                self.data['TotalClaims'] > 0, 
                self.data['TotalClaims'], 
                np.nan
            )
        
        if 'margin' not in self.data.columns:
            self.data['margin'] = self.data['TotalPremium'] - self.data['TotalClaims']
        
        if 'loss_ratio' not in self.data.columns:
            self.data['loss_ratio'] = np.where(
                self.data['TotalPremium'] > 0,
                self.data['TotalClaims'] / self.data['TotalPremium'],
                np.nan
            )
    
    def test_province_risk(self, alpha: float = 0.05) -> Dict:
        """
        Test H₀: There are no risk differences across provinces.
        
        Returns:
        --------
        Dict containing test results and business interpretation
        """
        if 'Province' not in self.data.columns:
            raise ValueError("Province column not found in data")
        
        results = {}
        
        # Claim Frequency Test (Chi-square)
        contingency = pd.crosstab(self.data['Province'], self.data['has_claim'])
        chi2, p_freq, dof, expected = chi2_contingency(contingency)
        
        results['claim_frequency'] = {
            'test': 'chi2',
            'statistic': chi2,
            'p_value': p_freq,
            'dof': dof,
            'reject_null': p_freq < alpha,
            'contingency_table': contingency,
            'proportions': contingency.div(contingency.sum(axis=1), axis=0)
        }
        
        # Claim Severity Test (ANOVA)
        severity_data = self.data.dropna(subset=['claim_severity', 'Province'])
        province_groups = [group['claim_severity'].values 
                          for name, group in severity_data.groupby('Province')]
        
        if len(province_groups) > 1:
            f_stat, p_sev = f_oneway(*province_groups)
            
            results['claim_severity'] = {
                'test': 'anova',
                'statistic': f_stat,
                'p_value': p_sev,
                'reject_null': p_sev < alpha,
                'group_stats': severity_data.groupby('Province')['claim_severity'].agg(['mean', 'std', 'count'])
            }
        
        self.results['province_risk'] = results
        return results
    
    def test_zipcode_risk(self, top_n: int = 10, alpha: float = 0.05) -> Dict:
        """
        Test H₀: There are no risk differences between zip codes.
        
        Parameters:
        -----------
        top_n : int
            Number of top zip codes to analyze
        alpha : float
            Significance level
            
        Returns:
        --------
        Dict containing test results
        """
        if 'ZipCode' not in self.data.columns:
            raise ValueError("ZipCode column not found in data")
        
        # Analyze top N zip codes
        top_zipcodes = self.data['ZipCode'].value_counts().head(top_n).index
        zipcode_data = self.data[self.data['ZipCode'].isin(top_zipcodes)].copy()
        
        results = {}
        
        if len(zipcode_data) > 0:
            # Claim Frequency Test
            contingency = pd.crosstab(zipcode_data['ZipCode'], zipcode_data['has_claim'])
            chi2, p_freq, dof, expected = chi2_contingency(contingency)
            
            results['claim_frequency'] = {
                'test': 'chi2',
                'statistic': chi2,
                'p_value': p_freq,
                'dof': dof,
                'reject_null': p_freq < alpha,
                'proportions': contingency.div(contingency.sum(axis=1), axis=0)
            }
            
            # Claim Severity Test
            severity_data = zipcode_data.dropna(subset=['claim_severity'])
            if severity_data['ZipCode'].nunique() > 1:
                zip_groups = [group['claim_severity'].values 
                            for name, group in severity_data.groupby('ZipCode')]
                f_stat, p_sev = f_oneway(*zip_groups)
                
                results['claim_severity'] = {
                    'test': 'anova',
                    'statistic': f_stat,
                    'p_value': p_sev,
                    'reject_null': p_sev < alpha,
                    'group_stats': severity_data.groupby('ZipCode')['claim_severity'].agg(['mean', 'std', 'count'])
                }
        
        self.results['zipcode_risk'] = results
        return results
    
    def test_zipcode_margin(self, top_n: int = 10, alpha: float = 0.05) -> Dict:
        """
        Test H₀: There is no significant margin difference between zip codes.
        
        Returns:
        --------
        Dict containing test results
        """
        if 'ZipCode' not in self.data.columns or 'margin' not in self.data.columns:
            raise ValueError("Required columns (ZipCode or margin) not found")
        
        top_zipcodes = self.data['ZipCode'].value_counts().head(top_n).index
        margin_data = self.data[self.data['ZipCode'].isin(top_zipcodes)].copy()
        
        results = {}
        
        if margin_data['ZipCode'].nunique() > 1:
            # ANOVA test
            zip_groups = [group['margin'].dropna().values 
                         for name, group in margin_data.groupby('ZipCode')]
            f_stat, p_val = f_oneway(*zip_groups)
            
            results['margin'] = {
                'test': 'anova',
                'statistic': f_stat,
                'p_value': p_val,
                'reject_null': p_val < alpha,
                'group_stats': margin_data.groupby('ZipCode')['margin'].agg(['mean', 'std', 'count'])
            }
            
            # Post-hoc Tukey test if significant
            if p_val < alpha and len(zip_groups) > 2:
                tukey_data = margin_data[['ZipCode', 'margin']].dropna()
                tukey_result = pairwise_tukeyhsd(
                    tukey_data['margin'], 
                    tukey_data['ZipCode'], 
                    alpha=alpha
                )
                results['tukey_hsd'] = tukey_result
        
        self.results['zipcode_margin'] = results
        return results
    
    def test_gender_risk(self, alpha: float = 0.05) -> Dict:
        """
        Test H₀: There is no significant risk difference between Women and Men.
        
        Returns:
        --------
        Dict containing test results
        """
        if 'Gender' not in self.data.columns:
            raise ValueError("Gender column not found in data")
        
        # Standardize gender labels
        gender_data = self.data.copy()
        gender_mapping = {'M': 'Male', 'F': 'Female', 'male': 'Male', 'female': 'Female'}
        gender_data['Gender'] = gender_data['Gender'].replace(gender_mapping)
        gender_data = gender_data[gender_data['Gender'].isin(['Male', 'Female'])]
        
        results = {}
        
        if len(gender_data) > 0:
            # Claim Frequency Test (Two-proportion z-test)
            male_claims = gender_data[gender_data['Gender'] == 'Male']['has_claim'].sum()
            male_total = (gender_data['Gender'] == 'Male').sum()
            female_claims = gender_data[gender_data['Gender'] == 'Female']['has_claim'].sum()
            female_total = (gender_data['Gender'] == 'Female').sum()
            
            count = [male_claims, female_claims]
            nobs = [male_total, female_total]
            z_stat, p_freq = smprop.proportions_ztest(count, nobs)
            
            results['claim_frequency'] = {
                'test': 'ztest',
                'statistic': z_stat,
                'p_value': p_freq,
                'reject_null': p_freq < alpha,
                'male_proportion': male_claims / male_total if male_total > 0 else 0,
                'female_proportion': female_claims / female_total if female_total > 0 else 0
            }
            
            # Claim Severity Test (t-test)
            male_severity = gender_data[gender_data['Gender'] == 'Male']['claim_severity'].dropna()
            female_severity = gender_data[gender_data['Gender'] == 'Female']['claim_severity'].dropna()
            
            if len(male_severity) > 0 and len(female_severity) > 0:
                t_stat, p_sev = ttest_ind(male_severity, female_severity, equal_var=False)
                
                results['claim_severity'] = {
                    'test': 'ttest',
                    'statistic': t_stat,
                    'p_value': p_sev,
                    'reject_null': p_sev < alpha,
                    'male_mean': male_severity.mean(),
                    'female_mean': female_severity.mean(),
                    'male_std': male_severity.std(),
                    'female_std': female_severity.std()
                }
        
        self.results['gender_risk'] = results
        return results
    
    def generate_report(self) -> str:
        """Generate comprehensive report of all hypothesis tests."""
        report = "# Hypothesis Testing Report\n\n"
        report += "## Summary of Results\n\n"
        
        for test_name, results in self.results.items():
            report += f"### {test_name.replace('_', ' ').title()}\n"
            
            for metric_name, metric_results in results.items():
                if 'reject_null' in metric_results:
                    report += f"- **{metric_name.replace('_', ' ').title()}**: "
                    report += f"P-value = {metric_results['p_value']:.6f} "
                    report += f"({'REJECT' if metric_results['reject_null'] else 'FAIL TO REJECT'} null hypothesis)\n"
            
            report += "\n"
        
        # Business recommendations
        report += "## Business Recommendations\n\n"
        
        if 'province_risk' in self.results:
            prov_results = self.results['province_risk']
            if any(r.get('reject_null', False) for r in prov_results.values()):
                report += "1. **Province Risk Differences**: Consider implementing region-specific pricing.\n"
        
        if 'gender_risk' in self.results:
            gender_results = self.results['gender_risk']
            if any(r.get('reject_null', False) for r in gender_results.values()):
                report += "2. **Gender Risk Differences**: Use in underwriting models (check legal restrictions).\n"
        
        if 'zipcode_risk' in self.results:
            zip_results = self.results['zipcode_risk']
            if any(r.get('reject_null', False) for r in zip_results.values()):
                report += "3. **Zip Code Risk Differences**: Implement granular risk assessment.\n"
        
        if 'zipcode_margin' in self.results:
            margin_results = self.results['zipcode_margin']
            if margin_results.get('margin', {}).get('reject_null', False):
                report += "4. **Zip Code Margin Differences**: Review pricing in unprofitable areas.\n"
        
        return report