"""
Task 3: Statistical Hypothesis Testing for Insurance Risk Analysis
Author: Senior Data Scientist
Date: 2024
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
import statsmodels.api as sm
from statsmodels.stats.power import TTestIndPower
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class InsuranceHypothesisTesting:
    """Class for conducting A/B hypothesis testing on insurance data"""
    
    def __init__(self, data_path='data/insurance_data.csv'):
        """
        Initialize the hypothesis testing framework
        
        Parameters:
        -----------
        data_path : str
            Path to the insurance data file
        """
        self.data = pd.read_csv(data_path)
        self.results = {}
        
        # Calculate derived metrics
        self._calculate_metrics()
        
    def _calculate_metrics(self):
        """Calculate necessary metrics for hypothesis testing"""
        # Claim Frequency (binary: 1 if claims > 0, else 0)
        self.data['Claim_Frequency'] = (self.data['TotalClaims'] > 0).astype(int)
        
        # Claim Severity (average claim amount when claim occurs)
        self.data['Claim_Severity'] = self.data.apply(
            lambda x: x['TotalClaims'] if x['TotalClaims'] > 0 else np.nan, axis=1
        )
        
        # Margin
        self.data['Margin'] = self.data['TotalPremium'] - self.data['TotalClaims']
        
        # Loss Ratio
        self.data['Loss_Ratio'] = self.data['TotalClaims'] / self.data['TotalPremium']
        
    def test_province_risk_differences(self):
        """
        H₀: There are no risk differences across provinces
        
        Tests:
        1. Claim Frequency differences (Chi-squared test)
        2. Claim Severity differences (ANOVA)
        3. Margin differences (ANOVA)
        """
        print("=" * 80)
        print("HYPOTHESIS 1: Province Risk Differences")
        print("=" * 80)
        
        results = {}
        
        # 1. Claim Frequency Test (Chi-squared)
        contingency_table = pd.crosstab(self.data['Province'], 
                                        self.data['Claim_Frequency'])
        chi2, p_freq, dof, expected = chi2_contingency(contingency_table)
        results['Claim_Frequency'] = {
            'test': 'Chi-squared',
            'statistic': chi2,
            'p_value': p_freq,
            'reject_null': p_freq < 0.05
        }
        
        # 2. Claim Severity Test (ANOVA)
        # Filter only claims that occurred
        severity_data = self.data.dropna(subset=['Claim_Severity'])
        provinces = severity_data['Province'].unique()
        
        # Prepare data for ANOVA
        severity_groups = [severity_data[severity_data['Province'] == p]['Claim_Severity'] 
                          for p in provinces if len(severity_data[severity_data['Province'] == p]) > 1]
        
        if len(severity_groups) > 1:
            f_stat, p_sev = f_oneway(*severity_groups)
            results['Claim_Severity'] = {
                'test': 'ANOVA',
                'statistic': f_stat,
                'p_value': p_sev,
                'reject_null': p_sev < 0.05
            }
        
        # 3. Margin Test (ANOVA)
        margin_groups = [self.data[self.data['Province'] == p]['Margin'] 
                        for p in self.data['Province'].unique() 
                        if len(self.data[self.data['Province'] == p]) > 1]
        
        if len(margin_groups) > 1:
            f_stat, p_margin = f_oneway(*margin_groups)
            results['Margin'] = {
                'test': 'ANOVA',
                'statistic': f_stat,
                'p_value': p_margin,
                'reject_null': p_margin < 0.05
            }
        
        # Calculate effect sizes and business metrics
        province_stats = self.data.groupby('Province').agg({
            'Claim_Frequency': 'mean',
            'Claim_Severity': 'mean',
            'Margin': 'mean',
            'Loss_Ratio': 'mean'
        }).round(4)
        
        self.results['province'] = {
            'test_results': results,
            'province_stats': province_stats
        }
        
        # Visualization
        self._plot_province_comparisons(province_stats)
        
        return results, province_stats
    
    def test_zipcode_risk_differences(self, top_n=10):
        """
        H₀: There are no risk differences between zip codes
        
        Note: Testing all zip codes might be too granular.
        We'll test the top N zip codes by policy count.
        """
        print("\n" + "=" * 80)
        print("HYPOTHESIS 2: Zip Code Risk Differences")
        print("=" * 80)
        
        # Get top N zip codes by policy count
        zip_counts = self.data['ZipCode'].value_counts().head(top_n)
        top_zips = zip_counts.index.tolist()
        
        # Filter data for top zip codes
        zip_data = self.data[self.data['ZipCode'].isin(top_zips)]
        
        results = {}
        
        # 1. Claim Frequency Test (Chi-squared)
        contingency_table = pd.crosstab(zip_data['ZipCode'], 
                                        zip_data['Claim_Frequency'])
        chi2, p_freq, dof, expected = chi2_contingency(contingency_table)
        results['Claim_Frequency'] = {
            'test': 'Chi-squared',
            'statistic': chi2,
            'p_value': p_freq,
            'reject_null': p_freq < 0.05
        }
        
        # 2. Margin Test (ANOVA)
        margin_groups = [zip_data[zip_data['ZipCode'] == z]['Margin'] 
                        for z in top_zips]
        
        if len(margin_groups) > 1:
            f_stat, p_margin = f_oneway(*margin_groups)
            results['Margin'] = {
                'test': 'ANOVA',
                'statistic': f_stat,
                'p_value': p_margin,
                'reject_null': p_margin < 0.05
            }
        
        # Calculate zip code statistics
        zip_stats = zip_data.groupby('ZipCode').agg({
            'Claim_Frequency': ['mean', 'count'],
            'Claim_Severity': 'mean',
            'Margin': 'mean',
            'Loss_Ratio': 'mean'
        }).round(4)
        
        self.results['zipcode'] = {
            'test_results': results,
            'zip_stats': zip_stats,
            'top_n': top_n
        }
        
        return results, zip_stats
    
    def test_gender_risk_differences(self):
        """
        H₀: There is no significant risk difference between Women and Men
        """
        print("\n" + "=" * 80)
        print("HYPOTHESIS 3: Gender Risk Differences")
        print("=" * 80)
        
        results = {}
        
        # Filter data for binary gender classification (simplified)
        gender_data = self.data[self.data['Gender'].isin(['Male', 'Female'])]
        
        # 1. Claim Frequency Test (Chi-squared)
        contingency_table = pd.crosstab(gender_data['Gender'], 
                                        gender_data['Claim_Frequency'])
        chi2, p_freq, dof, expected = chi2_contingency(contingency_table)
        results['Claim_Frequency'] = {
            'test': 'Chi-squared',
            'statistic': chi2,
            'p_value': p_freq,
            'reject_null': p_freq < 0.05
        }
        
        # 2. Claim Severity Test (T-test)
        male_severity = gender_data[gender_data['Gender'] == 'Male']['Claim_Severity'].dropna()
        female_severity = gender_data[gender_data['Gender'] == 'Female']['Claim_Severity'].dropna()
        
        if len(male_severity) > 1 and len(female_severity) > 1:
            t_stat, p_sev = ttest_ind(male_severity, female_severity, equal_var=False)
            results['Claim_Severity'] = {
                'test': 'Welch\'s t-test',
                'statistic': t_stat,
                'p_value': p_sev,
                'reject_null': p_sev < 0.05
            }
        
        # 3. Margin Test (T-test)
        male_margin = gender_data[gender_data['Gender'] == 'Male']['Margin']
        female_margin = gender_data[gender_data['Gender'] == 'Female']['Margin']
        
        t_stat, p_margin = ttest_ind(male_margin, female_margin, equal_var=False)
        results['Margin'] = {
            'test': 'Welch\'s t-test',
            'statistic': t_stat,
            'p_value': p_margin,
            'reject_null': p_margin < 0.05
        }
        
        # Calculate gender statistics
        gender_stats = gender_data.groupby('Gender').agg({
            'Claim_Frequency': 'mean',
            'Claim_Severity': 'mean',
            'Margin': 'mean',
            'Loss_Ratio': 'mean',
            'TotalPremium': 'mean'
        }).round(4)
        
        # Calculate percentage differences
        if len(gender_stats) == 2:
            male_vals = gender_stats.loc['Male']
            female_vals = gender_stats.loc['Female']
            pct_diff = ((male_vals - female_vals) / female_vals * 100).round(2)
            gender_stats.loc['% Difference'] = pct_diff
        
        self.results['gender'] = {
            'test_results': results,
            'gender_stats': gender_stats
        }
        
        # Visualization
        self._plot_gender_comparisons(gender_data)
        
        return results, gender_stats
    
    def _plot_province_comparisons(self, province_stats):
        """Visualize province comparisons"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Claim Frequency by Province
        province_stats['Claim_Frequency'].sort_values().plot(
            kind='barh', ax=axes[0, 0], color='skyblue'
        )
        axes[0, 0].set_title('Claim Frequency by Province')
        axes[0, 0].set_xlabel('Claim Frequency')
        axes[0, 0].axvline(x=province_stats['Claim_Frequency'].mean(), 
                          color='red', linestyle='--', alpha=0.7)
        
        # Plot 2: Average Margin by Province
        province_stats['Margin'].sort_values().plot(
            kind='barh', ax=axes[0, 1], color='lightgreen'
        )
        axes[0, 1].set_title('Average Margin by Province')
        axes[0, 1].set_xlabel('Margin (Premium - Claims)')
        axes[0, 1].axvline(x=province_stats['Margin'].mean(), 
                          color='red', linestyle='--', alpha=0.7)
        
        # Plot 3: Loss Ratio by Province
        province_stats['Loss_Ratio'].sort_values().plot(
            kind='barh', ax=axes[1, 0], color='salmon'
        )
        axes[1, 0].set_title('Loss Ratio by Province')
        axes[1, 0].set_xlabel('Loss Ratio (Claims/Premium)')
        axes[1, 0].axvline(x=province_stats['Loss_Ratio'].mean(), 
                          color='red', linestyle='--', alpha=0.7)
        
        # Plot 4: Claim Severity by Province
        province_stats['Claim_Severity'].sort_values().plot(
            kind='barh', ax=axes[1, 1], color='gold'
        )
        axes[1, 1].set_title('Average Claim Severity by Province')
        axes[1, 1].set_xlabel('Claim Severity')
        axes[1, 1].axvline(x=province_stats['Claim_Severity'].mean(), 
                          color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('output/province_comparisons.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_gender_comparisons(self, gender_data):
        """Visualize gender comparisons"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Claim Frequency Distribution
        claim_freq_by_gender = pd.crosstab(gender_data['Gender'], 
                                           gender_data['Claim_Frequency'], 
                                           normalize='index')
        claim_freq_by_gender.plot(kind='bar', stacked=True, ax=axes[0, 0], 
                                  color=['lightblue', 'coral'])
        axes[0, 0].set_title('Claim Frequency Distribution by Gender')
        axes[0, 0].set_ylabel('Proportion')
        axes[0, 0].legend(['No Claim', 'Claim'])
        
        # Plot 2: Margin Distribution
        gender_data.boxplot(column='Margin', by='Gender', ax=axes[0, 1])
        axes[0, 1].set_title('Margin Distribution by Gender')
        axes[0, 1].set_ylabel('Margin')
        
        # Plot 3: Premium Distribution
        gender_data.boxplot(column='TotalPremium', by='Gender', ax=axes[1, 0])
        axes[1, 0].set_title('Premium Distribution by Gender')
        axes[1, 0].set_ylabel('Premium')
        
        # Plot 4: Claim Severity Distribution (only claims)
        severity_data = gender_data.dropna(subset=['Claim_Severity'])
        severity_data.boxplot(column='Claim_Severity', by='Gender', ax=axes[1, 1])
        axes[1, 1].set_title('Claim Severity Distribution by Gender')
        axes[1, 1].set_ylabel('Claim Severity')
        
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig('output/gender_comparisons.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_business_report(self):
        """Generate comprehensive business report from hypothesis testing"""
        report = []
        report.append("=" * 80)
        report.append("INSURANCE RISK ANALYSIS - HYPOTHESIS TESTING REPORT")
        report.append("=" * 80)
        
        for hypothesis_name, results in self.results.items():
            report.append(f"\n{hypothesis_name.upper()} ANALYSIS")
            report.append("-" * 40)
            
            test_results = results['test_results']
            for metric, test_info in test_results.items():
                reject_text = "REJECT" if test_info['reject_null'] else "FAIL TO REJECT"
                report.append(f"{metric}:")
                report.append(f"  Test: {test_info['test']}")
                report.append(f"  p-value: {test_info['p_value']:.6f}")
                report.append(f"  Decision: {reject_text} null hypothesis")
                report.append("")
            
            # Add business interpretation
            if hypothesis_name == 'province':
                province_stats = results['province_stats']
                highest_loss = province_stats['Loss_Ratio'].idxmax()
                lowest_loss = province_stats['Loss_Ratio'].idxmin()
                loss_diff = province_stats.loc[highest_loss, 'Loss_Ratio'] - \
                           province_stats.loc[lowest_loss, 'Loss_Ratio']
                
                report.append("BUSINESS INTERPRETATION:")
                report.append(f"  • {highest_loss} has the highest loss ratio")
                report.append(f"  • {lowest_loss} has the lowest loss ratio")
                report.append(f"  • Difference: {loss_diff:.2%}")
                report.append("  • RECOMMENDATION: Consider regional premium adjustments")
                
            elif hypothesis_name == 'gender':
                gender_stats = results['gender_stats']
                if 'Male' in gender_stats.index and 'Female' in gender_stats.index:
                    freq_diff = gender_stats.loc['Male', 'Claim_Frequency'] - \
                               gender_stats.loc['Female', 'Claim_Frequency']
                    margin_diff = gender_stats.loc['Male', 'Margin'] - \
                                 gender_stats.loc['Female', 'Margin']
                    
                    report.append("BUSINESS INTERPRETATION:")
                    report.append(f"  • Claim frequency difference: {freq_diff:.2%}")
                    report.append(f"  • Margin difference: R{margin_diff:.2f}")
                    if test_results.get('Claim_Frequency', {}).get('reject_null', False):
                        report.append("  • RECOMMENDATION: Gender-based risk assessment may be warranted")
                    else:
                        report.append("  • RECOMMENDATION: No significant gender difference found")
        
        # Save report
        report_text = "\n".join(report)
        with open('output/hypothesis_testing_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        return report_text

def main():
    """Main execution function"""
    print("Starting Insurance Hypothesis Testing...")
    
    # Initialize testing framework
    tester = InsuranceHypothesisTesting()
    
    # Run all hypothesis tests
    print("\nRunning Hypothesis Tests...")
    print("-" * 80)
    
    # Test 1: Province differences
    province_results, province_stats = tester.test_province_risk_differences()
    
    # Test 2: Zip code differences
    zip_results, zip_stats = tester.test_zipcode_risk_differences(top_n=10)
    
    # Test 3: Gender differences
    gender_results, gender_stats = tester.test_gender_risk_differences()
    
    # Generate comprehensive report
    print("\n" + "=" * 80)
    print("GENERATING BUSINESS REPORT...")
    print("=" * 80)
    report = tester.generate_business_report()
    
    print("\n" + "=" * 80)
    print("HYPOTHESIS TESTING COMPLETED")
    print("=" * 80)
    print("Results saved to:")
    print("  - output/hypothesis_testing_report.txt")
    print("  - output/province_comparisons.png")
    print("  - output/gender_comparisons.png")
    
    return tester

if __name__ == "__main__":
    tester = main()"""
Task 3: Statistical Hypothesis Testing for Insurance Risk Analysis
Author: Senior Data Scientist
Date: 2024
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
import statsmodels.api as sm
from statsmodels.stats.power import TTestIndPower
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class InsuranceHypothesisTesting:
    """Class for conducting A/B hypothesis testing on insurance data"""
    
    def __init__(self, data_path='data/insurance_data.csv'):
        """
        Initialize the hypothesis testing framework
        
        Parameters:
        -----------
        data_path : str
            Path to the insurance data file
        """
        self.data = pd.read_csv(data_path)
        self.results = {}
        
        # Calculate derived metrics
        self._calculate_metrics()
        
    def _calculate_metrics(self):
        """Calculate necessary metrics for hypothesis testing"""
        # Claim Frequency (binary: 1 if claims > 0, else 0)
        self.data['Claim_Frequency'] = (self.data['TotalClaims'] > 0).astype(int)
        
        # Claim Severity (average claim amount when claim occurs)
        self.data['Claim_Severity'] = self.data.apply(
            lambda x: x['TotalClaims'] if x['TotalClaims'] > 0 else np.nan, axis=1
        )
        
        # Margin
        self.data['Margin'] = self.data['TotalPremium'] - self.data['TotalClaims']
        
        # Loss Ratio
        self.data['Loss_Ratio'] = self.data['TotalClaims'] / self.data['TotalPremium']
        
    def test_province_risk_differences(self):
        """
        H₀: There are no risk differences across provinces
        
        Tests:
        1. Claim Frequency differences (Chi-squared test)
        2. Claim Severity differences (ANOVA)
        3. Margin differences (ANOVA)
        """
        print("=" * 80)
        print("HYPOTHESIS 1: Province Risk Differences")
        print("=" * 80)
        
        results = {}
        
        # 1. Claim Frequency Test (Chi-squared)
        contingency_table = pd.crosstab(self.data['Province'], 
                                        self.data['Claim_Frequency'])
        chi2, p_freq, dof, expected = chi2_contingency(contingency_table)
        results['Claim_Frequency'] = {
            'test': 'Chi-squared',
            'statistic': chi2,
            'p_value': p_freq,
            'reject_null': p_freq < 0.05
        }
        
        # 2. Claim Severity Test (ANOVA)
        # Filter only claims that occurred
        severity_data = self.data.dropna(subset=['Claim_Severity'])
        provinces = severity_data['Province'].unique()
        
        # Prepare data for ANOVA
        severity_groups = [severity_data[severity_data['Province'] == p]['Claim_Severity'] 
                          for p in provinces if len(severity_data[severity_data['Province'] == p]) > 1]
        
        if len(severity_groups) > 1:
            f_stat, p_sev = f_oneway(*severity_groups)
            results['Claim_Severity'] = {
                'test': 'ANOVA',
                'statistic': f_stat,
                'p_value': p_sev,
                'reject_null': p_sev < 0.05
            }
        
        # 3. Margin Test (ANOVA)
        margin_groups = [self.data[self.data['Province'] == p]['Margin'] 
                        for p in self.data['Province'].unique() 
                        if len(self.data[self.data['Province'] == p]) > 1]
        
        if len(margin_groups) > 1:
            f_stat, p_margin = f_oneway(*margin_groups)
            results['Margin'] = {
                'test': 'ANOVA',
                'statistic': f_stat,
                'p_value': p_margin,
                'reject_null': p_margin < 0.05
            }
        
        # Calculate effect sizes and business metrics
        province_stats = self.data.groupby('Province').agg({
            'Claim_Frequency': 'mean',
            'Claim_Severity': 'mean',
            'Margin': 'mean',
            'Loss_Ratio': 'mean'
        }).round(4)
        
        self.results['province'] = {
            'test_results': results,
            'province_stats': province_stats
        }
        
        # Visualization
        self._plot_province_comparisons(province_stats)
        
        return results, province_stats
    
    def test_zipcode_risk_differences(self, top_n=10):
        """
        H₀: There are no risk differences between zip codes
        
        Note: Testing all zip codes might be too granular.
        We'll test the top N zip codes by policy count.
        """
        print("\n" + "=" * 80)
        print("HYPOTHESIS 2: Zip Code Risk Differences")
        print("=" * 80)
        
        # Get top N zip codes by policy count
        zip_counts = self.data['ZipCode'].value_counts().head(top_n)
        top_zips = zip_counts.index.tolist()
        
        # Filter data for top zip codes
        zip_data = self.data[self.data['ZipCode'].isin(top_zips)]
        
        results = {}
        
        # 1. Claim Frequency Test (Chi-squared)
        contingency_table = pd.crosstab(zip_data['ZipCode'], 
                                        zip_data['Claim_Frequency'])
        chi2, p_freq, dof, expected = chi2_contingency(contingency_table)
        results['Claim_Frequency'] = {
            'test': 'Chi-squared',
            'statistic': chi2,
            'p_value': p_freq,
            'reject_null': p_freq < 0.05
        }
        
        # 2. Margin Test (ANOVA)
        margin_groups = [zip_data[zip_data['ZipCode'] == z]['Margin'] 
                        for z in top_zips]
        
        if len(margin_groups) > 1:
            f_stat, p_margin = f_oneway(*margin_groups)
            results['Margin'] = {
                'test': 'ANOVA',
                'statistic': f_stat,
                'p_value': p_margin,
                'reject_null': p_margin < 0.05
            }
        
        # Calculate zip code statistics
        zip_stats = zip_data.groupby('ZipCode').agg({
            'Claim_Frequency': ['mean', 'count'],
            'Claim_Severity': 'mean',
            'Margin': 'mean',
            'Loss_Ratio': 'mean'
        }).round(4)
        
        self.results['zipcode'] = {
            'test_results': results,
            'zip_stats': zip_stats,
            'top_n': top_n
        }
        
        return results, zip_stats
    
    def test_gender_risk_differences(self):
        """
        H₀: There is no significant risk difference between Women and Men
        """
        print("\n" + "=" * 80)
        print("HYPOTHESIS 3: Gender Risk Differences")
        print("=" * 80)
        
        results = {}
        
        # Filter data for binary gender classification (simplified)
        gender_data = self.data[self.data['Gender'].isin(['Male', 'Female'])]
        
        # 1. Claim Frequency Test (Chi-squared)
        contingency_table = pd.crosstab(gender_data['Gender'], 
                                        gender_data['Claim_Frequency'])
        chi2, p_freq, dof, expected = chi2_contingency(contingency_table)
        results['Claim_Frequency'] = {
            'test': 'Chi-squared',
            'statistic': chi2,
            'p_value': p_freq,
            'reject_null': p_freq < 0.05
        }
        
        # 2. Claim Severity Test (T-test)
        male_severity = gender_data[gender_data['Gender'] == 'Male']['Claim_Severity'].dropna()
        female_severity = gender_data[gender_data['Gender'] == 'Female']['Claim_Severity'].dropna()
        
        if len(male_severity) > 1 and len(female_severity) > 1:
            t_stat, p_sev = ttest_ind(male_severity, female_severity, equal_var=False)
            results['Claim_Severity'] = {
                'test': 'Welch\'s t-test',
                'statistic': t_stat,
                'p_value': p_sev,
                'reject_null': p_sev < 0.05
            }
        
        # 3. Margin Test (T-test)
        male_margin = gender_data[gender_data['Gender'] == 'Male']['Margin']
        female_margin = gender_data[gender_data['Gender'] == 'Female']['Margin']
        
        t_stat, p_margin = ttest_ind(male_margin, female_margin, equal_var=False)
        results['Margin'] = {
            'test': 'Welch\'s t-test',
            'statistic': t_stat,
            'p_value': p_margin,
            'reject_null': p_margin < 0.05
        }
        
        # Calculate gender statistics
        gender_stats = gender_data.groupby('Gender').agg({
            'Claim_Frequency': 'mean',
            'Claim_Severity': 'mean',
            'Margin': 'mean',
            'Loss_Ratio': 'mean',
            'TotalPremium': 'mean'
        }).round(4)
        
        # Calculate percentage differences
        if len(gender_stats) == 2:
            male_vals = gender_stats.loc['Male']
            female_vals = gender_stats.loc['Female']
            pct_diff = ((male_vals - female_vals) / female_vals * 100).round(2)
            gender_stats.loc['% Difference'] = pct_diff
        
        self.results['gender'] = {
            'test_results': results,
            'gender_stats': gender_stats
        }
        
        # Visualization
        self._plot_gender_comparisons(gender_data)
        
        return results, gender_stats
    
    def _plot_province_comparisons(self, province_stats):
        """Visualize province comparisons"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Claim Frequency by Province
        province_stats['Claim_Frequency'].sort_values().plot(
            kind='barh', ax=axes[0, 0], color='skyblue'
        )
        axes[0, 0].set_title('Claim Frequency by Province')
        axes[0, 0].set_xlabel('Claim Frequency')
        axes[0, 0].axvline(x=province_stats['Claim_Frequency'].mean(), 
                          color='red', linestyle='--', alpha=0.7)
        
        # Plot 2: Average Margin by Province
        province_stats['Margin'].sort_values().plot(
            kind='barh', ax=axes[0, 1], color='lightgreen'
        )
        axes[0, 1].set_title('Average Margin by Province')
        axes[0, 1].set_xlabel('Margin (Premium - Claims)')
        axes[0, 1].axvline(x=province_stats['Margin'].mean(), 
                          color='red', linestyle='--', alpha=0.7)
        
        # Plot 3: Loss Ratio by Province
        province_stats['Loss_Ratio'].sort_values().plot(
            kind='barh', ax=axes[1, 0], color='salmon'
        )
        axes[1, 0].set_title('Loss Ratio by Province')
        axes[1, 0].set_xlabel('Loss Ratio (Claims/Premium)')
        axes[1, 0].axvline(x=province_stats['Loss_Ratio'].mean(), 
                          color='red', linestyle='--', alpha=0.7)
        
        # Plot 4: Claim Severity by Province
        province_stats['Claim_Severity'].sort_values().plot(
            kind='barh', ax=axes[1, 1], color='gold'
        )
        axes[1, 1].set_title('Average Claim Severity by Province')
        axes[1, 1].set_xlabel('Claim Severity')
        axes[1, 1].axvline(x=province_stats['Claim_Severity'].mean(), 
                          color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('output/province_comparisons.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_gender_comparisons(self, gender_data):
        """Visualize gender comparisons"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Claim Frequency Distribution
        claim_freq_by_gender = pd.crosstab(gender_data['Gender'], 
                                           gender_data['Claim_Frequency'], 
                                           normalize='index')
        claim_freq_by_gender.plot(kind='bar', stacked=True, ax=axes[0, 0], 
                                  color=['lightblue', 'coral'])
        axes[0, 0].set_title('Claim Frequency Distribution by Gender')
        axes[0, 0].set_ylabel('Proportion')
        axes[0, 0].legend(['No Claim', 'Claim'])
        
        # Plot 2: Margin Distribution
        gender_data.boxplot(column='Margin', by='Gender', ax=axes[0, 1])
        axes[0, 1].set_title('Margin Distribution by Gender')
        axes[0, 1].set_ylabel('Margin')
        
        # Plot 3: Premium Distribution
        gender_data.boxplot(column='TotalPremium', by='Gender', ax=axes[1, 0])
        axes[1, 0].set_title('Premium Distribution by Gender')
        axes[1, 0].set_ylabel('Premium')
        
        # Plot 4: Claim Severity Distribution (only claims)
        severity_data = gender_data.dropna(subset=['Claim_Severity'])
        severity_data.boxplot(column='Claim_Severity', by='Gender', ax=axes[1, 1])
        axes[1, 1].set_title('Claim Severity Distribution by Gender')
        axes[1, 1].set_ylabel('Claim Severity')
        
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig('output/gender_comparisons.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_business_report(self):
        """Generate comprehensive business report from hypothesis testing"""
        report = []
        report.append("=" * 80)
        report.append("INSURANCE RISK ANALYSIS - HYPOTHESIS TESTING REPORT")
        report.append("=" * 80)
        
        for hypothesis_name, results in self.results.items():
            report.append(f"\n{hypothesis_name.upper()} ANALYSIS")
            report.append("-" * 40)
            
            test_results = results['test_results']
            for metric, test_info in test_results.items():
                reject_text = "REJECT" if test_info['reject_null'] else "FAIL TO REJECT"
                report.append(f"{metric}:")
                report.append(f"  Test: {test_info['test']}")
                report.append(f"  p-value: {test_info['p_value']:.6f}")
                report.append(f"  Decision: {reject_text} null hypothesis")
                report.append("")
            
            # Add business interpretation
            if hypothesis_name == 'province':
                province_stats = results['province_stats']
                highest_loss = province_stats['Loss_Ratio'].idxmax()
                lowest_loss = province_stats['Loss_Ratio'].idxmin()
                loss_diff = province_stats.loc[highest_loss, 'Loss_Ratio'] - \
                           province_stats.loc[lowest_loss, 'Loss_Ratio']
                
                report.append("BUSINESS INTERPRETATION:")
                report.append(f"  • {highest_loss} has the highest loss ratio")
                report.append(f"  • {lowest_loss} has the lowest loss ratio")
                report.append(f"  • Difference: {loss_diff:.2%}")
                report.append("  • RECOMMENDATION: Consider regional premium adjustments")
                
            elif hypothesis_name == 'gender':
                gender_stats = results['gender_stats']
                if 'Male' in gender_stats.index and 'Female' in gender_stats.index:
                    freq_diff = gender_stats.loc['Male', 'Claim_Frequency'] - \
                               gender_stats.loc['Female', 'Claim_Frequency']
                    margin_diff = gender_stats.loc['Male', 'Margin'] - \
                                 gender_stats.loc['Female', 'Margin']
                    
                    report.append("BUSINESS INTERPRETATION:")
                    report.append(f"  • Claim frequency difference: {freq_diff:.2%}")
                    report.append(f"  • Margin difference: R{margin_diff:.2f}")
                    if test_results.get('Claim_Frequency', {}).get('reject_null', False):
                        report.append("  • RECOMMENDATION: Gender-based risk assessment may be warranted")
                    else:
                        report.append("  • RECOMMENDATION: No significant gender difference found")
        
        # Save report
        report_text = "\n".join(report)
        with open('output/hypothesis_testing_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        return report_text

def main():
    """Main execution function"""
    print("Starting Insurance Hypothesis Testing...")
    
    # Initialize testing framework
    tester = InsuranceHypothesisTesting()
    
    # Run all hypothesis tests
    print("\nRunning Hypothesis Tests...")
    print("-" * 80)
    
    # Test 1: Province differences
    province_results, province_stats = tester.test_province_risk_differences()
    
    # Test 2: Zip code differences
    zip_results, zip_stats = tester.test_zipcode_risk_differences(top_n=10)
    
    # Test 3: Gender differences
    gender_results, gender_stats = tester.test_gender_risk_differences()
    
    # Generate comprehensive report
    print("\n" + "=" * 80)
    print("GENERATING BUSINESS REPORT...")
    print("=" * 80)
    report = tester.generate_business_report()
    
    print("\n" + "=" * 80)
    print("HYPOTHESIS TESTING COMPLETED")
    print("=" * 80)
    print("Results saved to:")
    print("  - output/hypothesis_testing_report.txt")
    print("  - output/province_comparisons.png")
    print("  - output/gender_comparisons.png")
    
    return tester

if __name__ == "__main__":
    tester = main()