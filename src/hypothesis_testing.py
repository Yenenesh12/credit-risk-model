# task3_hypothesis_testing.py
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

class InsuranceHypothesisTesting:
    def __init__(self, data_path=r'C:\Users\admin\insurance-risk-analysis-week3\data\raw\insurance_data.csv'):
        """Initialize with insurance data"""
        self.df = pd.read_csv(data_path)
        print(f"Data loaded with shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        
    def prepare_metrics(self):
        """Prepare metrics for hypothesis testing"""
        # Ensure key numeric columns are actually numeric
        numeric_cols = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate', 'SumInsured', 
                        'CalculatedPremiumPerTerm', 'RegistrationYear', 'NumberOfDoors']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Calculate derived metrics if not already present
        if 'has_claim' not in self.df.columns:
            self.df['has_claim'] = (self.df['TotalClaims'] > 0).astype(int)
        
        if 'claim_severity' not in self.df.columns:
            self.df['claim_severity'] = self.df['TotalClaims'].where(self.df['TotalClaims'] > 0)
        
        if 'margin' not in self.df.columns:
            self.df['margin'] = self.df['TotalPremium'] - self.df['TotalClaims']
        
        if 'loss_ratio' not in self.df.columns:
            self.df['loss_ratio'] = self.df['TotalClaims'] / self.df['TotalPremium']
            self.df['loss_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    def test_province_risk_differences(self):
        """Test H₀: There are no risk differences across provinces"""
        print("\n" + "="*60)
        print("HYPOTHESIS 1: Risk Differences Across Provinces")
        print("="*60)
        
        if 'Province' not in self.df.columns:
            print("Province column not found. Skipping test.")
            return None, None
        
        province_claim_freq = self.df.groupby('Province')['has_claim'].mean().sort_values(ascending=False)
        province_claim_severity = self.df.groupby('Province')['claim_severity'].mean().sort_values(ascending=False)
        
        groups = [group['claim_severity'].dropna().values 
                 for name, group in self.df.groupby('Province') if len(group['claim_severity'].dropna()) > 0]
        
        if len(groups) < 2:
            print("Not enough provinces with claim severity data.")
            return province_claim_freq, province_claim_severity
        
        f_stat, p_value_severity = f_oneway(*groups)
        
        contingency_table = pd.crosstab(self.df['Province'], self.df['has_claim'])
        chi2, p_value_frequency, dof, expected = chi2_contingency(contingency_table)
        
        print(f"\nClaim Frequency by Province:")
        print(province_claim_freq.round(4))
        
        print(f"\nClaim Severity by Province:")
        print(province_claim_severity.round(2))
        
        print(f"\nStatistical Tests:")
        print(f"ANOVA Test for Claim Severity: F-statistic = {f_stat:.4f}, p-value = {p_value_severity:.6f}")
        print(f"Chi-square Test for Claim Frequency: χ² = {chi2:.4f}, p-value = {p_value_frequency:.6f}")
        
        alpha = 0.05
        if p_value_severity < alpha or p_value_frequency < alpha:
            print(f"\nDECISION: REJECT the null hypothesis (p < {alpha})")
            print("There are statistically significant risk differences across provinces.")
            
            highest_risk = province_claim_freq.idxmax()
            lowest_risk = province_claim_freq.idxmin()
            risk_diff = (province_claim_freq.max() - province_claim_freq.min()) * 100
            
            print(f"\nBUSINESS IMPLICATION:")
            print(f"• {highest_risk} has {risk_diff:.1f}% higher claim frequency than {lowest_risk}")
            print(f"• Regional premium adjustments are warranted")
            print(f"• Consider province-specific pricing strategies")
        else:
            print(f"\nDECISION: FAIL TO REJECT the null hypothesis")
            print("No statistically significant risk differences across provinces.")
        
        return province_claim_freq, province_claim_severity
    
    def test_zipcode_risk_differences(self):
        """Test H₀: There are no risk differences between zip codes (using 'Make' as proxy)"""
        print("\n" + "="*60)
        print("HYPOTHESIS 2: Risk Differences Between Zip Codes")
        print("="*60)
        
        proxy_column = 'Make'
        if proxy_column not in self.df.columns:
            print(f"Proxy column '{proxy_column}' not found. Skipping test.")
            return []
            
        top_categories = self.df[proxy_column].value_counts().nlargest(2).index.tolist()
        
        if len(top_categories) < 2:
            print("Not enough categories for comparison.")
            return top_categories
        
        group_a = self.df[self.df[proxy_column] == top_categories[0]]
        group_b = self.df[self.df[proxy_column] == top_categories[1]]
        
        print(f"\nTesting between: {top_categories[0]} (Group A) vs {top_categories[1]} (Group B)")
        print(f"Group A size: {len(group_a)}, Group B size: {len(group_b)}")
        
        # Balance check: only on numeric columns (excluding outcome and proxy)
        print("\nBalance Check (p-values for differences in numeric features):")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = {proxy_column, 'has_claim', 'claim_severity', 'margin', 'loss_ratio', 'PolicyID'}
        test_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        for col in test_cols:
            a_vals = group_a[col].dropna()
            b_vals = group_b[col].dropna()
            if len(a_vals) == 0 or len(b_vals) == 0:
                continue
            try:
                stat, p_val = ttest_ind(a_vals, b_vals, equal_var=False)
                print(f"  {col}: p-value = {p_val:.6f}")
            except Exception as e:
                print(f"  {col}: t-test failed ({e})")
        
        # Claim frequency
        freq_a = group_a['has_claim'].mean()
        freq_b = group_b['has_claim'].mean()
        
        cont_table = pd.DataFrame({
            'Group_A': [group_a['has_claim'].sum(), len(group_a) - group_a['has_claim'].sum()],
            'Group_B': [group_b['has_claim'].sum(), len(group_b) - group_b['has_claim'].sum()]
        }, index=['Claims', 'No Claims'])
        
        chi2, p_value_freq, dof, expected = chi2_contingency(cont_table)
        
        print(f"\nClaim Frequency Results:")
        print(f"Group A ({top_categories[0]}): {freq_a:.4f}")
        print(f"Group B ({top_categories[1]}): {freq_b:.4f}")
        print(f"Difference: {(freq_b - freq_a)*100:.2f}%")
        print(f"Chi-square Test p-value: {p_value_freq:.6f}")
        
        # Claim severity
        sev_a = group_a['claim_severity'].dropna()
        sev_b = group_b['claim_severity'].dropna()
        severity_a = sev_a.mean()
        severity_b = sev_b.mean()
        
        if len(sev_a) > 0 and len(sev_b) > 0:
            t_stat, p_value_sev = ttest_ind(sev_a, sev_b, equal_var=False)
        else:
            p_value_sev = 1.0
        
        print(f"\nClaim Severity Results:")
        print(f"Group A ({top_categories[0]}): R{severity_a:.2f}")
        print(f"Group B ({top_categories[1]}): R{severity_b:.2f}")
        print(f"Difference: R{severity_b - severity_a:.2f}")
        print(f"T-test p-value: {p_value_sev:.6f}")
        
        alpha = 0.05
        if p_value_freq < alpha or (len(sev_a) > 0 and len(sev_b) > 0 and p_value_sev < alpha):
            print(f"\nDECISION: REJECT the null hypothesis (p < {alpha})")
            print("There are statistically significant risk differences.")
            
            print(f"\nBUSINESS IMPLICATION:")
            if freq_b > freq_a:
                direction = "higher"
            else:
                direction = "lower"
            print(f"• {top_categories[1]} has {abs(freq_b - freq_a)*100:.1f}% {direction} claim frequency")
            print(f"• Consider location-based risk assessment for pricing")
            print(f"• Implement geospatial risk factors in underwriting")
        else:
            print(f"\nDECISION: FAIL TO REJECT the null hypothesis")
            print("No statistically significant risk differences.")
        
        return top_categories
    
    def test_zipcode_margin_differences(self):
        """Test H₀: There is no significant margin difference between zip codes (using 'Make' as proxy)"""
        print("\n" + "="*60)
        print("HYPOTHESIS 3: Margin Differences Between Zip Codes")
        print("="*60)
        
        proxy_column = 'Make'
        if proxy_column not in self.df.columns:
            print(f"Proxy column '{proxy_column}' not found. Skipping test.")
            return None, None, 1.0
        
        top_categories = self.df[proxy_column].value_counts().nlargest(2).index.tolist()
        
        if len(top_categories) < 2:
            print("Not enough categories for comparison.")
            return None, None, 1.0
        
        group_a = self.df[self.df[proxy_column] == top_categories[0]]
        group_b = self.df[self.df[proxy_column] == top_categories[1]]
        
        margin_a = group_a['margin'].mean()
        margin_b = group_b['margin'].mean()
        
        marg_a_clean = group_a['margin'].dropna()
        marg_b_clean = group_b['margin'].dropna()
        
        if len(marg_a_clean) > 0 and len(marg_b_clean) > 0:
            t_stat, p_value = ttest_ind(marg_a_clean, marg_b_clean, equal_var=False)
        else:
            p_value = 1.0
        
        print(f"\nMargin Analysis:")
        print(f"Group A ({top_categories[0]}): Average Margin = R{margin_a:.2f}")
        print(f"Group B ({top_categories[1]}): Average Margin = R{margin_b:.2f}")
        print(f"Difference: R{margin_b - margin_a:.2f}")
        print(f"T-test p-value: {p_value:.6f}")
        
        loss_ratio_a = group_a['loss_ratio'].mean()
        loss_ratio_b = group_b['loss_ratio'].mean()
        
        print(f"\nLoss Ratio Comparison:")
        print(f"Group A Loss Ratio: {loss_ratio_a:.4f}")
        print(f"Group B Loss Ratio: {loss_ratio_b:.4f}")
        
        alpha = 0.05
        if p_value < alpha:
            print(f"\nDECISION: REJECT the null hypothesis (p < {alpha})")
            print("There are statistically significant margin differences.")
            
            if margin_a > margin_b:
                profitable_group = top_categories[0]
                diff = margin_a - margin_b
            else:
                profitable_group = top_categories[1]
                diff = margin_b - margin_a
            
            print(f"\nBUSINESS IMPLICATION:")
            print(f"• {profitable_group} is R{diff:.2f} more profitable per policy")
            print(f"• Consider adjusting pricing strategy for less profitable segments")
            print(f"• Review underwriting criteria for low-margin areas")
        else:
            print(f"\nDECISION: FAIL TO REJECT the null hypothesis")
            print("No statistically significant margin differences.")
        
        return margin_a, margin_b, p_value
    
    def test_gender_risk_differences(self):
        """Test H₀: There is no significant risk difference between Women and Men"""
        print("\n" + "="*60)
        print("HYPOTHESIS 4: Risk Differences Between Genders")
        print("="*60)
        
        if 'Gender' not in self.df.columns:
            print("Gender column not found. Skipping test.")
            return None
        
        gender_df = self.df[self.df['Gender'].isin(['Male', 'Female'])]
        
        if gender_df.empty:
            print("No 'Male'/'Female' entries found in Gender column.")
            return None
        
        male_count = len(gender_df[gender_df['Gender'] == 'Male'])
        female_count = len(gender_df[gender_df['Gender'] == 'Female'])
        
        print(f"\nSample Sizes:")
        print(f"Male: {male_count} policies")
        print(f"Female: {female_count} policies")
        
        gender_metrics = gender_df.groupby('Gender').agg({
            'has_claim': 'mean',
            'claim_severity': 'mean',
            'TotalPremium': 'mean',
            'TotalClaims': 'mean',
            'margin': 'mean',
            'loss_ratio': 'mean'
        }).round(4)
        
        print(f"\nKey Metrics by Gender:")
        print(gender_metrics)
        
        male_data = gender_df[gender_df['Gender'] == 'Male']
        female_data = gender_df[gender_df['Gender'] == 'Female']
        
        # Claim frequency: Chi-square
        cont_table = pd.crosstab(gender_df['Gender'], gender_df['has_claim'])
        chi2, p_value_freq, dof, expected = chi2_contingency(cont_table)
        
        # Claim severity: T-test
        sev_male = male_data['claim_severity'].dropna()
        sev_female = female_data['claim_severity'].dropna()
        if len(sev_male) > 0 and len(sev_female) > 0:
            t_stat_sev, p_value_sev = ttest_ind(sev_male, sev_female, equal_var=False)
        else:
            p_value_sev = 1.0
        
        # Margin: T-test
        marg_male = male_data['margin'].dropna()
        marg_female = female_data['margin'].dropna()
        if len(marg_male) > 0 and len(marg_female) > 0:
            t_stat_margin, p_value_margin = ttest_ind(marg_male, marg_female, equal_var=False)
        else:
            p_value_margin = 1.0
        
        print(f"\nStatistical Tests:")
        print(f"Claim Frequency (Chi-square): χ² = {chi2:.4f}, p-value = {p_value_freq:.6f}")
        print(f"Claim Severity (T-test): p-value = {p_value_sev:.6f}")
        print(f"Margin (T-test): p-value = {p_value_margin:.6f}")
        
        alpha = 0.05
        significant = p_value_freq < alpha or p_value_sev < alpha
        
        if significant:
            print(f"\nDECISION: REJECT the null hypothesis (p < {alpha})")
            print("There are statistically significant risk differences between genders.")
            
            freq_male = gender_metrics.loc['Male', 'has_claim']
            freq_female = gender_metrics.loc['Female', 'has_claim']
            
            print(f"\nBUSINESS IMPLICATION:")
            if freq_male > freq_female:
                print(f"• Male policyholders have {(freq_male - freq_female)*100:.1f}% higher claim frequency")
            else:
                print(f"• Female policyholders have {(freq_female - freq_male)*100:.1f}% higher claim frequency")
            print(f"• Consider gender as a rating factor if legally permissible")
            print(f"• Ensure compliance with anti-discrimination regulations")
        else:
            print(f"\nDECISION: FAIL TO REJECT the null hypothesis")
            print("No statistically significant risk differences between genders.")
            print(f"\nBUSINESS IMPLICATION:")
            print("• Gender may not be a significant risk factor")
            print("• Consider removing gender as a pricing factor if legally permissible")
        
        return gender_metrics
    
    def visualize_results(self):
        """Create visualizations for hypothesis testing results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Claim Frequency by Province
        if 'Province' in self.df.columns:
            province_freq = self.df.groupby('Province')['has_claim'].mean().sort_values()
            axes[0, 0].barh(range(len(province_freq)), province_freq.values)
            axes[0, 0].set_yticks(range(len(province_freq)))
            axes[0, 0].set_yticklabels(province_freq.index)
            axes[0, 0].set_xlabel('Claim Frequency')
            axes[0, 0].set_title('Claim Frequency by Province')
            axes[0, 0].axvline(x=self.df['has_claim'].mean(), color='r', linestyle='--', alpha=0.5)
        else:
            axes[0, 0].text(0.5, 0.5, 'Province data missing', ha='center', va='center')
            axes[0, 0].set_title('Claim Frequency by Province')
        
        # 2. Claim Severity by Gender
        if 'Gender' in self.df.columns:
            gender_sev = self.df[self.df['Gender'].isin(['Male','Female'])].groupby('Gender')['claim_severity'].mean()
            if not gender_sev.empty:
                axes[0, 1].bar(gender_sev.index, gender_sev.values)
                axes[0, 1].set_ylabel('Average Claim Amount (R)')
                axes[0, 1].set_title('Claim Severity by Gender')
            else:
                axes[0, 1].text(0.5, 0.5, 'Gender data missing', ha='center', va='center')
        else:
            axes[0, 1].text(0.5, 0.5, 'Gender data missing', ha='center', va='center')
        axes[0, 1].set_title('Claim Severity by Gender')
        
        # 3. Margin by Make (proxy for zip)
        if 'Make' in self.df.columns:
            top_makes = self.df['Make'].value_counts().nlargest(5).index
            make_margin = self.df[self.df['Make'].isin(top_makes)].groupby('Make')['margin'].mean().sort_values()
            axes[1, 0].bar(range(len(make_margin)), make_margin.values)
            axes[1, 0].set_xticks(range(len(make_margin)))
            axes[1, 0].set_xticklabels(make_margin.index, rotation=45, ha='right')
            axes[1, 0].set_ylabel('Average Margin (R)')
            axes[1, 0].set_title('Profitability by Vehicle Make (Proxy for Zip)')
        else:
            axes[1, 0].text(0.5, 0.5, 'Make data missing', ha='center', va='center')
            axes[1, 0].set_title('Profitability by Vehicle Make')
        
        # 4. Loss Ratio Distribution
        loss_ratio_clean = self.df['loss_ratio'].dropna()
        if not loss_ratio_clean.empty:
            axes[1, 1].hist(loss_ratio_clean, bins=50, edgecolor='black', alpha=0.7)
            axes[1, 1].axvline(x=1, color='r', linestyle='--', label='Break-even (LR=1)')
            axes[1, 1].axvline(x=loss_ratio_clean.mean(), color='g', linestyle='--',
                               label=f'Mean: {loss_ratio_clean.mean():.2f}')
            axes[1, 1].set_xlabel('Loss Ratio (Claims/Premium)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Distribution of Loss Ratio')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'Loss ratio data missing', ha='center', va='center')
            axes[1, 1].set_title('Loss Ratio Distribution')
        
        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/hypothesis_testing_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive hypothesis testing report"""
        self.prepare_metrics()
        
        print("INSURANCE RISK HYPOTHESIS TESTING REPORT")
        print("="*60)
        
        # Run all hypothesis tests
        self.test_province_risk_differences()
        self.test_zipcode_risk_differences()
        self.test_zipcode_margin_differences()
        self.test_gender_risk_differences()
        
        # Executive summary
        summary_data = []
        summary_data.append({
            'Metric': 'Overall Claim Frequency',
            'Value': f"{self.df['has_claim'].mean():.4f}",
            'Interpretation': 'Proportion of policies with claims'
        })
        if self.df['claim_severity'].notna().any():
            summary_data.append({
                'Metric': 'Average Claim Severity',
                'Value': f"R{self.df['claim_severity'].mean():.2f}",
                'Interpretation': 'Average claim amount when claim occurs'
            })
        summary_data.append({
            'Metric': 'Average Margin per Policy',
            'Value': f"R{self.df['margin'].mean():.2f}",
            'Interpretation': 'Average profit per policy'
        })
        if self.df['loss_ratio'].notna().any():
            summary_data.append({
                'Metric': 'Overall Loss Ratio',
                'Value': f"{self.df['loss_ratio'].mean():.4f}",
                'Interpretation': 'Claims paid per premium received'
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        print("\n" + "="*60)
        print("EXECUTIVE SUMMARY")
        print("="*60)
        print(summary_df.to_string(index=False))
        
        # Save results
        os.makedirs('results', exist_ok=True)
        summary_df.to_csv('results/hypothesis_testing_summary.csv', index=False)
        
        # Create visualizations
        self.visualize_results()
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        print("1. Implement risk-based pricing that accounts for regional differences")
        print("2. Consider geographic segmentation in underwriting")
        print("3. Monitor profitability by location segments")
        print("4. Regularly re-evaluate risk factors as portfolio evolves")
        print("5. Ensure compliance with regulatory constraints on rating factors")


# Main execution
if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)
    
    tester = InsuranceHypothesisTesting(
        r'C:\Users\admin\insurance-risk-analysis-week3\data\raw\insurance_data.csv'
    )
    tester.generate_report()