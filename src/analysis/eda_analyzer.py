"""
Comprehensive EDA Analysis Module
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class EDAAnalyzer:
    """Perform comprehensive exploratory data analysis"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize EDA analyzer with data
        
        Args:
            df: DataFrame for analysis
        """
        self.df = df.copy()
        self.results = {}
        
        # Identify column types
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        logger.info(f"Identified {len(self.numeric_cols)} numeric columns, "
                   f"{len(self.categorical_cols)} categorical columns, "
                   f"{len(self.date_cols)} date columns")
    
    def compute_descriptive_statistics(self) -> pd.DataFrame:
        """
        Compute comprehensive descriptive statistics
        
        Returns:
            DataFrame: Descriptive statistics for all numeric columns
        """
        logger.info("Computing descriptive statistics")
        
        stats_list = []
        
        for col in self.numeric_cols:
            if col in self.df.columns:
                col_stats = {
                    'feature': col,
                    'count': self.df[col].count(),
                    'mean': self.df[col].mean(),
                    'std': self.df[col].std(),
                    'min': self.df[col].min(),
                    '25%': self.df[col].quantile(0.25),
                    'median': self.df[col].quantile(0.50),
                    '75%': self.df[col].quantile(0.75),
                    'max': self.df[col].max(),
                    'range': self.df[col].max() - self.df[col].min(),
                    'iqr': self.df[col].quantile(0.75) - self.df[col].quantile(0.25),
                    'skewness': self.df[col].skew(),
                    'kurtosis': self.df[col].kurtosis(),
                    'missing': self.df[col].isnull().sum(),
                    'missing_pct': (self.df[col].isnull().sum() / len(self.df)) * 100,
                    'zeros': (self.df[col] == 0).sum(),
                    'zeros_pct': ((self.df[col] == 0).sum() / len(self.df)) * 100,
                    'unique': self.df[col].nunique()
                }
                stats_list.append(col_stats)
        
        stats_df = pd.DataFrame(stats_list)
        self.results['descriptive_stats'] = stats_df
        
        logger.info(f"Computed descriptive statistics for {len(stats_list)} numeric features")
        return stats_df
    
    def analyze_missing_values(self) -> pd.DataFrame:
        """
        Analyze missing values pattern
        
        Returns:
            DataFrame: Missing values analysis
        """
        logger.info("Analyzing missing values")
        
        missing_data = []
        
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            
            missing_data.append({
                'feature': col,
                'dtype': str(self.df[col].dtype),
                'total': len(self.df),
                'missing': missing_count,
                'missing_pct': missing_pct,
                'unique': self.df[col].nunique(),
                'most_frequent': self.df[col].mode()[0] if not self.df[col].mode().empty else None,
                'freq_count': self.df[col].value_counts().iloc[0] if not self.df[col].empty else 0
            })
        
        missing_df = pd.DataFrame(missing_data)
        missing_df = missing_df.sort_values('missing_pct', ascending=False)
        
        self.results['missing_values'] = missing_df
        
        # Identify patterns in missing values
        missing_matrix = self.df.isnull().astype(int)
        correlation_matrix = missing_matrix.corr()
        
        self.results['missing_correlation'] = correlation_matrix
        
        logger.info(f"Missing values analysis completed. "
                   f"Features with >50% missing: {(missing_df['missing_pct'] > 50).sum()}")
        
        return missing_df
    
    def analyze_distributions(self) -> Dict:
        """
        Analyze distributions of all features
        
        Returns:
            Dict: Distribution analysis results
        """
        logger.info("Analyzing feature distributions")
        
        distribution_results = {
            'numeric_distributions': {},
            'categorical_distributions': {},
            'normality_tests': {}
        }
        
        # Analyze numeric distributions
        for col in self.numeric_cols[:20]:  # Limit to first 20 for performance
            if col in self.df.columns:
                # Shapiro-Wilk test for normality
                try:
                    data = self.df[col].dropna()
                    if len(data) > 3 and len(data) < 5000:  # Shapiro-Wilk has sample size limits
                        stat, p_value = stats.shapiro(data)
                        distribution_results['normality_tests'][col] = {
                            'statistic': stat,
                            'p_value': p_value,
                            'is_normal': p_value > 0.05
                        }
                except:
                    pass
                
                # Distribution statistics
                distribution_results['numeric_distributions'][col] = {
                    'mean': float(self.df[col].mean()),
                    'median': float(self.df[col].median()),
                    'mode': float(self.df[col].mode()[0]) if not self.df[col].mode().empty else None,
                    'variance': float(self.df[col].var()),
                    'coefficient_of_variation': float(self.df[col].std() / self.df[col].mean()) 
                                                if self.df[col].mean() != 0 else None
                }
        
        # Analyze categorical distributions
        for col in self.categorical_cols[:20]:  # Limit to first 20
            if col in self.df.columns:
                value_counts = self.df[col].value_counts()
                
                distribution_results['categorical_distributions'][col] = {
                    'unique_values': int(self.df[col].nunique()),
                    'top_5_values': value_counts.head().to_dict(),
                    'entropy': self._calculate_entropy(value_counts),
                    'concentration_ratio': (value_counts.head(3).sum() / len(self.df)) 
                                          if len(self.df) > 0 else 0
                }
        
        self.results['distributions'] = distribution_results
        
        return distribution_results
    
    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calculate entropy of a distribution"""
        probabilities = value_counts / value_counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return float(entropy)
    
    def analyze_correlations(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Analyze correlations between numeric features
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            DataFrame: Correlation matrix
        """
        logger.info(f"Analyzing correlations using {method} method")
        
        # Select only numeric columns
        numeric_df = self.df[self.numeric_cols]
        
        # Compute correlation matrix
        correlation_matrix = numeric_df.corr(method=method)
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Threshold for high correlation
                    high_corr_pairs.append({
                        'feature_1': correlation_matrix.columns[i],
                        'feature_2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        correlation_results = {
            'correlation_matrix': correlation_matrix,
            'high_correlation_pairs': pd.DataFrame(high_corr_pairs) if high_corr_pairs else None,
            'correlation_with_target': self._analyze_target_correlations() 
                                       if 'TotalClaims' in self.df.columns else None
        }
        
        self.results['correlations'] = correlation_results
        
        return correlation_matrix
    
    def _analyze_target_correlations(self) -> pd.DataFrame:
        """Analyze correlations with target variables"""
        target_cols = ['TotalClaims', 'LossRatio']
        correlation_with_target = []
        
        for target in target_cols:
            if target in self.df.columns:
                for col in self.numeric_cols:
                    if col != target and col in self.df.columns:
                        corr = self.df[[col, target]].corr().iloc[0, 1]
                        correlation_with_target.append({
                            'feature': col,
                            'target': target,
                            'correlation': corr,
                            'abs_correlation': abs(corr)
                        })
        
        return pd.DataFrame(correlation_with_target).sort_values('abs_correlation', ascending=False)
    
    def detect_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> Dict:
        """
        Detect outliers in numeric features
        
        Args:
            method: Outlier detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Dict: Outlier detection results
        """
        logger.info(f"Detecting outliers using {method} method")
        
        outlier_results = {
            'outliers_by_feature': {},
            'summary': {}
        }
        
        total_outliers = 0
        outlier_records = set()
        
        for col in self.numeric_cols:
            if col in self.df.columns:
                data = self.df[col].dropna()
                
                if method == 'iqr':
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                    
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(data))
                    outlier_indices = np.where(z_scores > threshold)[0]
                    outliers = self.df.iloc[outlier_indices] if len(outlier_indices) > 0 else pd.DataFrame()
                
                outlier_count = len(outliers)
                total_outliers += outlier_count
                
                if outlier_count > 0:
                    outlier_records.update(outliers.index.tolist())
                
                outlier_results['outliers_by_feature'][col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(self.df)) * 100,
                    'lower_bound': float(lower_bound) if method == 'iqr' else None,
                    'upper_bound': float(upper_bound) if method == 'iqr' else None,
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'outlier_indices': outliers.index.tolist() if outlier_count > 0 else []
                }
        
        outlier_results['summary'] = {
            'total_outlier_records': len(outlier_records),
            'percentage_outlier_records': (len(outlier_records) / len(self.df)) * 100,
            'features_with_outliers': sum(1 for v in outlier_results['outliers_by_feature'].values() 
                                         if v['count'] > 0)
        }
        
        self.results['outliers'] = outlier_results
        
        logger.info(f"Detected outliers in {outlier_results['summary']['features_with_outliers']} "
                   f"features affecting {len(outlier_records)} records")
        
        return outlier_results
    
    def analyze_temporal_trends(self, date_col: str = 'TransactionMonth') -> Dict:
        """
        Analyze temporal trends in the data
        
        Args:
            date_col: Name of the date column
            
        Returns:
            Dict: Temporal analysis results
        """
        logger.info(f"Analyzing temporal trends using {date_col}")
        
        if date_col not in self.df.columns:
            logger.warning(f"Date column {date_col} not found in data")
            return {}
        
        temporal_results = {}
        
        # Resample data by month
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        monthly_data = self.df.set_index(date_col).resample('M').agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum',
            'PolicyID': 'nunique' if 'PolicyID' in self.df.columns else 'count'
        })
        
        monthly_data['LossRatio'] = monthly_data['TotalClaims'] / monthly_data['TotalPremium']
        monthly_data['AvgClaim'] = monthly_data['TotalClaims'] / monthly_data['PolicyID']
        monthly_data['AvgPremium'] = monthly_data['TotalPremium'] / monthly_data['PolicyID']
        
        # Calculate trends
        for col in ['TotalPremium', 'TotalClaims', 'LossRatio']:
            if col in monthly_data.columns:
                # Linear trend
                x = np.arange(len(monthly_data[col]))
                y = monthly_data[col].values
                
                if len(y) > 1 and not np.isnan(y).all():
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    temporal_results[f'{col}_trend'] = {
                        'slope': float(slope),
                        'intercept': float(intercept),
                        'r_squared': float(r_value ** 2),
                        'p_value': float(p_value),
                        'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                        'percent_change': ((monthly_data[col].iloc[-1] - monthly_data[col].iloc[0]) / 
                                          monthly_data[col].iloc[0] * 100) if monthly_data[col].iloc[0] != 0 else 0
                    }
        
        temporal_results['monthly_data'] = monthly_data
        
        # Seasonal decomposition
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            for col in ['TotalClaims', 'TotalPremium']:
                if col in monthly_data.columns:
                    decomposition = seasonal_decompose(
                        monthly_data[col].dropna(),
                        model='additive',
                        period=12  # Assuming yearly seasonality
                    )
                    
                    temporal_results[f'{col}_seasonality'] = {
                        'seasonal_amplitude': float(decomposition.seasonal.std()),
                        'trend_strength': float(1 - (decomposition.resid.var() / 
                                                   (decomposition.trend + decomposition.resid).var()))
                    }
        except ImportError:
            logger.warning("statsmodels not installed for seasonal decomposition")
        
        self.results['temporal_analysis'] = temporal_results
        
        return temporal_results
    
    def analyze_by_dimensions(self, dimensions: List[str]) -> Dict:
        """
        Analyze key metrics by different dimensions
        
        Args:
            dimensions: List of dimension columns to analyze by
            
        Returns:
            Dict: Analysis by dimensions
        """
        logger.info(f"Analyzing by dimensions: {dimensions}")
        
        dimension_results = {}
        
        for dim in dimensions:
            if dim in self.df.columns:
                # Group by dimension
                group_stats = self.df.groupby(dim).agg({
                    'TotalPremium': ['sum', 'mean', 'std', 'count'],
                    'TotalClaims': ['sum', 'mean', 'std'],
                    'LossRatio': ['mean', 'std', 'count']
                }).round(3)
                
                # Flatten column names
                group_stats.columns = ['_'.join(col).strip() for col in group_stats.columns.values]
                group_stats = group_stats.reset_index()
                
                # Calculate additional metrics
                group_stats['Premium_Share'] = (group_stats['TotalPremium_sum'] / 
                                               group_stats['TotalPremium_sum'].sum()) * 100
                group_stats['Claims_Share'] = (group_stats['TotalClaims_sum'] / 
                                              group_stats['TotalClaims_sum'].sum()) * 100
                
                dimension_results[dim] = group_stats.sort_values('LossRatio_mean', ascending=False)
        
        self.results['dimension_analysis'] = dimension_results
        
        return dimension_results
    
    def generate_summary_report(self) -> Dict:
        """
        Generate comprehensive summary report
        
        Returns:
            Dict: Complete EDA summary
        """
        logger.info("Generating comprehensive EDA summary report")
        
        summary_report = {
            'dataset_overview': {
                'shape': self.df.shape,
                'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
                'date_range': {
                    'start': self.df['TransactionMonth'].min() if 'TransactionMonth' in self.df.columns else None,
                    'end': self.df['TransactionMonth'].max() if 'TransactionMonth' in self.df.columns else None,
                    'duration_days': (self.df['TransactionMonth'].max() - 
                                     self.df['TransactionMonth'].min()).days 
                                     if 'TransactionMonth' in self.df.columns else None
                }
            },
            'data_quality': {
                'total_missing_values': self.df.isnull().sum().sum(),
                'missing_value_percentage': (self.df.isnull().sum().sum() / 
                                           (self.df.shape[0] * self.df.shape[1])) * 100,
                'duplicate_rows': self.df.duplicated().sum(),
                'duplicate_percentage': (self.df.duplicated().sum() / self.df.shape[0]) * 100
            },
            'key_metrics': {
                'overall_loss_ratio': float(self.df['LossRatio'].mean()) if 'LossRatio' in self.df.columns else None,
                'total_premium': float(self.df['TotalPremium'].sum()) if 'TotalPremium' in self.df.columns else None,
                'total_claims': float(self.df['TotalClaims'].sum()) if 'TotalClaims' in self.df.columns else None,
                'average_claim': float(self.df['TotalClaims'].mean()) if 'TotalClaims' in self.df.columns else None,
                'average_premium': float(self.df['TotalPremium'].mean()) if 'TotalPremium' in self.df.columns else None
            },
            'risk_insights': self._generate_risk_insights()
        }
        
        # Add previous analysis results
        summary_report.update(self.results)
        
        return summary_report
    
    def _generate_risk_insights(self) -> Dict:
        """Generate risk-related insights from the data"""
        insights = {}
        
        # Top risky provinces
        if 'Province' in self.df.columns and 'LossRatio' in self.df.columns:
            province_risk = self.df.groupby('Province')['LossRatio'].agg(['mean', 'count']).reset_index()
            province_risk = province_risk[province_risk['count'] > 10]  # Filter for sufficient data
            
            insights['top_risky_provinces'] = province_risk.sort_values('mean', ascending=False).head(5).to_dict('records')
            insights['safest_provinces'] = province_risk.sort_values('mean', ascending=True).head(5).to_dict('records')
        
        # Top risky vehicle types
        if 'VehicleType' in self.df.columns and 'LossRatio' in self.df.columns:
            vehicle_risk = self.df.groupby('VehicleType')['LossRatio'].agg(['mean', 'count']).reset_index()
            vehicle_risk = vehicle_risk[vehicle_risk['count'] > 10]
            
            insights['top_risky_vehicles'] = vehicle_risk.sort_values('mean', ascending=False).head(5).to_dict('records')
        
        # High claim makes/models
        if all(col in self.df.columns for col in ['Make', 'Model', 'TotalClaims']):
            make_model_claims = self.df.groupby(['Make', 'Model'])['TotalClaims'].agg(['sum', 'mean', 'count']).reset_index()
            make_model_claims = make_model_claims[make_model_claims['count'] > 5]
            
            insights['highest_claim_makes_models'] = make_model_claims.sort_values('sum', ascending=False).head(10).to_dict('records')
        
        return insights