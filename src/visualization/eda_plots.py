"""
Visualization Module for EDA
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EDAVisualizer:
    """Create comprehensive visualizations for EDA"""
    
    def __init__(self, df: pd.DataFrame, save_dir: str = "reports/figures"):
        """
        Initialize visualizer
        
        Args:
            df: DataFrame for visualization
            save_dir: Directory to save figures
        """
        self.df = df
        self.save_dir = save_dir
        
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_data_quality_summary(self, missing_df: pd.DataFrame):
        """
        Plot data quality summary
        
        Args:
            missing_df: DataFrame with missing value analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Missing values percentage
        top_missing = missing_df.head(20).sort_values('missing_pct', ascending=True)
        axes[0, 0].barh(range(len(top_missing)), top_missing['missing_pct'])
        axes[0, 0].set_yticks(range(len(top_missing)))
        axes[0, 0].set_yticklabels(top_missing['feature'])
        axes[0, 0].set_xlabel('Missing Percentage (%)')
        axes[0, 0].set_title('Top 20 Features with Missing Values')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Data types distribution
        dtype_counts = self.df.dtypes.value_counts()
        axes[0, 1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Data Types Distribution')
        
        # Plot 3: Unique values distribution
        unique_counts = self.df.nunique()
        top_unique = unique_counts.sort_values(ascending=False).head(20)
        axes[1, 0].bar(range(len(top_unique)), top_unique.values)
        axes[1, 0].set_xticks(range(len(top_unique)))
        axes[1, 0].set_xticklabels(top_unique.index, rotation=90)
        axes[1, 0].set_ylabel('Number of Unique Values')
        axes[1, 0].set_title('Top 20 Features by Unique Values')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Zero values percentage
        zero_counts = []
        for col in self.df.select_dtypes(include=[np.number]).columns:
            zero_pct = (self.df[col] == 0).sum() / len(self.df) * 100
            if zero_pct > 50:  # Only show features with >50% zeros
                zero_counts.append({'feature': col, 'zero_pct': zero_pct})
        
        if zero_counts:
            zero_df = pd.DataFrame(zero_counts).sort_values('zero_pct', ascending=True)
            axes[1, 1].barh(range(len(zero_df)), zero_df['zero_pct'])
            axes[1, 1].set_yticks(range(len(zero_df)))
            axes[1, 1].set_yticklabels(zero_df['feature'])
            axes[1, 1].set_xlabel('Percentage of Zeros (%)')
            axes[1, 1].set_title('Features with High Percentage of Zero Values')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No features with >50% zero values', 
                           ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('Zero Values Analysis')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/data_quality_summary.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_numeric_distributions(self, numeric_cols: List[str], n_cols: int = 3):
        """
        Plot distributions of numeric columns
        
        Args:
            numeric_cols: List of numeric column names
            n_cols: Number of columns in subplot grid
        """
        n_rows = int(np.ceil(len(numeric_cols) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, col in enumerate(numeric_cols[:len(axes)]):
            if col in self.df.columns:
                ax = axes[idx]
                
                # Histogram with KDE
                sns.histplot(self.df[col].dropna(), kde=True, ax=ax, bins=50)
                
                # Add statistics
                stats_text = (f"Mean: {self.df[col].mean():.2f}\n"
                             f"Std: {self.df[col].std():.2f}\n"
                             f"Skew: {self.df[col].skew():.2f}")
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_title(f'Distribution of {col}', fontsize=12)
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/numeric_distributions.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_categorical_distributions(self, categorical_cols: List[str], n_cols: int = 2):
        """
        Plot distributions of categorical columns
        
        Args:
            categorical_cols: List of categorical column names
            n_cols: Number of columns in subplot grid
        """
        n_rows = int(np.ceil(len(categorical_cols) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, col in enumerate(categorical_cols[:len(axes)]):
            if col in self.df.columns:
                ax = axes[idx]
                
                # Get top 10 categories
                value_counts = self.df[col].value_counts().head(10)
                
                # Create bar plot
                bars = ax.bar(range(len(value_counts)), value_counts.values, color='steelblue')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontsize=8)
                
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right', fontsize=9)
                ax.set_title(f'Top 10 {col} Distribution', fontsize=12)
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                ax.grid(True, alpha=0.3, axis='y')
        
        # Hide empty subplots
        for idx in range(len(categorical_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/categorical_distributions.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(self, correlation_matrix: pd.DataFrame):
        """
        Plot correlation matrix heatmap
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
        """
        plt.figure(figsize=(14, 12))
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
        
        plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_loss_ratio_analysis(self):
        """Plot comprehensive loss ratio analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss Ratio by Province', 'Loss Ratio by Vehicle Type',
                          'Loss Ratio by Gender', 'Loss Ratio by Cover Type'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Loss Ratio by Province
        if 'Province' in self.df.columns and 'LossRatio' in self.df.columns:
            province_loss = self.df.groupby('Province')['LossRatio'].mean().sort_values(ascending=False).head(10)
            fig.add_trace(
                go.Bar(x=province_loss.index, y=province_loss.values, name='Province'),
                row=1, col=1
            )
        
        # 2. Loss Ratio by Vehicle Type
        if 'VehicleType' in self.df.columns and 'LossRatio' in self.df.columns:
            vehicle_loss = self.df.groupby('VehicleType')['LossRatio'].mean().sort_values(ascending=False)
            fig.add_trace(
                go.Bar(x=vehicle_loss.index, y=vehicle_loss.values, name='Vehicle Type'),
                row=1, col=2
            )
        
        # 3. Loss Ratio by Gender
        if 'Gender' in self.df.columns and 'LossRatio' in self.df.columns:
            gender_loss = self.df.groupby('Gender')['LossRatio'].mean()
            fig.add_trace(
                go.Bar(x=gender_loss.index, y=gender_loss.values, name='Gender'),
                row=2, col=1
            )
        
        # 4. Loss Ratio by Cover Type
        if 'CoverType' in self.df.columns and 'LossRatio' in self.df.columns:
            cover_loss = self.df.groupby('CoverType')['LossRatio'].mean().sort_values(ascending=False).head(10)
            fig.add_trace(
                go.Bar(x=cover_loss.index, y=cover_loss.values, name='Cover Type'),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Loss Ratio Analysis by Different Dimensions",
            title_font_size=20
        )
        
        fig.update_xaxes(tickangle=45)
        
        # Save plot
        fig.write_html(f"{self.save_dir}/loss_ratio_analysis.html")
        fig.show()
    
    def plot_temporal_trends(self, monthly_data: pd.DataFrame):
        """
        Plot temporal trends
        
        Args:
            monthly_data: Monthly aggregated data
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Monthly Total Premium', 'Monthly Total Claims',
                          'Monthly Loss Ratio', 'Monthly Average Claim'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                  [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # 1. Monthly Total Premium
        fig.add_trace(
            go.Scatter(x=monthly_data.index, y=monthly_data['TotalPremium'],
                      mode='lines+markers', name='Total Premium',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # 2. Monthly Total Claims
        fig.add_trace(
            go.Scatter(x=monthly_data.index, y=monthly_data['TotalClaims'],
                      mode='lines+markers', name='Total Claims',
                      line=dict(color='red', width=2)),
            row=1, col=2
        )
        
        # 3. Monthly Loss Ratio
        fig.add_trace(
            go.Scatter(x=monthly_data.index, y=monthly_data['LossRatio'],
                      mode='lines+markers', name='Loss Ratio',
                      line=dict(color='green', width=2)),
            row=2, col=1
        )
        
        # 4. Monthly Average Claim
        if 'AvgClaim' in monthly_data.columns:
            fig.add_trace(
                go.Scatter(x=monthly_data.index, y=monthly_data['AvgClaim'],
                          mode='lines+markers', name='Average Claim',
                          line=dict(color='purple', width=2)),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Temporal Trends Analysis (Monthly)",
            title_font_size=20
        )
        
        fig.update_xaxes(title_text="Month")
        fig.update_yaxes(title_text="Amount", row=1, col=1)
        fig.update_yaxes(title_text="Amount", row=1, col=2)
        fig.update_yaxes(title_text="Ratio", row=2, col=1)
        fig.update_yaxes(title_text="Amount", row=2, col=2)
        
        # Save plot
        fig.write_html(f"{self.save_dir}/temporal_trends.html")
        fig.show()
    
    def plot_outlier_analysis(self, outlier_results: Dict):
        """
        Plot outlier analysis
        
        Args:
            outlier_results: Results from outlier detection
        """
        features_with_outliers = []
        outlier_counts = []
        
        for feature, stats in outlier_results.get('outliers_by_feature', {}).items():
            if stats['count'] > 0:
                features_with_outliers.append(feature)
                outlier_counts.append(stats['count'])
        
        if not features_with_outliers:
            print("No outliers detected in any features")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Features with most outliers
        top_n = min(10, len(features_with_outliers))
        top_features = features_with_outliers[:top_n]
        top_counts = outlier_counts[:top_n]
        
        axes[0, 0].barh(range(top_n), top_counts[::-1])
        axes[0, 0].set_yticks(range(top_n))
        axes[0, 0].set_yticklabels(top_features[::-1])
        axes[0, 0].set_xlabel('Number of Outliers')
        axes[0, 0].set_title('Top Features by Number of Outliers')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Box plots for top 4 features with outliers
        for idx, feature in enumerate(top_features[:4]):
            row = idx // 2
            col = idx % 2
            ax = axes[1, col] if row == 1 else axes[0, 1]
            
            if feature in self.df.columns:
                # Create box plot
                bp = ax.boxplot(self.df[feature].dropna(), vert=False, patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue')
                
                # Highlight outliers
                outlier_stats = outlier_results['outliers_by_feature'][feature]
                if 'lower_bound' in outlier_stats and 'upper_bound' in outlier_stats:
                    ax.axvline(outlier_stats['lower_bound'], color='red', linestyle='--', alpha=0.5)
                    ax.axvline(outlier_stats['upper_bound'], color='red', linestyle='--', alpha=0.5)
                
                ax.set_title(f'Box Plot: {feature}\nOutliers: {outlier_stats["count"]}')
                ax.set_xlabel(feature)
                ax.grid(True, alpha=0.3)
        
        # Plot 3: Outlier percentage distribution
        outlier_percentages = []
        for feature, stats in outlier_results.get('outliers_by_feature', {}).items():
            if stats['count'] > 0:
                outlier_percentages.append(stats['percentage'])
        
        axes[1, 1].hist(outlier_percentages, bins=20, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(np.mean(outlier_percentages), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(outlier_percentages):.2f}%')
        axes[1, 1].set_xlabel('Outlier Percentage (%)')
        axes[1, 1].set_ylabel('Number of Features')
        axes[1, 1].set_title('Distribution of Outlier Percentages')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/outlier_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_creative_visualization_1(self):
        """Creative Visualization 1: Risk Heatmap by Province and Vehicle Type"""
        # Prepare data for heatmap
        if all(col in self.df.columns for col in ['Province', 'VehicleType', 'LossRatio']):
            heatmap_data = self.df.groupby(['Province', 'VehicleType'])['LossRatio'].agg(
                ['mean', 'count']
            ).reset_index()
            
            # Filter for sufficient data
            heatmap_data = heatmap_data[heatmap_data['count'] > 10]
            
            # Create pivot table
            pivot_table = heatmap_data.pivot(index='Province', columns='VehicleType', values='mean')
            
            # Create interactive heatmap
            fig = px.imshow(pivot_table,
                          labels=dict(x="Vehicle Type", y="Province", color="Loss Ratio"),
                          x=pivot_table.columns,
                          y=pivot_table.index,
                          color_continuous_scale="RdYlGn_r",
                          title="Loss Ratio Heatmap: Province vs Vehicle Type",
                          aspect="auto")
            
            fig.update_layout(
                height=600,
                xaxis_title="Vehicle Type",
                yaxis_title="Province",
                font=dict(size=12)
            )
            
            fig.write_html(f"{self.save_dir}/creative_1_risk_heatmap.html")
            fig.show()
    
    def create_creative_visualization_2(self):
        """Creative Visualization 2: Interactive Risk Profile Dashboard"""
        if all(col in self.df.columns for col in ['Make', 'Model', 'TotalClaims', 'TotalPremium', 'LossRatio']):
            # Aggregate data by make and model
            vehicle_risk = self.df.groupby(['Make', 'Model']).agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum',
                'LossRatio': 'mean',
                'PolicyID': 'count'
            }).reset_index()
            
            vehicle_risk = vehicle_risk[vehicle_risk['PolicyID'] > 5]  # Filter for sufficient data
            
            # Calculate additional metrics
            vehicle_risk['ClaimFrequency'] = vehicle_risk['TotalClaims'] / vehicle_risk['TotalPremium']
            vehicle_risk['RiskScore'] = vehicle_risk['LossRatio'] * vehicle_risk['ClaimFrequency']
            
            # Create interactive scatter plot
            fig = px.scatter(vehicle_risk,
                           x='TotalPremium',
                           y='TotalClaims',
                           size='PolicyID',
                           color='LossRatio',
                           hover_name='Make',
                           hover_data=['Model', 'RiskScore', 'ClaimFrequency'],
                           color_continuous_scale=px.colors.sequential.Viridis,
                           title="Vehicle Risk Profile: Premium vs Claims",
                           labels={
                               'TotalPremium': 'Total Premium',
                               'TotalClaims': 'Total Claims',
                               'LossRatio': 'Loss Ratio',
                               'PolicyID': 'Number of Policies'
                           })
            
            fig.update_layout(
                height=700,
                xaxis_title="Total Premium (Log Scale)",
                yaxis_title="Total Claims (Log Scale)",
                xaxis_type="log",
                yaxis_type="log",
                font=dict(size=12)
            )
            
            # Add trend line
            x_log = np.log(vehicle_risk['TotalPremium'] + 1)
            y_log = np.log(vehicle_risk['TotalClaims'] + 1)
            coefficients = np.polyfit(x_log, y_log, 1)
            polynomial = np.poly1d(coefficients)
            
            x_range = np.linspace(x_log.min(), x_log.max(), 100)
            fig.add_trace(
                go.Scatter(x=np.exp(x_range),
                          y=np.exp(polynomial(x_range)),
                          mode='lines',
                          name='Trend Line',
                          line=dict(color='red', dash='dash'))
            )
            
            fig.write_html(f"{self.save_dir}/creative_2_risk_dashboard.html")
            fig.show()
    
    def create_creative_visualization_3(self):
        """Creative Visualization 3: Temporal Risk Evolution"""
        if 'TransactionMonth' in self.df.columns and 'LossRatio' in self.df.columns:
            # Prepare monthly data with rolling statistics
            monthly_data = self.df.set_index('TransactionMonth').resample('M').agg({
                'TotalPremium': 'sum',
                'TotalClaims': 'sum',
                'LossRatio': 'mean',
                'PolicyID': 'count'
            }).reset_index()
            
            monthly_data['RollingLossRatio'] = monthly_data['LossRatio'].rolling(window=3, center=True).mean()
            monthly_data['PremiumGrowth'] = monthly_data['TotalPremium'].pct_change() * 100
            monthly_data['ClaimsGrowth'] = monthly_data['TotalClaims'].pct_change() * 100
            
            # Create subplot figure
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Loss Ratio Trend', 'Premium vs Claims',
                              'Premium Growth Rate', 'Claims Growth Rate',
                              'Risk Evolution Matrix', 'Monthly Policy Count'),
                specs=[[{}, {}], [{}, {}], [{"colspan": 2}, None]],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # 1. Loss Ratio Trend
            fig.add_trace(
                go.Scatter(x=monthly_data['TransactionMonth'],
                          y=monthly_data['LossRatio'],
                          mode='lines',
                          name='Loss Ratio',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=monthly_data['TransactionMonth'],
                          y=monthly_data['RollingLossRatio'],
                          mode='lines',
                          name='3-Month Rolling Avg',
                          line=dict(color='red', width=2, dash='dash')),
                row=1, col=1
            )
            
            # 2. Premium vs Claims
            fig.add_trace(
                go.Scatter(x=monthly_data['TotalPremium'],
                          y=monthly_data['TotalClaims'],
                          mode='markers+lines',
                          name='Premium vs Claims',
                          marker=dict(size=10,
                                     color=monthly_data.index,
                                     colorscale='Viridis',
                                     showscale=True,
                                     colorbar=dict(title="Month Index")),
                          line=dict(color='gray', width=1)),
                row=1, col=2
            )
            
            # 3. Premium Growth Rate
            fig.add_trace(
                go.Bar(x=monthly_data['TransactionMonth'],
                      y=monthly_data['PremiumGrowth'],
                      name='Premium Growth',
                      marker_color='green'),
                row=2, col=1
            )
            
            # 4. Claims Growth Rate
            fig.add_trace(
                go.Bar(x=monthly_data['TransactionMonth'],
                      y=monthly_data['ClaimsGrowth'],
                      name='Claims Growth',
                      marker_color='red'),
                row=2, col=2
            )
            
            # 5. Risk Evolution Matrix
            # Create risk categories
            monthly_data['RiskCategory'] = pd.cut(monthly_data['LossRatio'],
                                                 bins=[0, 0.5, 0.8, 1.2, np.inf],
                                                 labels=['Low', 'Medium', 'High', 'Very High'])
            
            risk_matrix = monthly_data.pivot_table(index=monthly_data['TransactionMonth'].dt.year,
                                                  columns=monthly_data['TransactionMonth'].dt.month,
                                                  values='RiskCategory',
                                                  aggfunc='first')
            
            fig.add_trace(
                go.Heatmap(z=risk_matrix.values,
                          x=risk_matrix.columns,
                          y=risk_matrix.index,
                          colorscale='RdYlGn_r',
                          showscale=True,
                          colorbar=dict(title="Risk Level")),
                row=3, col=1
            )
            
            # 6. Monthly Policy Count
            fig.add_trace(
                go.Bar(x=monthly_data['TransactionMonth'],
                      y=monthly_data['PolicyID'],
                      name='Policy Count',
                      marker_color='purple'),
                row=2, col=2  # Overlay on claims growth
            )
            
            fig.update_layout(
                height=1200,
                showlegend=True,
                title_text="Temporal Risk Evolution Dashboard",
                title_font_size=20
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_yaxes(title_text="Loss Ratio", row=1, col=1)
            fig.update_xaxes(title_text="Total Premium", row=1, col=2)
            fig.update_yaxes(title_text="Total Claims", row=1, col=2)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Growth Rate (%)", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=2)
            fig.update_yaxes(title_text="Growth Rate (%)", row=2, col=2)
            fig.update_xaxes(title_text="Month", row=3, col=1)
            fig.update_yaxes(title_text="Year", row=3, col=1)
            
            fig.write_html(f"{self.save_dir}/creative_3_temporal_risk.html")
            fig.show()