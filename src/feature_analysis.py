"""
Feature Analysis and Visualization Module
Deep Learning-Based Lung Cancer Risk Prediction

This module provides feature importance analysis, correlation analysis,
and clinical feature visualization tools.

Author: Research Team
Date: October 2025
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
import warnings
warnings.filterwarnings('ignore')


class FeatureAnalyzer:
    """
    Comprehensive feature analysis for clinical features.
    """
    
    def __init__(self, X, y, feature_names):
        """
        Initialize the feature analyzer.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
        """
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.importance_scores = {}
    
    def calculate_feature_importance_rf(self):
        """
        Calculate feature importance using Random Forest.
        
        Returns:
            pd.DataFrame: Feature importance scores
        """
        print("Calculating feature importance using Random Forest...")
        
        rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
        rf.fit(self.X, self.y)
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        self.importance_scores['random_forest'] = importance_df
        
        return importance_df
    
    def calculate_mutual_information(self):
        """
        Calculate mutual information between features and target.
        
        Returns:
            pd.DataFrame: Mutual information scores
        """
        print("Calculating mutual information scores...")
        
        mi_scores = mutual_info_classif(self.X, self.y, random_state=42)
        
        mi_df = pd.DataFrame({
            'Feature': self.feature_names,
            'MI_Score': mi_scores
        }).sort_values('MI_Score', ascending=False)
        
        self.importance_scores['mutual_information'] = mi_df
        
        return mi_df
    
    def calculate_chi_square(self):
        """
        Calculate chi-square scores for feature selection.
        
        Returns:
            pd.DataFrame: Chi-square scores
        """
        print("Calculating chi-square scores...")
        
        # Ensure non-negative values for chi-square test
        X_positive = self.X - self.X.min() + 0.01
        
        chi_scores, p_values = chi2(X_positive, self.y)
        
        chi_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Chi2_Score': chi_scores,
            'P_Value': p_values
        }).sort_values('Chi2_Score', ascending=False)
        
        self.importance_scores['chi_square'] = chi_df
        
        return chi_df
    
    def perform_statistical_tests(self, df_processed):
        """
        Perform statistical tests to identify significant features.
        
        Args:
            df_processed: Preprocessed dataframe with all features
        
        Returns:
            pd.DataFrame: Statistical test results
        """
        print("Performing statistical tests...")
        
        results = []
        
        # Separate cancer and non-cancer groups
        cancer_group = df_processed[df_processed['LUNG_CANCER'] == 'YES']
        no_cancer_group = df_processed[df_processed['LUNG_CANCER'] == 'NO']
        
        for feature in self.feature_names:
            if feature in df_processed.columns:
                # T-test
                t_stat, p_value_t = stats.ttest_ind(
                    cancer_group[feature], 
                    no_cancer_group[feature]
                )
                
                # Mann-Whitney U test
                u_stat, p_value_u = stats.mannwhitneyu(
                    cancer_group[feature], 
                    no_cancer_group[feature]
                )
                
                # Effect size (Cohen's d)
                mean_diff = cancer_group[feature].mean() - no_cancer_group[feature].mean()
                pooled_std = np.sqrt((cancer_group[feature].std()**2 + 
                                     no_cancer_group[feature].std()**2) / 2)
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                
                results.append({
                    'Feature': feature,
                    'T_Statistic': t_stat,
                    'T_P_Value': p_value_t,
                    'U_Statistic': u_stat,
                    'U_P_Value': p_value_u,
                    'Cohens_D': cohens_d,
                    'Significant': p_value_t < 0.05
                })
        
        stats_df = pd.DataFrame(results).sort_values('T_P_Value')
        
        return stats_df
    
    def get_comprehensive_importance(self):
        """
        Get comprehensive feature importance combining multiple methods.
        
        Returns:
            pd.DataFrame: Combined importance scores
        """
        if not self.importance_scores:
            self.calculate_feature_importance_rf()
            self.calculate_mutual_information()
            self.calculate_chi_square()
        
        # Merge all importance scores
        df_combined = self.importance_scores['random_forest'].copy()
        df_combined = df_combined.merge(
            self.importance_scores['mutual_information'],
            on='Feature'
        )
        df_combined = df_combined.merge(
            self.importance_scores['chi_square'][['Feature', 'Chi2_Score']],
            on='Feature'
        )
        
        # Normalize scores to 0-1 range
        for col in ['Importance', 'MI_Score', 'Chi2_Score']:
            max_val = df_combined[col].max()
            if max_val > 0:
                df_combined[f'{col}_Normalized'] = df_combined[col] / max_val
        
        # Calculate average importance
        df_combined['Average_Importance'] = df_combined[[
            'Importance_Normalized', 
            'MI_Score_Normalized', 
            'Chi2_Score_Normalized'
        ]].mean(axis=1)
        
        df_combined = df_combined.sort_values('Average_Importance', ascending=False)
        
        return df_combined


class FeatureVisualizer:
    """
    Visualization tools for feature analysis.
    """
    
    @staticmethod
    def plot_feature_importance(importance_df, title="Feature Importance", save_path=None):
        """
        Plot feature importance bar chart.
        
        Args:
            importance_df: DataFrame with Feature and Importance columns
            title (str): Plot title
            save_path (str): Path to save the figure
        """
        plt.figure(figsize=(12, 8))
        
        colors = sns.color_palette("viridis", len(importance_df))
        
        plt.barh(importance_df['Feature'], importance_df.iloc[:, 1], color=colors)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_correlation_matrix(df, save_path=None):
        """
        Plot correlation matrix heatmap.
        
        Args:
            df: DataFrame with features
            save_path (str): Path to save the figure
        """
        plt.figure(figsize=(14, 12))
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True,
                   linewidths=1, cbar_kws={"shrink": 0.8})
        
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation matrix saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_feature_distributions(df, target_col='LUNG_CANCER', save_path=None):
        """
        Plot feature distributions by target class.
        
        Args:
            df: DataFrame with features and target
            target_col (str): Name of target column
            save_path (str): Path to save the figure
        """
        # Get feature columns (exclude target and derived columns)
        feature_cols = [col for col in df.columns 
                       if col not in [target_col, 'LUNG_CANCER_ENCODED', 'AGE_NORMALIZED']]
        
        n_features = len(feature_cols)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
        axes = axes.ravel() if n_rows > 1 else [axes]
        
        for idx, feature in enumerate(feature_cols):
            ax = axes[idx]
            
            # Plot distributions for each class
            for label in df[target_col].unique():
                data = df[df[target_col] == label][feature]
                ax.hist(data, alpha=0.6, label=label, bins=15)
            
            ax.set_title(feature, fontsize=10, fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Feature Distributions by Lung Cancer Status', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature distributions saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_age_distribution(df, save_path=None):
        """
        Plot age distribution by cancer status.
        
        Args:
            df: DataFrame with AGE and LUNG_CANCER columns
            save_path (str): Path to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        for label in df['LUNG_CANCER'].unique():
            data = df[df['LUNG_CANCER'] == label]['AGE']
            axes[0].hist(data, alpha=0.6, label=label, bins=20)
        
        axes[0].set_xlabel('Age', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Age Distribution by Cancer Status', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        df.boxplot(column='AGE', by='LUNG_CANCER', ax=axes[1])
        axes[1].set_xlabel('Lung Cancer Status', fontsize=12)
        axes[1].set_ylabel('Age', fontsize=12)
        axes[1].set_title('Age Distribution by Cancer Status', fontsize=12, fontweight='bold')
        plt.suptitle('')  # Remove default title
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Age distribution plot saved to {save_path}")
        
        plt.close()


def main():
    """
    Main function for testing feature analysis module.
    """
    print("="*70)
    print("FEATURE ANALYSIS MODULE")
    print("="*70)
    print("\nThis module provides comprehensive feature analysis and visualization")
    print("for clinical features in the lung cancer prediction study.")


if __name__ == "__main__":
    main()

