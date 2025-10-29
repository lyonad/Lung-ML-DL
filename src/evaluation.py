"""
Model Evaluation and Comparison Module
Deep Learning-Based Lung Cancer Risk Prediction

This module provides comprehensive evaluation metrics, confusion matrix analysis,
and comparative performance visualization for multiple ANN models.

Author: Research Team
Date: October 2025
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score
)
from sklearn.model_selection import StratifiedKFold
import json
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison class.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = {}
        
    def evaluate_model(self, model_name, y_true, y_pred_proba, threshold=0.5):
        """
        Evaluate a single model with comprehensive metrics.
        
        Args:
            model_name (str): Name of the model
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold (float): Classification threshold
        
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall/TPR
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = f1_score(y_true, y_pred)
        
        # Additional metrics
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # ROC and AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall_curve, precision_curve)
        
        # Store results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'balanced_accuracy': balanced_acc,
            'mcc': mcc,
            'kappa': kappa,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm,
            'true_positive': tp,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve
        }
        
        self.results[model_name] = results
        
        return results
    
    def print_evaluation_report(self, model_name):
        """
        Print a formatted evaluation report for a model.
        
        Args:
            model_name (str): Name of the model
        """
        if model_name not in self.results:
            print(f"No results found for {model_name}")
            return
        
        results = self.results[model_name]
        
        print(f"\n{'='*70}")
        print(f"EVALUATION REPORT: {model_name}")
        print(f"{'='*70}\n")
        
        print("Performance Metrics:")
        print(f"  Accuracy:           {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"  Balanced Accuracy:  {results['balanced_accuracy']:.4f}")
        print(f"  Sensitivity (TPR):  {results['sensitivity']:.4f}")
        print(f"  Specificity (TNR):  {results['specificity']:.4f}")
        print(f"  Precision:          {results['precision']:.4f}")
        print(f"  F1-Score:           {results['f1_score']:.4f}")
        print(f"  ROC-AUC:            {results['roc_auc']:.4f}")
        print(f"  PR-AUC:             {results['pr_auc']:.4f}")
        print(f"  MCC:                {results['mcc']:.4f}")
        print(f"  Cohen's Kappa:      {results['kappa']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Negative:  {results['true_negative']}")
        print(f"  False Positive: {results['false_positive']}")
        print(f"  False Negative: {results['false_negative']}")
        print(f"  True Positive:  {results['true_positive']}")
        
    def compare_models(self):
        """
        Compare all evaluated models and create a comparison dataframe.
        
        Returns:
            pd.DataFrame: Comparison table
        """
        if not self.results:
            print("No models have been evaluated yet.")
            return None
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Sensitivity': results['sensitivity'],
                'Specificity': results['specificity'],
                'Precision': results['precision'],
                'F1-Score': results['f1_score'],
                'ROC-AUC': results['roc_auc'],
                'PR-AUC': results['pr_auc'],
                'MCC': results['mcc'],
                'Balanced Accuracy': results['balanced_accuracy']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('ROC-AUC', ascending=False)
        
        print(f"\n{'='*70}")
        print("MODEL COMPARISON")
        print(f"{'='*70}\n")
        print(df_comparison.to_string(index=False))
        
        # Identify best model for each metric
        print(f"\n{'='*70}")
        print("BEST MODELS PER METRIC")
        print(f"{'='*70}\n")
        
        metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 
                  'F1-Score', 'ROC-AUC', 'MCC']
        
        for metric in metrics:
            best_idx = df_comparison[metric].idxmax()
            best_model = df_comparison.loc[best_idx, 'Model']
            best_value = df_comparison.loc[best_idx, metric]
            print(f"  {metric:20s}: {best_model:20s} ({best_value:.4f})")
        
        return df_comparison
    
    def save_results(self, filepath='../results/model_results.json'):
        """
        Save evaluation results to JSON file.
        
        Args:
            filepath (str): Path to save the results
        """
        # Prepare results for JSON serialization
        json_results = {}
        
        for model_name, results in self.results.items():
            json_results[model_name] = {
                'accuracy': float(results['accuracy']),
                'sensitivity': float(results['sensitivity']),
                'specificity': float(results['specificity']),
                'precision': float(results['precision']),
                'f1_score': float(results['f1_score']),
                'balanced_accuracy': float(results['balanced_accuracy']),
                'mcc': float(results['mcc']),
                'kappa': float(results['kappa']),
                'roc_auc': float(results['roc_auc']),
                'pr_auc': float(results['pr_auc']),
                'confusion_matrix': results['confusion_matrix'].tolist(),
                'tp': int(results['true_positive']),
                'tn': int(results['true_negative']),
                'fp': int(results['false_positive']),
                'fn': int(results['false_negative'])
            }
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=4)
        
        print(f"\nResults saved to {filepath}")


class VisualizationTools:
    """
    Visualization tools for model evaluation and comparison.
    """
    
    @staticmethod
    def plot_confusion_matrix(cm, model_name, save_path=None):
        """
        Plot confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix
            model_name (str): Name of the model
            save_path (str): Path to save the figure
        """
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Cancer', 'Cancer'],
                   yticklabels=['No Cancer', 'Cancer'],
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_roc_curves(evaluator, save_path=None):
        """
        Plot ROC curves for all models.
        
        Args:
            evaluator (ModelEvaluator): Evaluator object with results
            save_path (str): Path to save the figure
        """
        plt.figure(figsize=(10, 8))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for idx, (model_name, results) in enumerate(evaluator.results.items()):
            color = colors[idx % len(colors)]
            plt.plot(results['fpr'], results['tpr'], 
                    color=color, lw=2, 
                    label=f"{model_name} (AUC = {results['roc_auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_model_comparison_bar(df_comparison, save_path=None):
        """
        Plot bar chart comparing models across metrics.
        
        Args:
            df_comparison (pd.DataFrame): Comparison dataframe
            save_path (str): Path to save the figure
        """
        metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'ROC-AUC']
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.ravel()
        
        colors = sns.color_palette("husl", len(df_comparison))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            bars = ax.bar(df_comparison['Model'], df_comparison[metric], color=colors)
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=10)
            ax.set_ylim([0, 1.1])
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Model Performance Comparison Across Metrics', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison chart saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_training_history(history, model_name, save_path=None):
        """
        Plot training history (loss and metrics).
        
        Args:
            history: Keras History object
            model_name (str): Name of the model
            save_path (str): Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot Loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Model Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot AUC
        if 'auc' in history.history:
            axes[1, 0].plot(history.history['auc'], label='Training AUC', linewidth=2)
            axes[1, 0].plot(history.history['val_auc'], label='Validation AUC', linewidth=2)
            axes[1, 0].set_title('Model AUC', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot Precision and Recall
        if 'precision' in history.history:
            axes[1, 1].plot(history.history['precision'], label='Training Precision', linewidth=2)
            axes[1, 1].plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
            if 'recall' in history.history:
                axes[1, 1].plot(history.history['recall'], label='Training Recall', linewidth=2, linestyle='--')
                axes[1, 1].plot(history.history['val_recall'], label='Validation Recall', linewidth=2, linestyle='--')
            axes[1, 1].set_title('Precision & Recall', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Training History - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        
        plt.close()


def main():
    """
    Main function for testing evaluation module.
    """
    print("="*70)
    print("MODEL EVALUATION MODULE")
    print("="*70)
    print("\nThis module provides comprehensive evaluation and comparison tools")
    print("for multiple ANN models in the lung cancer prediction study.")


if __name__ == "__main__":
    main()

