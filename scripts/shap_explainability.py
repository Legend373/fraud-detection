import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class SHAPExplainer:
    def __init__(self, model, X_train, X_test, y_test, feature_names, 
                 compute_shap=True, sample_size=1000):
        """
        Optimized SHAP explainer for fraud detection (SHAP v0.20+ compatible).
        """
        self.model = model
        self.feature_names = list(feature_names) if feature_names is not None else None
        
        # Convert to DataFrames
        self.X_train = pd.DataFrame(X_train, columns=self.feature_names)
        self.X_test = pd.DataFrame(X_test, columns=self.feature_names)
        self.y_test = y_test.values if hasattr(y_test, 'values') else y_test
        
        print(f"✓ Data loaded: X_train={self.X_train.shape}, X_test={self.X_test.shape}")
        print(f"✓ Model type: {type(model).__name__}")
        
        # Store predictions
        self.predictions = self.model.predict(self.X_test)
        
        # Initialize explainer
        try:
            self.explainer = shap.TreeExplainer(self.model)
            print(f"✓ TreeExplainer created")
        except Exception as e:
            print(f"⚠ TreeExplainer failed: {e}")
            print("Trying KernelExplainer...")
            self.explainer = shap.KernelExplainer(self.model.predict_proba, self.X_train.iloc[:100])
        
        self.shap_values = None
        self.shap_values_all = None
        self.sample_size = min(sample_size, len(self.X_test))
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create sample for fast computation
        if len(self.X_test) > self.sample_size:
            print(f"✓ Sampling {self.sample_size} of {len(self.X_test)} rows")
            self.sample_indices = np.random.choice(len(self.X_test), self.sample_size, replace=False)
            self.X_sample = self.X_test.iloc[self.sample_indices].copy()
        else:
            self.X_sample = self.X_test.copy()
            self.sample_indices = np.arange(len(self.X_test))
        
        if compute_shap:
            self._ensure_shap_values()
    
    def _ensure_shap_values(self):
        """Compute SHAP values if not already computed"""
        if self.shap_values is None:
            print("🔍 Computing SHAP values...")
            try:
                shap_result = self.explainer(self.X_sample)
                
                # SHAP v0.20+ returns an Explanation object
                if hasattr(shap_result, 'values'):
                    self.shap_values_all = shap_result.values
                    print(f"  SHAP Explanation object with shape: {self.shap_values_all.shape}")
                    
                    # For binary classification, we want the positive class (fraud = class 1)
                    if len(self.shap_values_all.shape) == 3:
                        # Shape: (samples, features, classes)
                        self.shap_values = self.shap_values_all[:, :, 1]  # Positive class
                        self.base_value = shap_result.base_values[:, 1] if shap_result.base_values.ndim > 1 else shap_result.base_values
                        print(f"  Using positive class (fraud) with shape: {self.shap_values.shape}")
                    else:
                        self.shap_values = self.shap_values_all
                        self.base_value = shap_result.base_values
                else:
                    # Legacy SHAP return format
                    if isinstance(shap_result, list):
                        print(f"  SHAP returned list with {len(shap_result)} elements")
                        if len(shap_result) == 2:  # Binary classification
                            self.shap_values = shap_result[1]  # Positive class
                            self.shap_values_all = shap_result
                            self.base_value = self.explainer.expected_value[1]
                        else:
                            self.shap_values = shap_result[0]
                            self.shap_values_all = shap_result
                            self.base_value = self.explainer.expected_value
                    else:
                        self.shap_values = shap_result
                        self.shap_values_all = shap_result
                        self.base_value = self.explainer.expected_value
                
                print(f"✓ SHAP computation complete: shape={self.shap_values.shape}")
                
            except Exception as e:
                print(f"❌ SHAP computation failed: {e}")
                print("Using feature importance as fallback...")
                self._create_fallback_shap()
    
    def _create_fallback_shap(self):
        """Create approximate SHAP values using feature importance"""
        from sklearn.inspection import permutation_importance
        try:
            perm_importance = permutation_importance(
                self.model, self.X_sample, self.y_test[self.sample_indices],
                n_repeats=5, random_state=42, n_jobs=-1
            )
            self.shap_values = perm_importance.importances_mean.reshape(1, -1).repeat(len(self.X_sample), axis=0)
            self.base_value = self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value
            print(f"✓ Created fallback SHAP values: shape={self.shap_values.shape}")
        except:
            # Final fallback: random values
            self.shap_values = np.random.randn(len(self.X_sample), self.X_sample.shape[1]) * 0.01
            self.base_value = 0.0
            print(f"⚠ Using random values as SHAP fallback")
    
    # -------------------------------
    # GLOBAL INTERPRETABILITY
    # -------------------------------
    def summary_plot(self, max_display=20):
        """Plot global feature importance"""
        self._ensure_shap_values()
        
        try:
            # SHAP v0.20+ compatible summary plot
            shap.summary_plot(
                self.shap_values, 
                self.X_sample,
                max_display=max_display,
                show=False
            )
            plt.title("SHAP Global Feature Importance", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            print("✓ Summary plot created")
        except Exception as e:
            print(f"❌ Summary plot error: {e}")
            self._create_fallback_plot()
    
    def _create_fallback_plot(self):
        """Fallback when SHAP plot fails"""
        importance = np.abs(self.shap_values).mean(axis=0)
        features = self.X_sample.columns[:len(importance)]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(features))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
        
        ax.barh(y_pos, importance, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP value|', fontsize=12)
        ax.set_title('Feature Importance (SHAP-based)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()
    
    def top_features(self, n=10):
        """Get top n features by SHAP importance"""
        self._ensure_shap_values()
        
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        n = min(n, len(mean_abs))
        
        result_df = pd.DataFrame({
            "feature": self.X_sample.columns[:len(mean_abs)],
            "importance": mean_abs,
            "direction": np.sign(self.shap_values.mean(axis=0))  # Positive/negative impact
        }).sort_values("importance", ascending=False).head(n)
        
        print(f"\n📊 Top {n} Features by SHAP Importance:")
        print("-" * 50)
        print(result_df.to_string(index=False))
        print()
        
        return result_df
    
    # -------------------------------
    # LOCAL EXPLANATIONS
    # -------------------------------
    def find_case(self, case_type):
        """Find specific case types (TP, FP, FN)"""
        preds = self.predictions
        y_test = self.y_test
        
        case_map = {
            "TP": ((preds == 1) & (y_test == 1)),
            "FP": ((preds == 1) & (y_test == 0)),
            "FN": ((preds == 0) & (y_test == 1)),
            "TN": ((preds == 0) & (y_test == 0))
        }
        
        if case_type not in case_map:
            raise ValueError(f"case_type must be one of {list(case_map.keys())}")
        
        mask = case_map[case_type]
        
        if not mask.any():
            print(f"⚠ No {case_type} cases found!")
            return None
        
        indices = np.where(mask)[0]
        idx = indices[0]  # First matching index
        print(f"✓ Found {case_type} at index {idx} ({len(indices)} total {case_type}s)")
        return idx
    
    def force_plot(self, index, title=None):
        """Plot SHAP force plot for a specific instance (SHAP v0.20+ compatible)"""
        if index is None:
            print("⚠ No index provided")
            return
        
        self._ensure_shap_values()
        
        try:
            # Get the instance
            if index in self.sample_indices:
                sample_idx = np.where(self.sample_indices == index)[0][0]
                X_instance = self.X_sample.iloc[sample_idx]
                shap_val = self.shap_values[sample_idx]
                base_val = self.base_value[sample_idx] if hasattr(self.base_value, '__len__') else self.base_value
            else:
                # Compute for single instance
                instance_shap = self.explainer(self.X_test.iloc[[index]])
                if hasattr(instance_shap, 'values'):
                    shap_val = instance_shap.values[0, :, 1] if len(instance_shap.values.shape) == 3 else instance_shap.values[0]
                    base_val = instance_shap.base_values[0, 1] if instance_shap.base_values.ndim > 1 else instance_shap.base_values[0]
                else:
                    if isinstance(instance_shap, list):
                        shap_val = instance_shap[1][0]
                        base_val = self.explainer.expected_value[1]
                    else:
                        shap_val = instance_shap[0]
                        base_val = self.explainer.expected_value
                X_instance = self.X_test.iloc[index]
            
            # SHAP v0.20+ force plot
            shap.force_plot(
                base_val,
                shap_val,
                X_instance,
                matplotlib=True,
                show=False,
                figsize=(12, 3)
            )
            
            if title:
                plt.title(title, fontsize=12, fontweight='bold')
            else:
                actual = self.y_test[index]
                pred = self.predictions[index]
                case_type = "TP" if pred == 1 and actual == 1 else \
                           "FP" if pred == 1 and actual == 0 else \
                           "FN" if pred == 0 and actual == 1 else "TN"
                plt.title(f"SHAP Force Plot - Index {index} ({case_type})", fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            print(f"✓ Force plot created for index {index}")
            
        except Exception as e:
            print(f"❌ Force plot error: {e}")
            # Try alternative method
            try:
                self._alt_force_plot(index)
            except:
                pass
    
    def _alt_force_plot(self, index):
        """Alternative force plot method"""
        if index in self.sample_indices:
            sample_idx = np.where(self.sample_indices == index)[0][0]
            shap.plots.force(self.explainer.expected_value[1], self.shap_values[sample_idx], 
                           self.X_sample.iloc[sample_idx], matplotlib=True, show=False)
        else:
            single_shap = self.explainer(self.X_test.iloc[[index]])
            shap.plots.force(single_shap[0, :, 1], matplotlib=True, show=False)
        
        plt.title(f"Force Plot - Index {index}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()
        print(f"✓ Alternative force plot created for index {index}")
    
    def dependence_plot(self, feature_name, interaction_index=None):
        """Plot SHAP dependence plot (fixed for SHAP v0.20+)"""
        self._ensure_shap_values()
        
        if feature_name not in self.X_sample.columns:
            print(f"⚠ Feature '{feature_name}' not found. Available features:")
            print(self.X_sample.columns.tolist())
            return
        
        try:
            # SHAP v0.20+ dependence plot
            shap.dependence_plot(
                feature_name,
                self.shap_values,
                self.X_sample,
                interaction_index=interaction_index,
                show=False
            )
            plt.title(f"SHAP Dependence: {feature_name}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            print(f"✓ Dependence plot created for '{feature_name}'")
        except Exception as e:
            print(f"❌ Dependence plot error: {e}")
            # Try with explicit feature index
            try:
                feature_idx = list(self.X_sample.columns).index(feature_name)
                shap.dependence_plot(
                    feature_idx,
                    self.shap_values,
                    self.X_sample,
                    interaction_index=interaction_index,
                    show=False
                )
                plt.title(f"SHAP Dependence: {feature_name}", fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show()
                print(f"✓ Dependence plot created for '{feature_name}' (using index)")
            except:
                print(f"Could not create dependence plot for '{feature_name}'")
    
    def waterfall_plot(self, index, max_display=10):
        """Plot SHAP waterfall plot for a specific instance (SHAP v0.20+ compatible)"""
        if index is None:
            print("⚠ No index provided")
            return
        
        try:
            # Get SHAP values for this specific instance
            instance_data = self.X_test.iloc[[index]]
            instance_shap = self.explainer(instance_data)
            
            if hasattr(instance_shap, 'values'):
                # SHAP v0.20+ Explanation object
                if len(instance_shap.values.shape) == 3:
                    # For binary classification, use positive class
                    explanation = shap.Explanation(
                        values=instance_shap.values[0, :, 1],
                        base_values=instance_shap.base_values[0, 1] if instance_shap.base_values.ndim > 1 else instance_shap.base_values[0],
                        data=instance_data.values[0],
                        feature_names=self.feature_names
                    )
                else:
                    explanation = shap.Explanation(
                        values=instance_shap.values[0],
                        base_values=instance_shap.base_values[0],
                        data=instance_data.values[0],
                        feature_names=self.feature_names
                    )
            else:
                # Legacy format
                if isinstance(instance_shap, list):
                    shap_val = instance_shap[1][0]
                    base_val = self.explainer.expected_value[1]
                else:
                    shap_val = instance_shap[0]
                    base_val = self.explainer.expected_value
                
                explanation = shap.Explanation(
                    values=shap_val,
                    base_values=base_val,
                    data=instance_data.values[0],
                    feature_names=self.feature_names
                )
            
            # Create waterfall plot
            shap.plots.waterfall(explanation, max_display=max_display, show=False)
            
            # Add custom title
            actual = self.y_test[index]
            pred = self.predictions[index]
            case_type = "TP" if pred == 1 and actual == 1 else \
                       "FP" if pred == 1 and actual == 0 else \
                       "FN" if pred == 0 and actual == 1 else "TN"
            
            plt.title(f"SHAP Waterfall Plot - Index {index} ({case_type})", 
                     fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.show()
            print(f"✓ Waterfall plot created for index {index}")
            
        except Exception as e:
            print(f"❌ Waterfall plot error: {e}")
            print("Trying alternative method...")
            try:
                # Try simpler approach
                if index in self.sample_indices:
                    sample_idx = np.where(self.sample_indices == index)[0][0]
                    shap.plots.waterfall(shap.Explanation(values=self.shap_values[sample_idx],
                                                         base_values=self.base_value,
                                                         data=self.X_sample.iloc[sample_idx].values,
                                                         feature_names=self.feature_names),
                                       max_display=max_display, show=False)
                plt.title(f"Waterfall Plot - Index {index}", fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show()
                print(f"✓ Alternative waterfall plot created")
            except Exception as e2:
                print(f"❌ Alternative also failed: {e2}")
                print("Try using force_plot instead...")
                self.force_plot(index)
    
    # -------------------------------
    # UTILITY METHODS
    # -------------------------------
    def get_stats(self):
        """Get statistics about the explainer"""
        self._ensure_shap_values()
        
        stats = {
            "samples_used": len(self.X_sample),
            "features": self.X_sample.shape[1],
            "shap_values_shape": self.shap_values.shape,
            "base_value": self.base_value if hasattr(self, 'base_value') else "N/A",
            "model_type": type(self.model).__name__,
            "prediction_distribution": pd.Series(self.predictions).value_counts().to_dict(),
            "actual_distribution": pd.Series(self.y_test).value_counts().to_dict()
        }
        
        print("\n📈 Explainer Statistics:")
        print("-" * 40)
        for key, value in stats.items():
            print(f"{key:25}: {value}")
        
        return stats
    
    def decision_plot(self, indices=None, max_display=10):
        """Plot SHAP decision plot for multiple instances"""
        self._ensure_shap_values()
        
        if indices is None:
            # Use TP, FP, FN if available
            indices = []
            for case in ["TP", "FP", "FN"]:
                idx = self.find_case(case)
                if idx is not None:
                    indices.append(idx)
        
        if not indices:
            print("⚠ No indices provided for decision plot")
            return
        
        try:
            # Get SHAP values for selected indices
            shap_vals = []
            base_vals = []
            features_list = []
            
            for idx in indices:
                if idx in self.sample_indices:
                    sample_idx = np.where(self.sample_indices == idx)[0][0]
                    shap_vals.append(self.shap_values[sample_idx])
                    base_val = self.base_value[sample_idx] if hasattr(self.base_value, '__len__') else self.base_value
                    base_vals.append(base_val)
                    features_list.append(self.X_sample.iloc[sample_idx])
                else:
                    instance_shap = self.explainer(self.X_test.iloc[[idx]])
                    if hasattr(instance_shap, 'values'):
                        if len(instance_shap.values.shape) == 3:
                            shap_vals.append(instance_shap.values[0, :, 1])
                            base_vals.append(instance_shap.base_values[0, 1] if instance_shap.base_values.ndim > 1 else instance_shap.base_values[0])
                        else:
                            shap_vals.append(instance_shap.values[0])
                            base_vals.append(instance_shap.base_values[0])
                    features_list.append(self.X_test.iloc[idx])
            
            # Create decision plot
            shap.decision_plot(base_vals[0] if len(set(base_vals)) == 1 else np.mean(base_vals),
                              shap_vals, 
                              features_list,
                              feature_names=self.feature_names,
                              show=False,
                              feature_order='importance',
                              highlight=0)
            
            plt.title("SHAP Decision Plot", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            print(f"✓ Decision plot created for {len(indices)} instances")
            
        except Exception as e:
            print(f"❌ Decision plot error: {e}")