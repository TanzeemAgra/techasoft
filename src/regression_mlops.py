#!/usr/bin/env python3
import os
import yaml
import pandas as pd
import numpy as np
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
import joblib

# MLflow Libraries
import mlflow
from mlflow.tracking import MlflowClient


class RegressionMLOps:
    """
    Comprehensive Regression MLOps Manager for automated ML workflows.
    """
    
    def __init__(self, config_path: str = "params.yaml"):
        """Initialize with configuration from YAML file."""
        self.config_path = config_path
        self.config = self._load_config()
        self.client = None
        self._setup_mlflow()
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            print(f"✅ Configuration loaded from {self.config_path}")
            
            # Display key configuration values
            print("📋 Key Configuration:")
            print(f"   Project: {config['base']['project']}")
            print(f"   Target Column: {config['base']['target_col']}")
            print(f"   Random State: {config['base']['random_state']}")
            print(f"   Base Alpha: {config['estimators']['ElasticNet']['params']['alpha']}")
            print(f"   Base L1 Ratio: {config['estimators']['ElasticNet']['params']['l1_ratio']}")
            
            # Show experiment parameters if available
            experiment_params = config['estimators']['ElasticNet'].get('experiment_params', {})
            if experiment_params:
                print(f"   Experiment Alpha Values: {experiment_params.get('alpha_values', 'Not configured')}")
                print(f"   Experiment L1 Ratio Values: {experiment_params.get('l1_ratio_values', 'Not configured')}")
            
            print(f"   MLflow URI: {config['mlflow_config']['remote_server_uri']}")
            print(f"   Experiment: {config['mlflow_config']['experiments_name']}")
            print(f"   Model Name: {config['mlflow_config']['registered_model_name']}")
            print(f"   Best Metric: {config.get('model_deployment', {}).get('metric_for_best_model', 'rmse')}")
            
            return config
        except Exception as e:
            print(f"❌ Failed to load config: {str(e)}")
            raise
    
    def _setup_mlflow(self):
        """Setup MLflow tracking and client."""
        try:
            mlflow_config = self.config["mlflow_config"]
            remote_server_uri = mlflow_config["remote_server_uri"]
            
            mlflow.set_tracking_uri(remote_server_uri)
            self.client = MlflowClient()
            
            print(f"✅ MLflow initialized: {remote_server_uri}")
        except Exception as e:
            print(f"❌ MLflow initialization failed: {str(e)}")
            raise
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load and preprocess training data."""
        try:
            # Configuration
            train_path = self.config["split_data"]["train_path"]
            test_path = self.config["split_data"]["test_path"]
            target_col = self.config["base"]["target_col"]
            
            print("📊 Loading data...")
            
            # Load datasets
            train_data = pd.read_csv(train_path, sep=",")
            test_data = pd.read_csv(test_path, sep=",")
            
            # Preprocessing: numeric columns only, fill missing values
            train_data = train_data.select_dtypes(include=[np.number]).fillna(0)
            test_data = test_data.select_dtypes(include=[np.number]).fillna(0)
            
            # Feature-target split
            X_train = train_data.drop(target_col, axis=1)
            X_test = test_data.drop(target_col, axis=1)
            y_train = train_data[target_col]
            y_test = test_data[target_col]
            
            print(f"✅ Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"❌ Data loading failed: {str(e)}")
            raise
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def train_and_register_model(self, alpha: float = None, l1_ratio: float = None) -> Dict:
        """Train ElasticNet model and register in MLflow."""
        try:
            # Load data
            X_train, X_test, y_train, y_test = self.load_data()
            
            # Configuration from params.yaml
            mlflow_config = self.config["mlflow_config"]
            experiment_name = mlflow_config["experiments_name"]
            registered_model_name = mlflow_config["registered_model_name"]
            run_name = mlflow_config.get("run_name", "mlops")
            random_state = self.config["base"]["random_state"]
            
            # Get alpha and l1_ratio from config if not provided
            if alpha is None:
                alpha = self.config["estimators"]["ElasticNet"]["params"]["alpha"]
            if l1_ratio is None:
                l1_ratio = self.config["estimators"]["ElasticNet"]["params"]["l1_ratio"]
            
            # Create or get experiment
            try:
                experiment = self.client.get_experiment_by_name(experiment_name)
                experiment_id = experiment.experiment_id if experiment else self.client.create_experiment(experiment_name)
            except:
                experiment_id = self.client.create_experiment(experiment_name)
            
            # MLflow run
            dynamic_run_name = f"{run_name}_alpha_{alpha}_l1_{l1_ratio}"
            with mlflow.start_run(experiment_id=experiment_id, run_name=dynamic_run_name) as run:
                print(f"🚀 Training: α={alpha}, l1_ratio={l1_ratio}")
                
                # Train model
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
                model.fit(X_train, y_train)
                
                # Predictions and metrics
                y_pred = model.predict(X_test)
                metrics = self.evaluate_model(y_test, y_pred)
                
                # Log to MLflow
                mlflow.log_param("alpha", alpha)
                mlflow.log_param("l1_ratio", l1_ratio)
                mlflow.log_param("random_state", random_state)
                
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, value)
                
                # Register model
                mlflow.sklearn.log_model(
                    model, 
                    "models", 
                    registered_model_name=registered_model_name
                )
                
                # Save local model
                model_path = self.config["model_dirs"]
                model_directory = os.path.dirname(model_path)
                os.makedirs(model_directory, exist_ok=True)
                
                # Create unique filename for each model
                unique_model_path = model_path.replace('.joblib', f'_alpha_{alpha}_l1_{l1_ratio}.joblib')
                joblib.dump(model, unique_model_path)
                
                # Also save as the main model file (latest trained)
                joblib.dump(model, model_path)
                
                result = {
                    'run_id': run.info.run_id,
                    'alpha': alpha,
                    'l1_ratio': l1_ratio,
                    **metrics,
                    'model_uri': f"runs:/{run.info.run_id}/models"
                }
                
                print(f"✅ Model trained - RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.3f}")
                return result
                
        except Exception as e:
            print(f"❌ Training failed for α={alpha}, l1_ratio={l1_ratio}: {str(e)}")
            return None
    
    def train_with_config_params(self) -> Dict:
        """Train a single model using parameters from config file."""
        print("🎯 Training model with config parameters...")
        
        # Get parameters from config
        alpha = self.config["estimators"]["ElasticNet"]["params"]["alpha"]
        l1_ratio = self.config["estimators"]["ElasticNet"]["params"]["l1_ratio"]
        
        print(f"📋 Config parameters: α={alpha}, l1_ratio={l1_ratio}")
        return self.train_and_register_model(alpha, l1_ratio)
    
    def validate_experiment_config(self) -> bool:
        """Validate experiment configuration and provide helpful messages."""
        try:
            experiment_config = self.config["estimators"]["ElasticNet"].get("experiment_params", {})
            
            if not experiment_config:
                print("⚠️  No experiment_params found in config. Using base parameters only.")
                print("💡 To configure parameter experimentation, add 'experiment_params' to your config:")
                print("   estimators:")
                print("     ElasticNet:")
                print("       experiment_params:")
                print("         alpha_values: [0.1, 0.5, 1.0]")
                print("         l1_ratio_values: [0.1, 0.5, 0.9]")
                return False
            
            alpha_values = experiment_config.get("alpha_values", [])
            l1_ratio_values = experiment_config.get("l1_ratio_values", [])
            
            if not alpha_values or not l1_ratio_values:
                print("⚠️  Incomplete experiment configuration!")
                if not alpha_values:
                    print("   Missing 'alpha_values' in experiment_params")
                if not l1_ratio_values:
                    print("   Missing 'l1_ratio_values' in experiment_params")
                return False
            
            # Validate parameter ranges
            for alpha in alpha_values:
                if alpha <= 0:
                    print(f"❌ Invalid alpha value: {alpha}. Must be positive.")
                    return False
            
            for l1_ratio in l1_ratio_values:
                if not (0 <= l1_ratio <= 1):
                    print(f"❌ Invalid l1_ratio value: {l1_ratio}. Must be between 0 and 1.")
                    return False
            
            total_combinations = len(alpha_values) * len(l1_ratio_values)
            if total_combinations > 50:
                print(f"⚠️  Large experiment: {total_combinations} combinations configured!")
                print("   Consider reducing parameter ranges for faster experimentation.")
            
            return True
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {str(e)}")
            return False
    
    def experiment_with_parameters(self) -> List[Dict]:
        """Train multiple models with different parameter combinations."""
        print("🔬 Starting parameter experimentation...")
        
        # Validate experiment configuration
        if not self.validate_experiment_config():
            print("🎯 Falling back to base parameters only...")
            base_alpha = self.config["estimators"]["ElasticNet"]["params"]["alpha"]
            base_l1_ratio = self.config["estimators"]["ElasticNet"]["params"]["l1_ratio"]
            result = self.train_and_register_model(base_alpha, base_l1_ratio)
            return [result] if result else []
        
        # Get base parameters from config
        base_alpha = self.config["estimators"]["ElasticNet"]["params"]["alpha"]
        base_l1_ratio = self.config["estimators"]["ElasticNet"]["params"]["l1_ratio"]
        
        # Get experiment parameter values from config
        experiment_config = self.config["estimators"]["ElasticNet"]["experiment_params"]
        alpha_values = experiment_config["alpha_values"]
        l1_ratio_values = experiment_config["l1_ratio_values"]
        
        # Ensure base parameters are included
        if base_alpha not in alpha_values:
            alpha_values.append(base_alpha)
        if base_l1_ratio not in l1_ratio_values:
            l1_ratio_values.append(base_l1_ratio)
        
        # Remove duplicates and sort
        alpha_values = sorted(list(set(alpha_values)))
        l1_ratio_values = sorted(list(set(l1_ratio_values)))
        
        param_combinations = [(alpha, l1_ratio) for alpha in alpha_values for l1_ratio in l1_ratio_values]
        
        print(f"📊 Parameter experimentation configuration:")
        print(f"   Total combinations: {len(param_combinations)}")
        print(f"   Alpha values: {alpha_values}")
        print(f"   L1-ratio values: {l1_ratio_values}")
        print(f"   Base config: α={base_alpha}, l1_ratio={base_l1_ratio}")
        
        results = []
        total_combinations = len(param_combinations)
        
        for i, (alpha, l1_ratio) in enumerate(param_combinations, 1):
            print(f"📈 Progress: {i}/{total_combinations}")
            result = self.train_and_register_model(alpha, l1_ratio)
            if result:
                results.append(result)
        
        print(f"✅ Experimentation complete: {len(results)}/{total_combinations} models trained")
        return results
    
    def find_best_model(self) -> Optional[Dict]:
        """Find the best model based on configured metric."""
        try:
            # Configuration
            mlflow_config = self.config["mlflow_config"]
            deployment_config = self.config.get("model_deployment", {})
            
            experiment_name = mlflow_config["experiments_name"]
            target_metric = deployment_config.get("metric_for_best_model", "rmse")
            sort_order = deployment_config.get("sort_order", "DESC")
            
            # Get experiment
            experiment = self.client.get_experiment_by_name(experiment_name)
            if not experiment:
                print(f"❌ Experiment '{experiment_name}' not found!")
                return None
            
            # Search runs
            order_clause = f"metrics.{target_metric} {sort_order}"
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[order_clause]
            )
            
            if not runs:
                print("❌ No runs found!")
                return None
            
            print(f"📊 Analyzing {len(runs)} runs for best {target_metric}...")
            
            # Display top runs
            print(f"\n🏆 Top 5 Models by {target_metric.upper()}:")
            for i, run in enumerate(runs[:5], 1):
                rmse = run.data.metrics.get('rmse', 0)
                mae = run.data.metrics.get('mae', 0)
                r2 = run.data.metrics.get('r2', 0)
                alpha = run.data.params.get('alpha', 'Unknown')
                l1_ratio = run.data.params.get('l1_ratio', 'Unknown')
                
                print(f"  {i}. α={alpha}, l1={l1_ratio} → RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")
            
            # Best model
            best_run = runs[0]
            best_model = {
                'run_id': best_run.info.run_id,
                'rmse': best_run.data.metrics.get('rmse', 0),
                'mae': best_run.data.metrics.get('mae', 0),
                'r2': best_run.data.metrics.get('r2', 0),
                'alpha': best_run.data.params.get('alpha', 'Unknown'),
                'l1_ratio': best_run.data.params.get('l1_ratio', 'Unknown'),
                'metric_name': target_metric,
                'metric_value': best_run.data.metrics.get(target_metric, 0),
                'model_uri': f"runs:/{best_run.info.run_id}/models"
            }
            
            print(f"\n🎯 Best Model Selected:")
            print(f"   Run ID: {best_model['run_id']}")
            print(f"   Parameters: α={best_model['alpha']}, l1_ratio={best_model['l1_ratio']}")
            print(f"   Best {target_metric.upper()}: {best_model['metric_value']:.4f}")
            print(f"   RMSE: {best_model['rmse']:.4f}")
            print(f"   MAE: {best_model['mae']:.4f}")
            print(f"   R2: {best_model['r2']:.4f}")
            
            return best_model
            
        except Exception as e:
            print(f"❌ Best model selection failed: {str(e)}")
            return None
    
    def deploy_to_production(self, best_model: Dict = None) -> bool:
        """Deploy best model to production and organize others in staging."""
        try:
            if not best_model:
                best_model = self.find_best_model()
                if not best_model:
                    return False
            
            # Configuration
            mlflow_config = self.config["mlflow_config"]
            deployment_config = self.config.get("model_deployment", {})
            
            registered_model_name = mlflow_config["registered_model_name"]
            production_stage = deployment_config.get("production_stage", "Production")
            staging_stage = deployment_config.get("staging_stage", "Staging")
            
            print(f"🚀 Deploying best model to {production_stage}...")
            
            # Register model if not already registered
            try:
                new_version = mlflow.register_model(
                    model_uri=best_model['model_uri'],
                    name=registered_model_name,
                    tags={
                        "best_model": "true",
                        "deployment_timestamp": datetime.now().isoformat(),
                        "source_run": best_model['run_id'],
                        "rmse": str(best_model['rmse']),
                        "mae": str(best_model['mae']),
                        "r2": str(best_model['r2']),
                        "alpha": str(best_model['alpha']),
                        "l1_ratio": str(best_model['l1_ratio'])
                    }
                )
                target_version = new_version.version
                print(f"✅ Model registered as version {target_version}")
                
            except Exception as e:
                # Use latest version if registration fails
                all_versions = self.client.search_model_versions(f"name='{registered_model_name}'")
                if not all_versions:
                    print(f"❌ No model versions found for {registered_model_name}")
                    return False
                target_version = max(all_versions, key=lambda x: int(x.version)).version
                print(f"⚠️ Using existing version {target_version}: {str(e)}")
            
            # Deploy to production
            try:
                self.client.transition_model_version_stage(
                    name=registered_model_name,
                    version=target_version,
                    stage=production_stage,
                    archive_existing_versions=True
                )
                print(f"🏆 Version {target_version} deployed to {production_stage}!")
                
            except Exception as e:
                print(f"❌ Production deployment failed: {str(e)}")
                return False
            
            # Organize other versions in staging
            try:
                all_versions = self.client.search_model_versions(f"name='{registered_model_name}'")
                staging_count = 0
                
                for version in all_versions:
                    if (version.version != target_version and 
                        version.current_stage not in [production_stage, "Archived"]):
                        try:
                            self.client.transition_model_version_stage(
                                name=registered_model_name,
                                version=version.version,
                                stage=staging_stage
                            )
                            staging_count += 1
                        except:
                            continue
                
                print(f"📦 Organized {staging_count} versions in {staging_stage}")
                
            except Exception as e:
                print(f"⚠️ Staging organization issues: {str(e)}")
            
            # Verify deployment
            try:
                production_models = self.client.get_latest_versions(registered_model_name, stages=[production_stage])
                if production_models:
                    prod_version = production_models[0]
                    print(f"✅ Deployment verified: Version {prod_version.version} active in {production_stage}")
                    
                    # Display final status
                    self.show_deployment_status()
                    return True
                else:
                    print(f"❌ Verification failed: No model in {production_stage}")
                    return False
                    
            except Exception as e:
                print(f"❌ Deployment verification failed: {str(e)}")
                return False
            
        except Exception as e:
            print(f"❌ Deployment process failed: {str(e)}")
            return False
    
    def show_deployment_status(self):
        """Display current deployment status."""
        try:
            registered_model_name = self.config["mlflow_config"]["registered_model_name"]
            
            print(f"\n📊 Deployment Status for '{registered_model_name}':")
            print("=" * 60)
            
            # Get all versions
            all_versions = self.client.search_model_versions(f"name='{registered_model_name}'")
            
            # Group by stage
            stages = {}
            for version in all_versions:
                stage = version.current_stage
                if stage not in stages:
                    stages[stage] = []
                stages[stage].append(version)
            
            # Display each stage
            for stage, versions in stages.items():
                if stage == "Production":
                    icon = "🏆"
                elif stage == "Staging":
                    icon = "📦"
                elif stage == "Archived":
                    icon = "📁"
                else:
                    icon = "⚪"
                
                print(f"\n{icon} {stage.upper()}: {len(versions)} versions")
                for version in sorted(versions, key=lambda x: int(x.version), reverse=True)[:3]:
                    print(f"   Version {version.version}")
            
            # MLflow UI links
            remote_uri = self.config["mlflow_config"]["remote_server_uri"]
            print(f"\n🌐 MLflow UI: {remote_uri}/#/models/{registered_model_name}")
            
        except Exception as e:
            print(f"❌ Status display failed: {str(e)}")
    
    def run_complete_mlops(self) -> bool:
        """Execute complete MLOps workflow: experiment → find best → deploy."""
        print("🎯 Starting Complete Regression MLOps Workflow")
        print("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Parameter experimentation
            print("\n🔬 STEP 1: PARAMETER EXPERIMENTATION")
            print("-" * 40)
            experiment_results = self.experiment_with_parameters()
            
            if not experiment_results:
                print("❌ No successful experiments!")
                return False
            
            # Step 2: Find best model
            print("\n🎯 STEP 2: BEST MODEL IDENTIFICATION")
            print("-" * 40)
            best_model = self.find_best_model()
            
            if not best_model:
                print("❌ Could not identify best model!")
                return False
            
            # Step 3: Deploy to production
            print("\n🚀 STEP 3: PRODUCTION DEPLOYMENT")
            print("-" * 40)
            deployment_success = self.deploy_to_production(best_model)
            
            # Summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "=" * 60)
            print("🎉 MLOPS WORKFLOW COMPLETED!")
            print("=" * 60)
            print(f"   Duration: {duration}")
            print(f"   Experiments: {len(experiment_results)} models trained")
            print(f"   Best Model RMSE: {best_model['rmse']:.4f}")
            print(f"   Best Model R2: {best_model['r2']:.4f}")
            print(f"   Deployment: {'✅ Success' if deployment_success else '❌ Failed'}")
            
            return deployment_success
            
        except Exception as e:
            print(f"❌ MLOps workflow failed: {str(e)}")
            return False


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Regression MLOps Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python regression_Mlops.py                           # Run complete MLOps workflow
  python regression_Mlops.py --action train-config    # Train model with config parameters
  python regression_Mlops.py --action experiment      # Run parameter experiments only
  python regression_Mlops.py --action find-best       # Find best model only  
  python regression_Mlops.py --action deploy          # Deploy best model only
  python regression_Mlops.py --action status          # Show deployment status
        """
    )
    
    parser.add_argument(
        "--config", 
        default="params.yaml", 
        help="Configuration file path (default: params.yaml)"
    )
    
    parser.add_argument(
        "--action", 
        choices=["complete", "experiment", "train-config", "find-best", "deploy", "status"],
        default="complete",
        help="Action to perform (default: complete)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize MLOps manager
        mlops = RegressionMLOps(args.config)
        
        # Execute requested action
        if args.action == "complete":
            success = mlops.run_complete_mlops()
            return 0 if success else 1
            
        elif args.action == "train-config":
            result = mlops.train_with_config_params()
            if result:
                print(f"✅ Model trained with config params - RMSE: {result['rmse']:.4f}")
            else:
                print("❌ Training failed")
                return 1
            
        elif args.action == "experiment":
            results = mlops.experiment_with_parameters()
            print(f"🧪 Experimentation complete: {len(results)} models")
            
        elif args.action == "find-best":
            best_model = mlops.find_best_model()
            if best_model:
                print(f"🏆 Best model found: RMSE = {best_model['rmse']:.4f}")
            else:
                print("❌ No best model found")
                return 1
                
        elif args.action == "deploy":
            success = mlops.deploy_to_production()
            print(f"🚀 Deployment: {'✅ Success' if success else '❌ Failed'}")
            return 0 if success else 1
            
        elif args.action == "status":
            mlops.show_deployment_status()
        
        return 0
        
    except Exception as e:
        print(f"❌ Application failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())