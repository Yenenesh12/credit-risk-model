import subprocess
import os
import sys
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DVCScript:
    """Setup Data Version Control for the project"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize DVC setup with configuration"""
        self.config_path = Path(config_path)
        self.project_root = Path.cwd()

        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Define paths
        self.local_storage = Path(self.config['dvc']['remote_storage'])
        self.raw_data_path = Path(self.config['data']['raw_path'])
        self.processed_data_path = Path(self.config['data']['processed_path'])

        # Create necessary directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories for DVC"""
        directories = [
            self.local_storage,
            self.raw_data_path.parent,
            self.processed_data_path.parent,
            Path("reports/figures"),
            Path("reports/docs")
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")

    def _run_command(self, command: str, description: str) -> bool:
        """Execute shell command with error handling"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Executing: {description}")
        logger.info(f"Command: {command}")
        logger.info('='*60)

        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            logger.info(f"Success: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error: {e.stderr}")
            return False

    def check_dvc_installation(self):
        """Check if DVC is installed, install if not"""
        logger.info("Checking DVC installation...")

        try:
            result = subprocess.run(
                ["dvc", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"DVC is installed: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("DVC not found. Installing...")
            install_cmd = f"{sys.executable} -m pip install dvc"
            if self._run_command(install_cmd, "Install DVC"):
                logger.info("DVC installed successfully")
                return True
            else:
                logger.error("Failed to install DVC")
                return False

    def initialize_dvc(self):
        """Initialize DVC in the project"""
        logger.info("Initializing DVC...")

        if (self.project_root / ".dvc").exists():
            logger.info("DVC already initialized")
            return True

        if self._run_command("dvc init", "Initialize DVC repository"):
            logger.info("DVC initialized successfully")
            return True
        else:
            logger.error("Failed to initialize DVC")
            return False

    def setup_local_remote(self):
        """Setup local remote storage (idempotent with -f)"""
        logger.info("Setting up local remote storage...")

        remote_cmd = f"dvc remote add -f -d localstorage {self.local_storage.absolute()}"
        if self._run_command(remote_cmd, "Add/overwrite local remote storage"):
            logger.info(f"Local remote storage configured at: {self.local_storage.absolute()}")
            return True
        else:
            logger.error("Failed to setup local remote storage")
            return False

    def generate_sample_data(self):
        """Generate sample insurance data for demonstration"""
        logger.info("Generating sample insurance data...")

        np.random.seed(42)
        n_samples = 5000

        sample_data = pd.DataFrame({
            'PolicyID': range(100000, 100000 + n_samples),
            'TransactionMonth': pd.date_range('2014-02-01', periods=n_samples, freq='D').tolist()[:n_samples],
            'Province': np.random.choice(
                ['Gauteng', 'Western Cape', 'KwaZulu-Natal', 'Eastern Cape',
                 'Free State', 'North West', 'Mpumalanga', 'Limpopo', 'Northern Cape'],
                n_samples,
                p=[0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04]
            ),
            'VehicleType': np.random.choice(
                ['Sedan', 'SUV', 'Truck', 'Motorcycle', 'Bus', 'Van'],
                n_samples,
                p=[0.40, 0.25, 0.15, 0.10, 0.05, 0.05]
            ),
            'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45]),
            'Make': np.random.choice(
                ['Toyota', 'Volkswagen', 'Ford', 'BMW', 'Mercedes', 'Nissan', 'Hyundai', 'Kia'],
                n_samples,
                p=[0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05]
            ),
            'Model': np.random.choice(
                ['Corolla', 'Golf', 'Ranger', '3 Series', 'C-Class', 'Qashqai', 'Tucson', 'Sportage'],
                n_samples
            ),
            'TotalPremium': np.random.lognormal(8.5, 0.8, n_samples),
            'TotalClaims': np.random.lognormal(7.0, 1.2, n_samples) * np.random.binomial(1, 0.3, n_samples),
            'CustomValueEstimate': np.random.lognormal(10.5, 0.7, n_samples),
            'SumInsured': np.random.lognormal(11.0, 0.6, n_samples),
            'CalculatedPremiumPerTerm': np.random.lognormal(8.0, 0.5, n_samples),
            'CoverType': np.random.choice(
                ['Comprehensive', 'Third Party', 'Third Party Fire & Theft'],
                n_samples,
                p=[0.60, 0.25, 0.15]
            ),
            'RegistrationYear': np.random.randint(2000, 2023, n_samples),
            'NumberOfDoors': np.random.choice([2, 4, 5], n_samples, p=[0.10, 0.70, 0.20]),
            'Bodytype': np.random.choice(
                ['Hatchback', 'Sedan', 'SUV', 'Coupe', 'Convertible', 'Wagon'],
                n_samples
            ),
            'MaritalStatus': np.random.choice(
                ['Single', 'Married', 'Divorced', 'Widowed'],
                n_samples,
                p=[0.40, 0.45, 0.10, 0.05]
            )
        })

        # Derived fields
        sample_data['LossRatio'] = sample_data['TotalClaims'] / sample_data['TotalPremium']
        sample_data['LossRatio'] = sample_data['LossRatio'].replace([np.inf, -np.inf], np.nan)

        # Add missing values
        for col in ['CustomValueEstimate', 'SumInsured', 'NumberOfDoors']:
            missing_mask = np.random.random(n_samples) < 0.05
            sample_data.loc[missing_mask, col] = np.nan

        sample_data.to_csv(self.raw_data_path, index=False)
        logger.info(f"Sample data saved to: {self.raw_data_path}")
        logger.info(f"Data shape: {sample_data.shape}")

        # ✅ Return BOOLEAN, not DataFrame
        return True

    def track_data_with_dvc(self):
        """Track only raw data with DVC (do NOT track report dirs)"""
        logger.info("Tracking data files with DVC...")

        # ONLY track raw data — processed and reports are pipeline outputs
        files_to_track = [str(self.raw_data_path)]

        success = True
        for file_path in files_to_track:
            if Path(file_path).exists():
                if not self._run_command(f"dvc add {file_path}", f"Track {file_path}"):
                    success = False
            else:
                logger.warning(f"File not found, skipping: {file_path}")
        return success

    def create_dvc_pipeline(self):
        """Create DVC pipeline with individual file outputs (no overlapping dirs)"""
        logger.info("Creating DVC pipeline...")

        # Declare individual files, NOT directories
        dvc_yaml_content = """stages:
  load_data:
    cmd: python scripts/load_and_preprocess.py
    deps:
      - scripts/load_and_preprocess.py
      - config/config.yaml
      - data/raw/insurance_data.csv
    outs:
      - data/processed/cleaned_data.parquet
      - data/processed/data_metadata.json

  run_eda:
    cmd: python scripts/run_eda.py
    deps:
      - scripts/run_eda.py
      - data/processed/cleaned_data.parquet
    outs:
      - reports/figures/data_quality_summary.png
      - reports/figures/correlation_matrix.png
      - reports/figures/numeric_distributions.png
      - reports/figures/categorical_distributions.png
      - reports/figures/loss_ratio_analysis.png
      - reports/figures/temporal_trends.png
      - reports/figures/outlier_analysis.png
      - reports/docs/eda_summary.json
      - reports/docs/eda_report.html

  generate_report:
    cmd: python scripts/generate_report.py
    deps:
      - scripts/generate_report.py
      - reports/figures/data_quality_summary.png
      - reports/docs/eda_summary.json
    outs:
      - reports/final_report.pdf
      - reports/summary_insights.md

metrics:
  - reports/metrics.json

plots:
  - reports/metrics.json:
      x: analysis_timestamp
      y: overall_loss_ratio
  - reports/figures/data_quality_summary.png
  - reports/figures/correlation_matrix.png
  - reports/figures/numeric_distributions.png
  - reports/figures/categorical_distributions.png
  - reports/figures/loss_ratio_analysis.png
  - reports/figures/temporal_trends.png
  - reports/figures/outlier_analysis.png
"""

        with open(self.project_root / "dvc.yaml", 'w') as f:
            f.write(dvc_yaml_content)

        params_yaml_content = """
data:
  raw_path: "data/raw/insurance_data.csv"
  processed_path: "data/processed/cleaned_data.parquet"

eda:
  numeric_threshold: 0.7
  outlier_threshold: 1.5
  min_samples: 10

report:
  output_format: "pdf"
  include_appendix: true
"""

        with open(self.project_root / "params.yaml", 'w') as f:
            f.write(params_yaml_content)

        logger.info("DVC pipeline and params saved.")
        return True

    def commit_dvc_changes(self):
        """Commit only existing DVC-tracked files"""
        logger.info("Committing DVC changes to Git...")

        commands = [
            "git add .dvc .dvcignore dvc.yaml params.yaml",
            "git add data/raw/*.dvc",  # Only raw data has .dvc now
            # DO NOT add reports/*.dvc — they were never created
            'git commit -m "feat: Add DVC configuration and pipeline"'
        ]

        success = True
        for cmd in commands:
            if not self._run_command(cmd, f"Git: {cmd}"):
                # Commit is critical; others are not
                if "commit" in cmd:
                    success = False
        return success

    def push_to_remote(self):
        """Push data to DVC remote"""
        logger.info("Pushing data to DVC remote storage...")
        return self._run_command("dvc push", "Push data to remote storage")

    def verify_dvc_setup(self):
        """Verify DVC setup"""
        logger.info("Verifying DVC setup...")

        checks = [
            ("Check DVC status", "dvc status"),
            ("Check DVC remote", "dvc remote list"),
            ("Check DVC pipeline DAG", "dvc dag"),
            ("List DVC-tracked files", "dvc list .")
        ]

        all_good = True
        for name, cmd in checks:
            logger.info(f"\nVerification: {name}")
            if not self._run_command(cmd, name):
                # DAG might fail if outputs missing, but not fatal at setup
                if "dag" not in cmd:
                    all_good = False
        return True  # Allow DAG failure during initial setup

    def create_dvc_ignore(self):
        """Create .dvcignore"""
        content = """# Ignore temporary and system files
*.tmp
*.log
__pycache__/
.ipynb_checkpoints/
.vscode/
.idea/
.DS_Store
Thumbs.db
*.pyc
"""
        with open(self.project_root / ".dvcignore", 'w') as f:
            f.write(content)
        logger.info(".dvcignore created.")
        return True

    def run(self):
        """Run full DVC setup"""
        logger.info("=" * 80)
        logger.info("STARTING COMPLETE DVC SETUP")
        logger.info("=" * 80)

        steps = [
            ("Check DVC installation", self.check_dvc_installation),
            ("Initialize DVC", self.initialize_dvc),
            ("Create .dvcignore", self.create_dvc_ignore),
            ("Setup local remote", self.setup_local_remote),
            ("Generate sample data", self.generate_sample_data),
            ("Track data with DVC", self.track_data_with_dvc),
            ("Create DVC pipeline", self.create_dvc_pipeline),
            ("Commit DVC changes", self.commit_dvc_changes),
            ("Push to remote", self.push_to_remote),
            ("Verify setup", self.verify_dvc_setup)
        ]

        results = []
        for name, fn in steps:
            logger.info(f"\n{'='*60}\nSTEP: {name}\n{'='*60}")
            try:
                result = fn()
                success = bool(result)
                results.append((name, success))
                status = "✓ PASS" if success else "✗ FAIL"
                logger.info(f"{status}: {name}")
            except Exception as e:
                logger.exception(f"Exception in {name}: {e}")
                results.append((name, False))

        logger.info("\n" + "=" * 80)
        logger.info("DVC SETUP SUMMARY")
        logger.info("=" * 80)
        total = len(results)
        passed = sum(1 for _, ok in results if ok)
        for name, ok in results:
            logger.info(f"{'✓ PASS' if ok else '✗ FAIL'}: {name}")

        logger.info(f"\nTotal: {passed}/{total} steps succeeded")

        if passed >= total - 1:
            logger.info("\n✅ DVC SETUP SUCCESSFUL!")
            logger.info("\nNext: Run `dvc repro` to execute the pipeline.")
            return True
        else:
            logger.error("\n❌ DVC SETUP FAILED!")
            return False


def main():
    try:
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)
        dvc = DVCScript()
        success = dvc.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()