#!/usr/bin/env python3
# run_vnu_kag.py
import os
import sys
import argparse
import streamlit.web.cli as stcli
import config

def check_artifacts_exist():
    """Check if KAG artifacts exist."""
    required_files = [config.FAISS_INDEX_PATH, config.GRAPH_PATH, config.DOC_STORE_PATH]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    return len(missing_files) == 0, missing_files

def run_builder():
    """Run the KAG builder process."""
    print("Starting KAG Builder process...")
    try:
        import build_kag_vnu
        build_kag_vnu.main()
        print("KAG Builder process completed successfully.")
        return True
    except Exception as e:
        print(f"Error running KAG Builder: {e}")
        return False

def run_ui():
    """Run the Streamlit UI."""
    print("Starting KAG VNU Bot UI...")
    
    # Check if artifacts exist
    artifacts_exist, missing_files = check_artifacts_exist()
    if not artifacts_exist:
        print(f"Error: Missing KAG artifacts: {', '.join(missing_files)}")
        print("Please run the builder first with: python run_vnu_kag.py --build")
        return False
    
    # Get app_ui.py path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_file = os.path.join(current_dir, "app_ui.py")
    
    if not os.path.exists(app_file):
        print(f"Error: app_ui.py not found at {app_file}")
        return False
    
    # Run Streamlit
    sys.argv = ["streamlit", "run", app_file]
    stcli.main()
    return True

def main():
    """Main function to run either builder or UI based on arguments."""
    parser = argparse.ArgumentParser(description="Run KAG VNU Bot Builder or UI")
    parser.add_argument("--build", action="store_true", help="Run the KAG Builder process")
    parser.add_argument("--ui", action="store_true", help="Run the Streamlit UI")
    
    args = parser.parse_args()
    
    # If no args provided, default to UI if artifacts exist, otherwise show help
    if not args.build and not args.ui:
        artifacts_exist, _ = check_artifacts_exist()
        if artifacts_exist:
            args.ui = True
        else:
            print("KAG artifacts not found. Please run the builder first.")
            parser.print_help()
            return
    
    if args.build:
        success = run_builder()
        if success and args.ui:
            run_ui()
    elif args.ui:
        run_ui()

if __name__ == "__main__":
    main() 