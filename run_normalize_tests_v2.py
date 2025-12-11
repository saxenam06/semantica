import unittest
import sys
import os

def run_tests():
    print("SCRIPT STARTED")
    log_path = os.path.join(os.getcwd(), "normalize_results_v3.log")
    print(f"Writing log to {log_path}")
    
    # Ensure we can import from semantica
    sys.path.append(os.getcwd())
    
    try:
        loader = unittest.TestLoader()
        start_dir = 'tests/normalize'
        print(f"Discovering tests in {start_dir}")
        suite = loader.discover(start_dir)
        
        print(f"Discovered {suite.countTestCases()} tests.")
        
        runner = unittest.TextTestRunner(verbosity=2)
        
        # Open a log file to write results
        with open(log_path, 'w') as f:
            f.write("Test Execution Log:\n")
            f.write("===================\n\n")
            
            # Use a custom runner that prints to both stdout and the file
            class TeeStream:
                def __init__(self, stream1, stream2):
                    self.stream1 = stream1
                    self.stream2 = stream2
                def write(self, data):
                    self.stream1.write(data)
                    self.stream2.write(data)
                def flush(self):
                    self.stream1.flush()
                    self.stream2.flush()
            
            runner = unittest.TextTestRunner(stream=TeeStream(sys.stdout, f), verbosity=2)
            result = runner.run(suite)
        
        if result.wasSuccessful():
            print("ALL TESTS PASSED")
            sys.exit(0)
        else:
            print("SOME TESTS FAILED")
            sys.exit(1)
            
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
