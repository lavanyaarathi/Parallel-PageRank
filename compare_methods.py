#!/usr/bin/env python3
"""
Quick comparison script to showcase differences between PageRank methods
Runs all implementations and displays a side-by-side comparison
"""

import subprocess
import time
import sys
import os
from pathlib import Path

def run_command(cmd, timeout=60):
    """Run a command and return output and execution time"""
    try:
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=False)
        elapsed = time.time() - start
        
        if result.returncode == 0:
            return {
                'success': True,
                'output': result.stdout,
                'time': elapsed,
                'error': None
            }
        else:
            return {
                'success': False,
                'output': result.stdout,
                'time': elapsed,
                'error': result.stderr
            }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'output': '',
            'time': timeout,
            'error': 'Timeout'
        }
    except FileNotFoundError:
        return {
            'success': False,
            'output': '',
            'time': 0,
            'error': 'Executable not found'
        }

def extract_metrics(output):
    """Extract key metrics from PageRank output"""
    import re
    
    metrics = {
        'iterations': None,
        'time': None,
        'converged': False,
        'final_l1': None,
        'final_max_error': None
    }
    
    # Extract iterations - try multiple patterns
    # First try to find explicit iteration count
    iter_match = re.search(r'Total iterations: (\d+)', output)
    if iter_match:
        metrics['iterations'] = int(iter_match.group(1))
    else:
        # Try to count iteration lines
        iter_matches = re.findall(r'Iteration (\d+)', output)
        if iter_matches:
            # Get the last (highest) iteration number
            iterations = [int(m) for m in iter_matches]
            if iterations:
                # If we see "Iteration 0", "Iteration 1", etc., the count is max + 1
                # But if we see "Iteration 1", "Iteration 2", the count is just max
                max_iter = max(iterations)
                # Check if we have iteration 0
                if 0 in iterations:
                    metrics['iterations'] = max_iter + 1
                else:
                    metrics['iterations'] = max_iter
    
    # Extract time - try multiple patterns (Serial/Pthreads use "Totaltime", MPI uses "Total time")
    time_patterns = [
        r'Totaltime = ([\d.]+) seconds',
        r'Total time = ([\d.]+) seconds',
        r'time = ([\d.]+) seconds',
    ]
    for pattern in time_patterns:
        time_match = re.search(pattern, output)
        if time_match:
            try:
                metrics['time'] = float(time_match.group(1))
                break
            except (ValueError, IndexError):
                continue
    
    # Extract convergence
    if 'Converged' in output or 'converged' in output.lower():
        metrics['converged'] = True
    
    # Extract last iteration metrics - try multiple patterns
    # Pattern 1: "Iteration X, Max Error = Y, L1 Norm = Z"
    pattern1 = r'Iteration (\d+)[^0-9]*Max Error[^0-9]*([\d.eE+-]+)[^0-9]*L1 Norm[^0-9]*([\d.eE+-]+)'
    matches1 = re.findall(pattern1, output)
    if matches1:
        last = matches1[-1]
        try:
            metrics['final_max_error'] = float(last[1])
            metrics['final_l1'] = float(last[2])
        except (ValueError, IndexError):
            pass
    
    # Pattern 2: "Max Error = X, L1 Norm = Y" (simpler format)
    if metrics['final_l1'] is None:
        pattern2 = r'Max Error[^0-9]*([\d.eE+-]+)[^0-9]*L1 Norm[^0-9]*([\d.eE+-]+)'
        matches2 = re.findall(pattern2, output)
        if matches2:
            last = matches2[-1]
            try:
                metrics['final_max_error'] = float(last[0])
                metrics['final_l1'] = float(last[1])
            except (ValueError, IndexError):
                pass
    
    # Pattern 3: Just L1 norm at the end
    if metrics['final_l1'] is None:
        l1_match = re.search(r'L1[^0-9]*Norm[^0-9]*([\d.eE+-]+)', output)
        if l1_match:
            try:
                metrics['final_l1'] = float(l1_match.group(1))
            except (ValueError, IndexError):
                pass
    
    return metrics

def print_comparison_table(results):
    """Print a formatted comparison table"""
    print("\n" + "=" * 80)
    print("PAGERANK METHOD COMPARISON")
    print("=" * 80)
    print(f"{'Method':<20} {'Status':<12} {'Time (s)':<12} {'Iterations':<12} {'L1 Norm':<12}")
    print("-" * 80)
    
    for method, result in results.items():
        if result['success']:
            metrics = result['metrics']
            status = "[OK] Success"
            time_str = f"{metrics.get('time', 0):.4f}" if metrics.get('time') else "N/A"
            iter_str = str(metrics.get('iterations', 'N/A'))
            l1_str = f"{metrics.get('final_l1', 0):.6f}" if metrics.get('final_l1') else "N/A"
        else:
            status = "[X] Failed"
            time_str = "N/A"
            iter_str = "N/A"
            l1_str = "N/A"
        
        print(f"{method:<20} {status:<12} {time_str:<12} {iter_str:<12} {l1_str:<12}")
    
    print("=" * 80)

def print_detailed_comparison(results):
    """Print detailed comparison with explanations"""
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    
    for method, result in results.items():
        print(f"\n{method}")
        print("-" * 80)
        
        if result['success']:
            metrics = result['metrics']
            print(f"  Status: [OK] Success")
            # Safely handle None values for time
            time_val = metrics.get('time')
            if time_val is not None:
                print(f"  Execution Time: {time_val:.4f} seconds")
            else:
                print(f"  Execution Time: N/A (not reported in output)")
            print(f"  Iterations: {metrics.get('iterations', 'N/A')}")
            # Safely handle None values for L1 norm
            l1_val = metrics.get('final_l1')
            if l1_val is not None:
                print(f"  Final L1 Norm: {l1_val:.6e}")
            else:
                print(f"  Final L1 Norm: N/A")
            # Safely handle None values for max error
            max_err = metrics.get('final_max_error')
            if max_err is not None:
                print(f"  Final Max Error: {max_err:.6e}")
            else:
                print(f"  Final Max Error: N/A")
            print(f"  Converged: {'Yes' if metrics.get('converged') else 'No'}")
        else:
            print(f"  Status: [X] Failed")
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    # Calculate speedup if serial succeeded
    if results.get('Serial', {}).get('success'):
        serial_time = results['Serial']['metrics'].get('time')
        if serial_time is not None and serial_time > 0:
            print("\n" + "=" * 80)
            print("SPEEDUP ANALYSIS")
            print("=" * 80)
            print(f"Baseline (Serial): {serial_time:.4f} seconds\n")
            
            for method in ['Pthreads', 'MPI']:
                if results.get(method, {}).get('success'):
                    method_time = results[method]['metrics'].get('time')
                    if method_time is not None and method_time > 0:
                        speedup = serial_time / method_time
                        print(f"{method}: {speedup:.2f}x speedup ({serial_time:.4f}s → {method_time:.4f}s)")
                    elif method_time is None:
                        print(f"{method}: Speedup calculation unavailable (time not reported)")

def print_method_explanations():
    """Print explanations of each method"""
    print("\n" + "=" * 80)
    print("METHOD EXPLANATIONS")
    print("=" * 80)
    
    explanations = {
        'Serial': """
Serial Implementation:
  • Single-threaded execution
  • Processes nodes sequentially
  • No parallelization overhead
  • Best for: Small graphs, development, baseline comparison
  • Complexity: O(I × (N + E)) where I=iterations, N=nodes, E=edges
        """,
        'Pthreads': """
Pthreads Implementation:
  • Multi-threaded using POSIX threads
  • Shared memory parallelism
  • Divides nodes among threads
  • Uses mutex locks for thread safety
  • Best for: Multi-core machines, medium graphs (100K-10M nodes)
  • Speedup: Typically 2-8x on 4-8 core machines
        """,
        'MPI': """
MPI Implementation:
  • Distributed memory parallelism
  • Uses Message Passing Interface
  • Graph partitioned across processes
  • CSR (Compressed Sparse Row) format for efficiency
  • Best for: Large graphs (>10M nodes), clusters
  • Speedup: Scales with number of processes (4-100x+)
        """
    }
    
    for method, explanation in explanations.items():
        print(explanation)

def main():
    if len(sys.argv) < 4:
        print("Usage: python compare_methods.py <graph_file> <nodes> <threshold> [d] [threads]")
        print("\nExample:")
        if os.name == 'nt':
            print("  python compare_methods.py pagerank_mpi\\small_graph.txt 4 0.0001 0.85 4")
        else:
            print("  python compare_methods.py pagerank_mpi/small_graph.txt 4 0.0001 0.85 4")
        sys.exit(1)
    
    graph_file = sys.argv[1]
    nodes = sys.argv[2]
    threshold = sys.argv[3]
    d = sys.argv[4] if len(sys.argv) > 4 else "0.85"
    threads = sys.argv[5] if len(sys.argv) > 5 else "4"
    
    # Change to project root
    os.chdir(Path(__file__).parent)
    
    # Detect OS and normalize paths
    is_windows = os.name == 'nt'
    exe_ext = '.exe' if is_windows else ''
    path_sep = '\\' if is_windows else '/'
    
    # Normalize graph file path for Windows
    if is_windows:
        graph_file = graph_file.replace('/', '\\')
    
    print("=" * 80)
    print("PAGERANK METHOD COMPARISON TOOL")
    print("=" * 80)
    print(f"Graph: {graph_file}")
    print(f"Nodes: {nodes}")
    print(f"Threshold: {threshold}")
    print(f"Damping Factor: {d}")
    print(f"Threads/Processes: {threads}")
    print("\nRunning all implementations...")
    
    results = {}
    
    # Run Serial
    print("\n[1/3] Running Serial implementation...")
    serial_exe = f".{path_sep}pagerank_serial{path_sep}pagerank_serial{exe_ext}"
    cmd = [serial_exe, graph_file, nodes, threshold, d]
    result = run_command(cmd)
    if result['success']:
        result['metrics'] = extract_metrics(result['output'])
    results['Serial'] = result
    
    # Run Pthreads
    print("[2/3] Running Pthreads implementation...")
    pthreads_exe = f".{path_sep}pagerank_pthreads{path_sep}pagerank_pthreads{exe_ext}"
    cmd = [pthreads_exe, graph_file, nodes, threshold, d, threads]
    result = run_command(cmd)
    if result['success']:
        result['metrics'] = extract_metrics(result['output'])
    results['Pthreads'] = result
    
    # Run MPI
    print("[3/3] Running MPI implementation...")
    mpi_exe = f".{path_sep}pagerank_mpi{path_sep}pagerank_mpi{exe_ext}"
    cmd = ["mpiexec", "-n", threads, mpi_exe, graph_file, nodes, threshold, d]
    result = run_command(cmd)
    if result['success']:
        result['metrics'] = extract_metrics(result['output'])
    results['MPI'] = result
    
    # Print results
    print_comparison_table(results)
    print_detailed_comparison(results)
    print_method_explanations()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    successful = sum(1 for r in results.values() if r['success'])
    print(f"Successfully ran {successful}/3 implementations")
    
    if successful == 3:
        print("\n[OK] All methods completed successfully!")
        print("  Compare the execution times and iterations above.")
        print("  See COMPARISON.md for detailed explanations of each method.")
    else:
        print("\n[WARNING] Some implementations failed. Check errors above.")
        print("  Make sure all executables are compiled:")
        print("    .\\build_all.bat")
        print("  See BUILD_AND_TEST.md for detailed instructions.")

if __name__ == "__main__":
    main()

