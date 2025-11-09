#!/usr/bin/env python3
"""
Benchmarking tool for PageRank implementations
Compares performance across Serial, Pthreads, and MPI versions
"""

import subprocess
import time
import json
import os
import sys
import argparse
from pathlib import Path
import csv

class BenchmarkRunner:
    def __init__(self, graph_file, nodes, threshold=0.0001, d=0.85):
        self.graph_file = graph_file
        self.nodes = nodes
        self.threshold = threshold
        self.d = d
        self.results = []
        
    def run_serial(self):
        """Run serial implementation"""
        print("Running Serial implementation...")
        cmd = ["./pagerank_serial/pagerank_serial", 
               self.graph_file, str(self.nodes), str(self.threshold), str(self.d)]
        
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start
        
        if result.returncode != 0:
            print(f"Error running serial: {result.stderr}")
            return None
            
        return {
            "version": "serial",
            "threads": 1,
            "time": elapsed,
            "output": result.stdout
        }
    
    def run_pthreads(self, num_threads):
        """Run pthreads implementation"""
        print(f"Running Pthreads implementation with {num_threads} threads...")
        cmd = ["./pagerank_pthreads/pagerank_pthreads",
               self.graph_file, str(self.nodes), str(self.threshold), str(self.d), str(num_threads)]
        
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start
        
        if result.returncode != 0:
            print(f"Error running pthreads: {result.stderr}")
            return None
            
        return {
            "version": "pthreads",
            "threads": num_threads,
            "time": elapsed,
            "output": result.stdout
        }
    
    def run_mpi(self, num_processes):
        """Run MPI implementation"""
        print(f"Running MPI implementation with {num_processes} processes...")
        cmd = ["mpiexec", "-n", str(num_processes),
               "./pagerank_mpi/pagerank_mpi",
               self.graph_file, str(self.nodes), str(self.threshold), str(self.d)]
        
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start
        
        if result.returncode != 0:
            print(f"Error running MPI: {result.stderr}")
            return None
            
        return {
            "version": "mpi",
            "threads": num_processes,
            "time": elapsed,
            "output": result.stdout
        }
    
    def benchmark_all(self, thread_counts=[1, 2, 4, 8]):
        """Run all benchmarks"""
        print("=" * 60)
        print("Starting PageRank Benchmark Suite")
        print("=" * 60)
        
        # Serial baseline
        serial_result = self.run_serial()
        if serial_result:
            self.results.append(serial_result)
            baseline_time = serial_result["time"]
        else:
            print("Serial implementation failed, cannot compute speedup")
            baseline_time = None
        
        # Pthreads
        for threads in thread_counts:
            result = self.run_pthreads(threads)
            if result:
                if baseline_time:
                    result["speedup"] = baseline_time / result["time"]
                    result["efficiency"] = result["speedup"] / threads
                self.results.append(result)
        
        # MPI (if available)
        try:
            for processes in thread_counts:
                result = self.run_mpi(processes)
                if result:
                    if baseline_time:
                        result["speedup"] = baseline_time / result["time"]
                        result["efficiency"] = result["speedup"] / processes
                    self.results.append(result)
        except FileNotFoundError:
            print("MPI not available, skipping MPI benchmarks")
        
        return self.results
    
    def save_results(self, filename="benchmark_results.json"):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filename}")
    
    def save_csv(self, filename="benchmark_results.csv"):
        """Save results to CSV file"""
        if not self.results:
            return
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["version", "threads", "time", "speedup", "efficiency"])
            writer.writeheader()
            for result in self.results:
                row = {
                    "version": result["version"],
                    "threads": result["threads"],
                    "time": f"{result['time']:.4f}",
                    "speedup": f"{result.get('speedup', 0):.4f}" if 'speedup' in result else "N/A",
                    "efficiency": f"{result.get('efficiency', 0):.4f}" if 'efficiency' in result else "N/A"
                }
                writer.writerow(row)
        print(f"CSV results saved to {filename}")
    
    def print_summary(self):
        """Print summary of results"""
        print("\n" + "=" * 60)
        print("Benchmark Summary")
        print("=" * 60)
        print(f"{'Version':<12} {'Threads':<10} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<10}")
        print("-" * 60)
        
        for result in self.results:
            speedup = result.get('speedup', 0)
            efficiency = result.get('efficiency', 0)
            print(f"{result['version']:<12} {result['threads']:<10} {result['time']:<12.4f} "
                  f"{speedup:<10.4f} {efficiency:<10.4f}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark PageRank implementations')
    parser.add_argument('graph_file', help='Path to graph file')
    parser.add_argument('nodes', type=int, help='Number of nodes')
    parser.add_argument('--threshold', type=float, default=0.0001, help='Convergence threshold')
    parser.add_argument('--d', type=float, default=0.85, help='Damping factor')
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 2, 4, 8],
                       help='Number of threads/processes to test')
    parser.add_argument('--output', default='benchmark_results', help='Output file prefix')
    
    args = parser.parse_args()
    
    # Change to project root
    os.chdir(Path(__file__).parent.parent)
    
    runner = BenchmarkRunner(args.graph_file, args.nodes, args.threshold, args.d)
    runner.benchmark_all(args.threads)
    runner.print_summary()
    runner.save_results(f"{args.output}.json")
    runner.save_csv(f"{args.output}.csv")


if __name__ == "__main__":
    main()

