#!/usr/bin/env python3
"""
Scalability Studies for PageRank implementations
Tests strong scaling and weak scaling
"""

import subprocess
import time
import json
import os
import sys
import argparse
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

class ScalabilityStudy:
    def __init__(self, graph_file, base_nodes, threshold=0.0001, d=0.85):
        self.graph_file = graph_file
        self.base_nodes = base_nodes
        self.threshold = threshold
        self.d = d
        self.strong_scaling_results = []
        self.weak_scaling_results = []
    
    def run_mpi(self, num_processes, nodes=None):
        """Run MPI implementation"""
        if nodes is None:
            nodes = self.base_nodes
            
        cmd = ["mpiexec", "-n", str(num_processes),
               "./pagerank_mpi/pagerank_mpi",
               self.graph_file, str(nodes), str(self.threshold), str(self.d)]
        
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start
        
        if result.returncode != 0:
            print(f"Error running MPI: {result.stderr}")
            return None
            
        return elapsed
    
    def strong_scaling(self, process_counts=[1, 2, 4, 8, 16]):
        """Strong scaling: fixed problem size, increasing processors"""
        print("=" * 60)
        print("Strong Scaling Study")
        print("=" * 60)
        print(f"Fixed problem size: {self.base_nodes} nodes")
        print(f"Testing with {process_counts} processes\n")
        
        baseline_time = None
        for processes in process_counts:
            print(f"Running with {processes} processes...")
            elapsed = self.run_mpi(processes)
            if elapsed:
                if baseline_time is None:
                    baseline_time = elapsed
                speedup = baseline_time / elapsed
                efficiency = speedup / processes
                
                result = {
                    "processes": processes,
                    "time": elapsed,
                    "speedup": speedup,
                    "efficiency": efficiency
                }
                self.strong_scaling_results.append(result)
                print(f"  Time: {elapsed:.4f}s, Speedup: {speedup:.4f}, Efficiency: {efficiency:.4f}")
        
        return self.strong_scaling_results
    
    def weak_scaling(self, process_counts=[1, 2, 4, 8, 16]):
        """Weak scaling: problem size increases proportionally with processors"""
        print("\n" + "=" * 60)
        print("Weak Scaling Study")
        print("=" * 60)
        print(f"Base problem size: {self.base_nodes} nodes per process")
        print(f"Testing with {process_counts} processes\n")
        
        baseline_time = None
        for processes in process_counts:
            nodes = self.base_nodes * processes
            print(f"Running with {processes} processes, {nodes} nodes...")
            elapsed = self.run_mpi(processes, nodes)
            if elapsed:
                if baseline_time is None:
                    baseline_time = elapsed
                scaled_efficiency = baseline_time / elapsed
                
                result = {
                    "processes": processes,
                    "nodes": nodes,
                    "time": elapsed,
                    "scaled_efficiency": scaled_efficiency
                }
                self.weak_scaling_results.append(result)
                print(f"  Time: {elapsed:.4f}s, Scaled Efficiency: {scaled_efficiency:.4f}")
        
        return self.weak_scaling_results
    
    def plot_results(self, output_dir="plots"):
        """Plot scalability results"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.strong_scaling_results:
            processes = [r["processes"] for r in self.strong_scaling_results]
            speedups = [r["speedup"] for r in self.strong_scaling_results]
            efficiencies = [r["efficiency"] for r in self.strong_scaling_results]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Speedup plot
            ax1.plot(processes, speedups, 'o-', label='Actual Speedup')
            ax1.plot(processes, processes, '--', label='Ideal Speedup')
            ax1.set_xlabel('Number of Processes')
            ax1.set_ylabel('Speedup')
            ax1.set_title('Strong Scaling - Speedup')
            ax1.legend()
            ax1.grid(True)
            
            # Efficiency plot
            ax2.plot(processes, efficiencies, 'o-', color='orange')
            ax2.axhline(y=1.0, color='r', linestyle='--', label='Ideal Efficiency')
            ax2.set_xlabel('Number of Processes')
            ax2.set_ylabel('Efficiency')
            ax2.set_title('Strong Scaling - Efficiency')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/strong_scaling.png")
            print(f"\nStrong scaling plot saved to {output_dir}/strong_scaling.png")
        
        if self.weak_scaling_results:
            processes = [r["processes"] for r in self.weak_scaling_results]
            times = [r["time"] for r in self.weak_scaling_results]
            efficiencies = [r["scaled_efficiency"] for r in self.weak_scaling_results]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Time plot
            ax1.plot(processes, times, 'o-', label='Actual Time')
            if len(times) > 0:
                ax1.axhline(y=times[0], color='r', linestyle='--', label='Ideal Time (constant)')
            ax1.set_xlabel('Number of Processes')
            ax1.set_ylabel('Execution Time (s)')
            ax1.set_title('Weak Scaling - Execution Time')
            ax1.legend()
            ax1.grid(True)
            
            # Efficiency plot
            ax2.plot(processes, efficiencies, 'o-', color='orange')
            ax2.axhline(y=1.0, color='r', linestyle='--', label='Ideal Efficiency')
            ax2.set_xlabel('Number of Processes')
            ax2.set_ylabel('Scaled Efficiency')
            ax2.set_title('Weak Scaling - Efficiency')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/weak_scaling.png")
            print(f"Weak scaling plot saved to {output_dir}/weak_scaling.png")
    
    def save_results(self, filename="scalability_results.json"):
        """Save results to JSON"""
        results = {
            "strong_scaling": self.strong_scaling_results,
            "weak_scaling": self.weak_scaling_results
        }
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Scalability study for PageRank')
    parser.add_argument('graph_file', help='Path to graph file')
    parser.add_argument('base_nodes', type=int, help='Base number of nodes')
    parser.add_argument('--threshold', type=float, default=0.0001, help='Convergence threshold')
    parser.add_argument('--d', type=float, default=0.85, help='Damping factor')
    parser.add_argument('--processes', nargs='+', type=int, default=[1, 2, 4, 8],
                       help='Number of processes to test')
    parser.add_argument('--strong-only', action='store_true', help='Run only strong scaling')
    parser.add_argument('--weak-only', action='store_true', help='Run only weak scaling')
    
    args = parser.parse_args()
    
    # Change to project root
    os.chdir(Path(__file__).parent.parent)
    
    study = ScalabilityStudy(args.graph_file, args.base_nodes, args.threshold, args.d)
    
    if not args.weak_only:
        study.strong_scaling(args.processes)
    
    if not args.strong_only:
        study.weak_scaling(args.processes)
    
    study.plot_results()
    study.save_results()


if __name__ == "__main__":
    main()

