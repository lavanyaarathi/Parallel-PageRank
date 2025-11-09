#!/usr/bin/env python3
"""
Graph Visualization Tools for PageRank
Visualizes convergence, top-ranked nodes, and PageRank distribution
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import json
import argparse
import sys
import os
from pathlib import Path
import re

class PageRankVisualizer:
    def __init__(self):
        self.fig = None
        self.ax = None
    
    def parse_pagerank_output(self, output_file):
        """Parse PageRank output to extract convergence data"""
        iterations = []
        max_errors = []
        l1_norms = []
        final_ranks = {}
        
        with open(output_file, 'r') as f:
            content = f.read()
            
            # Extract iteration data
            pattern = r'Iteration (\d+).*?Max Error = ([\d.]+).*?L1 Norm = ([\d.]+)'
            matches = re.findall(pattern, content)
            for match in matches:
                iterations.append(int(match[0]))
                max_errors.append(float(match[1]))
                l1_norms.append(float(match[2]))
        
        return {
            'iterations': iterations,
            'max_errors': max_errors,
            'l1_norms': l1_norms
        }
    
    def plot_convergence(self, data, output_file='convergence.png'):
        """Plot convergence over iterations"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        iterations = data['iterations']
        
        # Max error plot
        ax1.semilogy(iterations, data['max_errors'], 'o-', label='Max Error')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Max Error (log scale)')
        ax1.set_title('PageRank Convergence - Max Error')
        ax1.legend()
        ax1.grid(True)
        
        # L1 norm plot
        ax2.semilogy(iterations, data['l1_norms'], 'o-', color='orange', label='L1 Norm')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('L1 Norm (log scale)')
        ax2.set_title('PageRank Convergence - L1 Norm')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Convergence plot saved to {output_file}")
        plt.close()
    
    def visualize_graph(self, graph_file, pagerank_file=None, top_n=10, output_file='graph_visualization.png'):
        """Visualize graph with PageRank values"""
        G = nx.DiGraph()
        
        # Read graph
        with open(graph_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        from_node = int(parts[0])
                        to_node = int(parts[1])
                        G.add_edge(from_node, to_node)
                    except ValueError:
                        continue
        
        # Read PageRank values if provided
        node_colors = None
        node_sizes = None
        if pagerank_file:
            pagerank_values = {}
            with open(pagerank_file, 'r') as f:
                for line in f:
                    # Parse PageRank output format
                    match = re.search(r'P_t1\[(\d+)\]\s*=\s*([\d.]+)', line)
                    if match:
                        node = int(match.group(1))
                        rank = float(match.group(2))
                        pagerank_values[node] = rank
            
            if pagerank_values:
                node_colors = [pagerank_values.get(node, 0) for node in G.nodes()]
                node_sizes = [pagerank_values.get(node, 0) * 10000 for node in G.nodes()]
        
        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw
        plt.figure(figsize=(12, 8))
        
        if node_colors:
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                 node_size=node_sizes if node_sizes else 300,
                                 cmap=plt.cm.Reds, alpha=0.8)
        else:
            nx.draw_networkx_nodes(G, pos, node_size=300, alpha=0.8)
        
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=10)
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title('Graph Visualization with PageRank')
        if node_colors:
            plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), label='PageRank')
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Graph visualization saved to {output_file}")
        plt.close()
    
    def plot_top_ranked(self, pagerank_file, top_n=20, output_file='top_ranked.png'):
        """Plot top-ranked nodes"""
        pagerank_values = {}
        
        with open(pagerank_file, 'r') as f:
            for line in f:
                match = re.search(r'P_t1\[(\d+)\]\s*=\s*([\d.]+)', line)
                if match:
                    node = int(match.group(1))
                    rank = float(match.group(2))
                    pagerank_values[node] = rank
        
        if not pagerank_values:
            print("No PageRank values found in file")
            return
        
        # Sort by rank
        sorted_nodes = sorted(pagerank_values.items(), key=lambda x: x[1], reverse=True)
        top_nodes = sorted_nodes[:top_n]
        
        nodes = [node for node, _ in top_nodes]
        ranks = [rank for _, rank in top_nodes]
        
        plt.figure(figsize=(12, 6))
        plt.barh(range(len(nodes)), ranks)
        plt.yticks(range(len(nodes)), [f'Node {n}' for n in nodes])
        plt.xlabel('PageRank Value')
        plt.title(f'Top {top_n} Ranked Nodes')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Top ranked nodes plot saved to {output_file}")
        plt.close()
    
    def plot_rank_distribution(self, pagerank_file, output_file='rank_distribution.png'):
        """Plot distribution of PageRank values"""
        pagerank_values = []
        
        with open(pagerank_file, 'r') as f:
            for line in f:
                match = re.search(r'P_t1\[(\d+)\]\s*=\s*([\d.]+)', line)
                if match:
                    rank = float(match.group(2))
                    pagerank_values.append(rank)
        
        if not pagerank_values:
            print("No PageRank values found in file")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(pagerank_values, bins=50, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('PageRank Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('PageRank Value Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Log scale histogram
        ax2.hist(pagerank_values, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_yscale('log')
        ax2.set_xlabel('PageRank Value')
        ax2.set_ylabel('Frequency (log scale)')
        ax2.set_title('PageRank Value Distribution (Log Scale)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Rank distribution plot saved to {output_file}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize PageRank results')
    parser.add_argument('--convergence', help='Parse convergence from output file')
    parser.add_argument('--graph', help='Graph file to visualize')
    parser.add_argument('--pagerank', help='PageRank output file')
    parser.add_argument('--top-n', type=int, default=20, help='Number of top nodes to show')
    parser.add_argument('--output-dir', default='plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    viz = PageRankVisualizer()
    
    if args.convergence:
        data = viz.parse_pagerank_output(args.convergence)
        viz.plot_convergence(data, f"{args.output_dir}/convergence.png")
    
    if args.graph:
        viz.visualize_graph(args.graph, args.pagerank, args.top_n, 
                          f"{args.output_dir}/graph_visualization.png")
    
    if args.pagerank:
        viz.plot_top_ranked(args.pagerank, args.top_n, f"{args.output_dir}/top_ranked.png")
        viz.plot_rank_distribution(args.pagerank, f"{args.output_dir}/rank_distribution.png")


if __name__ == "__main__":
    main()

