#!/usr/bin/env python3
"""
Interactive Dashboard for PageRank
Built with Streamlit for easy web interface with comparison visualizations
"""

import streamlit as st
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import tempfile
import os
import re
from pathlib import Path
import json
import platform

st.set_page_config(page_title="PageRank Dashboard", layout="wide")

# Detect OS
is_windows = platform.system() == 'Windows'
exe_ext = '.exe' if is_windows else ''
path_sep = '\\' if is_windows else '/'

# PageRank Educational Section
with st.expander("📚 What is PageRank? Learn the Algorithm", expanded=False):
    st.header("Understanding PageRank")
    
    st.markdown("""
    ### What is PageRank?
    
    **PageRank** is an algorithm developed by Google founders Larry Page and Sergey Brin to rank web pages 
    in search results. It measures the importance of nodes in a directed graph by analyzing the structure 
    of links between them.
    
    ### Key Concepts:
    
    1. **Graph Representation**: The web is modeled as a directed graph where:
       - **Nodes** = Web pages
       - **Edges** = Links between pages (pointing from source to destination)
    
    2. **Importance Principle**: A page is important if:
       - Many important pages link to it
       - Pages that link to it have few outgoing links
    
    3. **Random Surfer Model**: PageRank simulates a random surfer who:
       - Follows links with probability **d** (damping factor, typically 0.85)
       - Jumps to a random page with probability **(1-d)**
    
    ### The PageRank Formula:
    
    For each node **i**, the PageRank value is calculated as:
    
    ```
    PR(i) = (1-d)/N + d × Σ(PR(j)/L(j))
    ```
    
    Where:
    - **PR(i)** = PageRank of node i
    - **d** = Damping factor (usually 0.85)
    - **N** = Total number of nodes
    - **PR(j)** = PageRank of node j (that links to i)
    - **L(j)** = Number of outgoing links from node j
    - **Σ** = Sum over all nodes j that link to i
    
    ### How to Calculate PageRank:
    
    **Step 1: Initialize**
    - Start with equal PageRank for all nodes: PR(i) = 1/N
    
    **Step 2: Iterate**
    - For each iteration:
      - Calculate new PageRank for each node using the formula above
      - Update all values simultaneously
    
    **Step 3: Check Convergence**
    - Calculate the difference between old and new PageRank values
    - If the difference (L1 norm) is below a threshold (ε), stop
    - Otherwise, repeat Step 2
    
    **Step 4: Final Ranking**
    - Nodes with higher PageRank values are more important
    
    ### Example:
    
    Consider a simple graph with 3 nodes:
    - Node A links to Node B
    - Node B links to Node C
    - Node C links to Node A
    
    Initially: PR(A) = PR(B) = PR(C) = 1/3 = 0.333
    
    After iteration 1 (with d=0.85):
    - PR(A) = 0.15/3 + 0.85 × (PR(C)/1) = 0.05 + 0.85 × 0.333 = 0.333
    - PR(B) = 0.15/3 + 0.85 × (PR(A)/1) = 0.05 + 0.85 × 0.333 = 0.333
    - PR(C) = 0.15/3 + 0.85 × (PR(B)/1) = 0.05 + 0.85 × 0.333 = 0.333
    
    This continues until convergence!
    
    ### Parameters:
    
    - **Damping Factor (d)**: Probability of following links (typically 0.85)
      - Higher d = More weight on link structure
      - Lower d = More random jumps
    
    - **Convergence Threshold (ε)**: Maximum allowed difference between iterations
      - Smaller threshold = More accurate but slower
      - Typical values: 0.0001 to 0.000001
    
    ### Why Parallelize?
    
    For large graphs (millions of nodes), computing PageRank sequentially is slow. 
    Parallel implementations distribute the work:
    
    - **Pthreads**: Uses multiple CPU cores on a single machine
    - **MPI**: Distributes computation across multiple machines/nodes
    
    This can provide **2-100x speedup** depending on graph size and hardware!
    """)
    
    st.markdown("---")
    st.markdown("**💡 Tip**: Try running the algorithm on different graphs to see how PageRank values change!")

# ============================================================================
# Function Definitions (must be defined before use in Streamlit)
# ============================================================================

def extract_metrics(output):
    """Extract key metrics from PageRank output"""
    metrics = {
        'iterations': None,
        'time': None,
        'converged': False,
        'final_l1': None,
        'final_max_error': None,
        'convergence_data': []
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
    
    # Extract convergence data - try multiple patterns
    # Pattern 1: "Iteration X, Max Error = Y, L1 Norm = Z" (MPI format)
    pattern1 = r'Iteration (\d+)[^0-9]*Max Error[^0-9]*([\d.eE+-]+)[^0-9]*L1 Norm[^0-9]*([\d.eE+-]+)'
    matches1 = re.findall(pattern1, output)
    if matches1:
        for match in matches1:
            try:
                metrics['convergence_data'].append({
                    'iteration': int(match[0]),
                    'max_error': float(match[1]),
                    'l1_norm': float(match[2])
                })
            except (ValueError, IndexError):
                continue
    
    # Pattern 2: "Iteration X...Max Error = Y...L1 Norm = Z" (Serial/Pthreads format)
    if not metrics['convergence_data']:
        pattern2 = r'Iteration (\d+).*?Max Error[:\s=]+([\d.eE+-]+).*?L1 Norm[:\s=]+([\d.eE+-]+)'
        matches2 = re.findall(pattern2, output, re.MULTILINE | re.DOTALL)
        if matches2:
            for match in matches2:
                try:
                    metrics['convergence_data'].append({
                        'iteration': int(match[0]),
                        'max_error': float(match[1]),
                        'l1_norm': float(match[2])
                    })
                except (ValueError, IndexError):
                    continue
    
    # Extract final values from convergence data
    if metrics['convergence_data']:
        last = metrics['convergence_data'][-1]
        metrics['final_max_error'] = last['max_error']
        metrics['final_l1'] = last['l1_norm']
    else:
        # Try to extract just L1 norm and max error without iteration data
        l1_match = re.search(r'L1[^0-9]*Norm[^0-9]*([\d.eE+-]+)', output)
        if l1_match:
            try:
                metrics['final_l1'] = float(l1_match.group(1))
            except (ValueError, IndexError):
                pass
        
        max_err_match = re.search(r'Max Error[^0-9]*([\d.eE+-]+)', output)
        if max_err_match:
            try:
                metrics['final_max_error'] = float(max_err_match.group(1))
            except (ValueError, IndexError):
                pass
    
    # Extract PageRank values
    pagerank_pattern = r'P_t1\[(\d+)\]\s*=\s*([\d.e-]+)'
    pagerank_matches = re.findall(pagerank_pattern, output)
    if pagerank_matches:
        metrics['pagerank_values'] = {int(node): float(rank) for node, rank in pagerank_matches}
    
    return metrics

def get_project_root():
    """Get project root directory - works in both normal Python and Streamlit"""
    # Try using __file__ first (works in normal Python)
    try:
        if __file__:
            root = Path(__file__).parent.parent
            if (root / "pagerank_serial").exists():
                return root
    except:
        pass
    
    # Fallback: try current working directory
    cwd = Path.cwd()
    if (cwd / "pagerank_serial").exists():
        return cwd
    
    # Fallback: try parent of current directory
    parent = cwd.parent
    if (parent / "pagerank_serial").exists():
        return parent
    
    # Last resort: return current directory
    return cwd

def run_all_implementations(graph_file, nodes, threshold, d, num_threads):
    """Run all implementations and return results"""
    results = {}
    project_root = get_project_root()
    
    # Run Serial
    exe_path = project_root / "pagerank_serial" / f"pagerank_serial{exe_ext}"
    if not exe_path.exists():
        results['Serial'] = {'success': False, 'error': f'Executable not found: {exe_path}'}
    else:
        # Ensure graph_file is an absolute path
        if graph_file and not os.path.isabs(graph_file):
            graph_file = os.path.abspath(graph_file)
        
        cmd = [str(exe_path), str(graph_file), str(nodes), str(threshold), str(d)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=project_root)
            if result.returncode == 0:
                metrics = extract_metrics(result.stdout)
                results['Serial'] = {'success': True, 'output': result.stdout, 'metrics': metrics}
            else:
                # Include both stderr and stdout in error message for debugging
                error_msg = result.stderr if result.stderr else result.stdout
                if not error_msg:
                    error_msg = f"Process exited with code {result.returncode}"
                # Add more context to error message
                error_msg = f"Command: {' '.join(cmd)}\nError: {error_msg}"
                results['Serial'] = {'success': False, 'error': error_msg}
        except subprocess.TimeoutExpired:
            results['Serial'] = {'success': False, 'error': 'Execution timed out after 300 seconds'}
        except Exception as e:
            results['Serial'] = {'success': False, 'error': f'{type(e).__name__}: {str(e)}'}
    
    # Run Pthreads
    exe_path = project_root / "pagerank_pthreads" / f"pagerank_pthreads{exe_ext}"
    if not exe_path.exists():
        results['Pthreads'] = {'success': False, 'error': f'Executable not found: {exe_path}'}
    else:
        # Ensure graph_file is an absolute path
        if graph_file and not os.path.isabs(graph_file):
            graph_file = os.path.abspath(graph_file)
        
        cmd = [str(exe_path), str(graph_file), str(nodes), str(threshold), str(d), str(num_threads)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=project_root)
            if result.returncode == 0:
                metrics = extract_metrics(result.stdout)
                results['Pthreads'] = {'success': True, 'output': result.stdout, 'metrics': metrics}
            else:
                # Include both stderr and stdout in error message for debugging
                error_msg = result.stderr if result.stderr else result.stdout
                if not error_msg:
                    error_msg = f"Process exited with code {result.returncode}"
                results['Pthreads'] = {'success': False, 'error': error_msg}
        except subprocess.TimeoutExpired:
            results['Pthreads'] = {'success': False, 'error': 'Execution timed out after 300 seconds'}
        except Exception as e:
            results['Pthreads'] = {'success': False, 'error': f'{type(e).__name__}: {str(e)}'}
    
    # Run MPI
    exe_path = project_root / "pagerank_mpi" / f"pagerank_mpi{exe_ext}"
    if not exe_path.exists():
        results['MPI'] = {'success': False, 'error': f'Executable not found: {exe_path}'}
    else:
        # Ensure graph_file is an absolute path
        if graph_file and not os.path.isabs(graph_file):
            graph_file = os.path.abspath(graph_file)
        
        cmd = ["mpiexec", "-n", str(num_threads), str(exe_path), str(graph_file), str(nodes), str(threshold), str(d)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=project_root)
            if result.returncode == 0:
                metrics = extract_metrics(result.stdout)
                results['MPI'] = {'success': True, 'output': result.stdout, 'metrics': metrics}
            else:
                results['MPI'] = {'success': False, 'error': result.stderr}
        except Exception as e:
            results['MPI'] = {'success': False, 'error': str(e)}
    
    return results

def display_single_results(output, implementation, num_threads):
    """Display results for a single implementation"""
    st.header("📈 Results")
    
    metrics = extract_metrics(output)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if metrics.get('iterations'):
            st.metric("Iterations", metrics['iterations'])
    with col2:
        if metrics.get('time'):
            st.metric("Execution Time", f"{metrics['time']:.4f}s")
    with col3:
        st.metric("Implementation", implementation)
    with col4:
        if num_threads > 1:
            st.metric("Threads/Processes", num_threads)
    
    # Convergence plots
    if metrics.get('convergence_data'):
        st.subheader("📉 Convergence Analysis")
        conv_data = metrics['convergence_data']
        iterations = [d['iteration'] for d in conv_data]
        max_errors = [d['max_error'] for d in conv_data]
        l1_norms = [d['l1_norm'] for d in conv_data]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.semilogy(iterations, max_errors, 'o-', label='Max Error', linewidth=2, markersize=6)
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Max Error (log scale)', fontsize=12)
            ax.set_title('Convergence - Max Error', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.semilogy(iterations, l1_norms, 'o-', color='orange', label='L1 Norm', linewidth=2, markersize=6)
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('L1 Norm (log scale)', fontsize=12)
            ax.set_title('Convergence - L1 Norm', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
    
    # PageRank values
    if metrics.get('pagerank_values'):
        st.subheader("🏆 Top Ranked Nodes")
        pagerank_data = metrics['pagerank_values']
        sorted_nodes = sorted(pagerank_data.items(), key=lambda x: x[1], reverse=True)
        top_n = st.slider("Number of top nodes to display", min_value=5, max_value=min(50, len(sorted_nodes)), value=10, key="top_n_single")
        
        top_nodes = sorted_nodes[:top_n]
        df = pd.DataFrame(top_nodes, columns=['Node', 'PageRank'])
        st.dataframe(df, width='stretch')
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            nodes_list = [f'Node {n}' for n, _ in top_nodes]
            ranks = [r for _, r in top_nodes]
            ax.barh(range(len(nodes_list)), ranks, color='steelblue')
            ax.set_yticks(range(len(nodes_list)))
            ax.set_yticklabels(nodes_list)
            ax.set_xlabel('PageRank Value', fontsize=12)
            ax.set_title(f'Top {top_n} Ranked Nodes', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            all_ranks = list(pagerank_data.values())
            ax.hist(all_ranks, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_xlabel('PageRank Value', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('PageRank Value Distribution', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
    
    # Full output
    with st.expander("View Full Output"):
        st.code(output)

def analyze_graph_characteristics(graph_file):
    """Analyze graph characteristics for method recommendations"""
    try:
        G = nx.DiGraph()
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
        
        if len(G.nodes()) == 0:
            return None
        
        num_nodes = len(G.nodes())
        num_edges = len(G.edges())
        density = nx.density(G)
        
        # Calculate graph metrics
        in_degrees = [G.in_degree(n) for n in G.nodes()]
        out_degrees = [G.out_degree(n) for n in G.nodes()]
        avg_in_degree = sum(in_degrees) / num_nodes if num_nodes > 0 else 0
        avg_out_degree = sum(out_degrees) / num_nodes if num_nodes > 0 else 0
        
        # Check for isolated nodes
        isolated = list(nx.isolates(G))
        num_isolated = len(isolated)
        
        # Check graph type
        if density > 0.8:
            graph_type = "Dense"
        elif density > 0.3:
            graph_type = "Moderate"
        else:
            graph_type = "Sparse"
        
        # Determine if graph is scale-free (power-law distribution)
        degree_dist = dict(G.degree())
        degrees = list(degree_dist.values())
        if len(degrees) > 1:
            max_degree = max(degrees)
            high_degree_nodes = sum(1 for d in degrees if d > max_degree * 0.5)
            is_scale_free = high_degree_nodes < len(degrees) * 0.2 and max_degree > avg_out_degree * 3
        else:
            is_scale_free = False
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'graph_type': graph_type,
            'avg_in_degree': avg_in_degree,
            'avg_out_degree': avg_out_degree,
            'num_isolated': num_isolated,
            'is_scale_free': is_scale_free,
            'edges_per_node': num_edges / num_nodes if num_nodes > 0 else 0
        }
    except Exception as e:
        return None

def display_comparison_results(results, num_nodes=None, graph_file=None):
    """Display comparison results with visualizations"""
    st.header("🔬 Method Comparison Dashboard")
    st.markdown("---")
    
    # Analyze graph characteristics FIRST (before checking results)
    # This allows us to show graph analysis even if comparison hasn't been run
    graph_analysis = None
    if graph_file and os.path.exists(graph_file):
        graph_analysis = analyze_graph_characteristics(graph_file)
        if graph_analysis:
            num_nodes = graph_analysis['num_nodes']
    
    # Get number of nodes from results or use provided value
    if num_nodes is None:
        # Try to extract from graph file or use default
        num_nodes = 4
    
    # Check if results exist and filter successful ones (safely handle None/empty)
    results_exist = results is not None and isinstance(results, dict) and len(results) > 0
    successful = {}
    if results_exist:
        successful = {k: v for k, v in results.items() if v.get('success')}
    
    # Graph-Specific Analysis Section
    if graph_analysis:
        st.subheader("📊 Graph Analysis & Method Recommendations")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes", graph_analysis['num_nodes'])
        with col2:
            st.metric("Edges", graph_analysis['num_edges'])
        with col3:
            st.metric("Density", f"{graph_analysis['density']:.4f}")
        with col4:
            st.metric("Graph Type", graph_analysis['graph_type'])
        
        # Graph-specific recommendations
        st.markdown("#### 🎯 Graph-Specific Recommendations")
        
        recommendations_text = []
        
        # Size-based recommendations
        if graph_analysis['num_nodes'] < 100:
            recommendations_text.append("**Small Graph**: Serial implementation is optimal for graphs this size. Parallel overhead would outweigh benefits.")
            best_method = "Serial"
        elif graph_analysis['num_nodes'] < 10000:
            recommendations_text.append("**Medium Graph**: Pthreads typically provides the best performance for graphs of this size on multi-core systems.")
            best_method = "Pthreads"
        else:
            recommendations_text.append("**Large Graph**: MPI is recommended for graphs this large, especially if running on clusters.")
            best_method = "MPI"
        
        # Density-based recommendations
        if graph_analysis['density'] > 0.5:
            recommendations_text.append("**Dense Graph**: High edge density means more computation per node. Parallel methods (Pthreads/MPI) should show significant speedup.")
        elif graph_analysis['density'] < 0.1:
            recommendations_text.append("**Sparse Graph**: Low edge density means less computation. Serial may be sufficient unless graph is very large.")
        
        # Structure-based recommendations
        if graph_analysis['is_scale_free']:
            recommendations_text.append("**Scale-Free Structure**: Uneven degree distribution. Parallel methods may show load imbalance - monitor thread/process utilization.")
        
        if graph_analysis['num_isolated'] > 0:
            recommendations_text.append(f"**Isolated Nodes**: {graph_analysis['num_isolated']} isolated nodes detected. These don't affect PageRank computation but add overhead.")
        
        # Show recommendations
        for rec in recommendations_text:
            st.info(rec)
        
        # Performance prediction based on graph characteristics
        st.markdown("#### 📈 Expected Performance Characteristics")
        st.markdown("Based on your graph structure, here's what to expect from each method:")
        
        perf_pred = []
        # Show all methods, regardless of whether they've been run
        all_methods = ['Serial', 'Pthreads', 'MPI']
        
        # Get project root for executable checking
        project_root = get_project_root()
        
        for method in all_methods:
            # Get actual metrics if available
            time_val = None
            if results_exist and method in successful:
                metrics = successful[method].get('metrics', {})
                time_val = metrics.get('time')
            
            # Predict based on graph characteristics
            if method == 'Serial':
                complexity = f"O({graph_analysis['num_edges']} × I)"  # I = iterations
                expected = "Baseline performance - no parallelization overhead"
            elif method == 'Pthreads':
                complexity = f"O({graph_analysis['num_edges']} × I / P)"  # P = threads
                if graph_analysis['num_nodes'] > 1000 and graph_analysis['density'] > 0.2:
                    expected = "Good speedup expected (2-4x on 4 cores)"
                elif graph_analysis['num_nodes'] > 100:
                    expected = "Moderate speedup expected (1.5-2x on 4 cores)"
                else:
                    expected = "Limited speedup (small graph - overhead dominates)"
            else:  # MPI
                complexity = f"O({graph_analysis['num_edges']} × I / P + comm)"
                if graph_analysis['num_nodes'] > 10000:
                    expected = "Good speedup expected (scales with processes)"
                elif graph_analysis['num_nodes'] > 1000:
                    expected = "Moderate speedup (network overhead)"
                else:
                    expected = "Overhead may dominate (small graph)"
            
            # Check if executable exists (for status display)
            exe_path_check = project_root / f"pagerank_{method.lower()}" / f"pagerank_{method.lower()}{exe_ext}"
            exe_exists = exe_path_check.exists()
            
            # Determine status
            if not results_exist:
                # Comparison hasn't been run yet - check if executables exist
                if exe_exists:
                    status = "⏸️ Not run yet"
                    time_str = "N/A (click Compare button)"
                else:
                    status = "❌ Executable missing"
                    time_str = "N/A"
            elif method in successful:
                status = "✅ Ran successfully"
                time_str = f"{time_val:.6f}" if time_val is not None else "N/A"
            elif results_exist and method in results:
                # Method was attempted but failed
                error = results[method].get('error', '')
                error_str = str(error).lower()
                if 'not found' in error_str or 'executable not found' in error_str or not exe_exists:
                    status = "❌ Executable missing"
                else:
                    status = "❌ Execution failed"
                time_str = "N/A"
            else:
                # Check executable existence for not-run methods
                if exe_exists:
                    status = "⏸️ Not run"
                else:
                    status = "❌ Executable missing"
                time_str = "N/A"
            
            perf_pred.append({
                'Method': method,
                'Status': status,
                'Actual Time (s)': time_str,
                'Complexity': complexity,
                'Expected Performance': expected
            })
        
        df_pred = pd.DataFrame(perf_pred)
        if len(perf_pred) > 0:
            st.dataframe(df_pred, width='stretch', hide_index=True)
        else:
            st.warning("⚠️ Unable to generate performance predictions.")
        
        if not results_exist:
            st.info("💡 **Run the comparison above to see actual performance results!**")
        
        st.markdown("---")
    
    # Check for executable files and provide detailed error messages
    project_root = get_project_root()
    missing_executables = []
    exe_paths = {
        'Serial': project_root / "pagerank_serial" / f"pagerank_serial{exe_ext}",
        'Pthreads': project_root / "pagerank_pthreads" / f"pagerank_pthreads{exe_ext}",
        'MPI': project_root / "pagerank_mpi" / f"pagerank_mpi{exe_ext}"
    }
    
    for method, exe_path in exe_paths.items():
        if not exe_path.exists():
            missing_executables.append(f"{method}: {exe_path}")
    
    # Check if results exist (safely handle None)
    results_exist = results is not None and isinstance(results, dict) and len(results) > 0
    if not results_exist:
        # If no results exist yet, just show graph analysis and return
        # (Graph analysis with expected performance is already shown above)
        st.markdown("---")
        st.info("💡 **Click the '🔬 Compare All Methods' button in the sidebar to run all implementations and see actual performance comparisons.**")
        if graph_analysis:
            st.info(f"📊 Your graph has {graph_analysis['num_nodes']} nodes and {graph_analysis['num_edges']} edges. See expected performance characteristics above.")
        return
    
    # Check if all implementations failed
    if not successful:
        # Show detailed error information
        st.error("❌ All implementations failed to execute successfully.")
        
        # Check for missing executables
        if missing_executables:
            st.error("**Missing Executables:**")
            for missing in missing_executables:
                st.error(f"  - {missing}")
            st.info("💡 **Tip:** Run `.\build_all.bat` in the project root to compile all executables.")
            st.info("💡 **Note:** After building, refresh this page and try again.")
        else:
            # Executables exist, but they failed to run
            st.error("**Execution Errors:**")
            for method, result in results.items():
                if not result.get('success'):
                    error = result.get('error', 'Unknown error')
                    st.error(f"  - **{method}**: {error}")
            
            st.info("💡 **Debugging Tips:**")
            st.markdown("""
            1. Check if the graph file path is correct
            2. Verify the number of nodes matches the graph file
            3. Try running from command line to see detailed errors:
            """)
            st.code(f"python compare_methods.py \"{graph_file}\" {num_nodes} {threshold if 'threshold' in locals() else 0.0001} {d if 'd' in locals() else 0.85} 4")
        
        # Don't return early - we already showed graph analysis above
        # Just skip the performance comparison sections
        if graph_analysis:
            return
        else:
            return
    
    # Show success status
    st.success(f"✅ Successfully ran {len(successful)}/{len(results)} implementations")
    
    # Summary metrics table with better formatting (only if we have successful results)
    if successful:
        st.subheader("📊 Performance Summary")
        
        comparison_data = []
        for method, result in successful.items():
            metrics = result.get('metrics', {})
            time_val = metrics.get('time')
            comparison_data.append({
                'Method': method,
                'Time (s)': f"{time_val:.6f}" if time_val is not None else "N/A",
                'Iterations': metrics.get('iterations', 'N/A'),
                'Final L1 Norm': f"{metrics.get('final_l1', 0):.8f}" if metrics.get('final_l1') else "N/A",
                'Converged': '✅ Yes' if metrics.get('converged') else '❌ No'
            })
        
            df_comparison = pd.DataFrame(comparison_data)
            
            # Style the dataframe
            styled_df = df_comparison.style.set_properties(**{
                'text-align': 'center',
                'font-size': '14px'
            }).set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#4CAF50'), ('color', 'white'), ('font-weight', 'bold')]},
                {'selector': 'td', 'props': [('padding', '10px')]}
            ])
            
            st.dataframe(styled_df, width='stretch', height=150)
    
    # Speedup analysis with enhanced visualization
    if 'Serial' in successful:
        serial_metrics = successful['Serial'].get('metrics', {})
        serial_time = serial_metrics.get('time')
        if serial_time is not None and serial_time > 0:
            st.subheader("⚡ Speedup Analysis")
            
            speedup_data = []
            efficiency_data = []
            for method in ['Pthreads', 'MPI']:
                if method in successful:
                    method_metrics = successful[method].get('metrics', {})
                    method_time = method_metrics.get('time')
                    if method_time is not None and method_time > 0:
                        speedup = serial_time / method_time
                        speedup_data.append({
                            'Method': method, 
                            'Speedup': speedup, 
                            'Time (s)': f"{method_time:.6f}",
                            'Serial Time (s)': f"{serial_time:.6f}"
                        })
                        # Calculate efficiency (assuming 4 threads/processes)
                        efficiency = speedup / 4.0  # Adjust based on actual threads used
                        efficiency_data.append({'Method': method, 'Efficiency': efficiency})
            
            if speedup_data:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Speedup Metrics**")
                    df_speedup = pd.DataFrame(speedup_data)
                    st.dataframe(df_speedup, width='stretch', hide_index=True)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    methods = [d['Method'] for d in speedup_data]
                    speedups = [d['Speedup'] for d in speedup_data]
                    colors = ['#2ecc71', '#3498db']
                    bars = ax.bar(methods, speedups, color=colors[:len(methods)], alpha=0.8, edgecolor='black', linewidth=2)
                    ax.set_ylabel('Speedup (x)', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Implementation', fontsize=14, fontweight='bold')
                    ax.set_title('Speedup vs Serial Implementation', fontsize=16, fontweight='bold', pad=20)
                    ax.axhline(y=1, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Baseline (Serial = 1x)')
                    ax.legend(fontsize=12)
                    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
                    
                    # Add value labels on bars
                    for bar, speedup in zip(bars, speedups):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                               f'{speedup:.2f}x', ha='center', va='bottom', 
                               fontweight='bold', fontsize=14, color='darkgreen')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col3:
                    # Efficiency chart
                    if efficiency_data:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        methods_eff = [d['Method'] for d in efficiency_data]
                        efficiencies = [d['Efficiency'] for d in efficiency_data]
                        bars = ax.bar(methods_eff, efficiencies, color=['#f39c12', '#9b59b6'], 
                                    alpha=0.8, edgecolor='black', linewidth=2)
                        ax.set_ylabel('Efficiency', fontsize=14, fontweight='bold')
                        ax.set_xlabel('Implementation', fontsize=14, fontweight='bold')
                        ax.set_title('Parallel Efficiency', fontsize=16, fontweight='bold', pad=20)
                        ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.7, linewidth=2, label='Ideal (100%)')
                        ax.set_ylim([0, max(1.2, max(efficiencies) * 1.2)])
                        ax.legend(fontsize=12)
                        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
                        
                        for bar, eff in zip(bars, efficiencies):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                   f'{eff:.2%}', ha='center', va='bottom', 
                                   fontweight='bold', fontsize=14)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
            else:
                st.info("⏸️ No speedup data available (Serial time is 0 or no parallel methods completed)")
    
    # Execution time comparison with enhanced visuals (only if we have successful results)
    if successful:
        st.subheader("⏱️ Performance Metrics Comparison")
        
        col1, col2 = st.columns(2)
        
        # Initialize variables for later use
        methods_with_time = []
        times_with_time = []
        methods_with_iters = []
        iterations_with_iters = []
        
        with col1:
            # Filter methods that have time data
            for method, result in successful.items():
                metrics = result.get('metrics', {})
                time_val = metrics.get('time')
                if time_val is not None:
                    methods_with_time.append(method)
                    times_with_time.append(time_val)
            
            if methods_with_time:
                colors_map = {'Serial': '#e74c3c', 'Pthreads': '#2ecc71', 'MPI': '#3498db'}
                colors = [colors_map.get(m, '#95a5a6') for m in methods_with_time]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(methods_with_time, times_with_time, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
                ax.set_ylabel('Time (seconds)', fontsize=14, fontweight='bold')
                ax.set_xlabel('Implementation', fontsize=14, fontweight='bold')
                ax.set_title('Execution Time Comparison', fontsize=16, fontweight='bold', pad=20)
                ax.grid(True, alpha=0.3, axis='y', linestyle='--')
                
                # Add value labels
                if times_with_time:
                    max_time = max(times_with_time)
                    for bar, time_val in zip(bars, times_with_time):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + max_time * 0.01,
                               f'{time_val:.6f}s', ha='center', va='bottom', 
                               fontweight='bold', fontsize=12, color='darkred')
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("⏸️ No timing data available for comparison")
        
        with col2:
            # Iterations comparison
            for method, result in successful.items():
                metrics = result.get('metrics', {})
                iters = metrics.get('iterations')
                if iters is not None:
                    methods_with_iters.append(method)
                    iterations_with_iters.append(iters)
            
            if methods_with_iters:
                colors_map = {'Serial': '#e74c3c', 'Pthreads': '#2ecc71', 'MPI': '#3498db'}
                colors = [colors_map.get(m, '#95a5a6') for m in methods_with_iters]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(methods_with_iters, iterations_with_iters, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
                ax.set_ylabel('Iterations', fontsize=14, fontweight='bold')
                ax.set_xlabel('Implementation', fontsize=14, fontweight='bold')
                ax.set_title('Convergence Iterations Comparison', fontsize=16, fontweight='bold', pad=20)
                ax.grid(True, alpha=0.3, axis='y', linestyle='--')
                
                # Add value labels
                if iterations_with_iters:
                    max_iters = max(iterations_with_iters)
                    for bar, iters in zip(bars, iterations_with_iters):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + max_iters * 0.01,
                               str(iters), ha='center', va='bottom', 
                               fontweight='bold', fontsize=14, color='darkblue')
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("⏸️ No iteration data available for comparison")
        
        # Combined comparison chart (only if we have data)
        if methods_with_time or methods_with_iters:
            st.markdown("---")
            st.subheader("📈 Side-by-Side Performance Comparison")
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            ax1, ax2 = axes
            
            # Time comparison
            if methods_with_time and times_with_time:
                colors_map = {'Serial': '#e74c3c', 'Pthreads': '#2ecc71', 'MPI': '#3498db'}
                colors = [colors_map.get(m, '#95a5a6') for m in methods_with_time]
                bars1 = ax1.bar(methods_with_time, times_with_time, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
                ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Implementation', fontsize=12, fontweight='bold')
                ax1.set_title('Execution Time', fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
                max_time = max(times_with_time) if times_with_time else 0
                for bar, time_val in zip(bars1, times_with_time):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + max_time * 0.01,
                           f'{time_val:.6f}s', ha='center', va='bottom', fontweight='bold', fontsize=10)
            else:
                ax1.text(0.5, 0.5, 'No timing data', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Execution Time', fontsize=14, fontweight='bold')
            
            # Iterations comparison
            if methods_with_iters and iterations_with_iters:
                colors_map = {'Serial': '#e74c3c', 'Pthreads': '#2ecc71', 'MPI': '#3498db'}
                colors = [colors_map.get(m, '#95a5a6') for m in methods_with_iters]
                bars2 = ax2.bar(methods_with_iters, iterations_with_iters, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
                ax2.set_ylabel('Iterations', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Implementation', fontsize=12, fontweight='bold')
                ax2.set_title('Convergence Iterations', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
                max_iters = max(iterations_with_iters) if iterations_with_iters else 0
                for bar, iters in zip(bars2, iterations_with_iters):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + max_iters * 0.01,
                           str(iters), ha='center', va='bottom', fontweight='bold', fontsize=12)
            else:
                ax2.text(0.5, 0.5, 'No iteration data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Convergence Iterations', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Enhanced Convergence comparison (only if we have successful results)
    if successful:
        st.subheader("📉 Convergence Analysis")
        st.markdown("Compare how each method converges to the solution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # L1 Norm convergence
            convergence_fig, ax = plt.subplots(figsize=(12, 7))
            
            colors_map = {'Serial': '#e74c3c', 'Pthreads': '#2ecc71', 'MPI': '#3498db'}
            markers = {'Serial': 'o', 'Pthreads': 's', 'MPI': '^'}
            line_styles = {'Serial': '-', 'Pthreads': '--', 'MPI': '-.'}
            
            has_conv_data = False
            for method, result in successful.items():
                metrics = result.get('metrics', {})
                conv_data = metrics.get('convergence_data', [])
                if conv_data:
                    has_conv_data = True
                    iterations = [d['iteration'] for d in conv_data]
                    l1_norms = [d['l1_norm'] for d in conv_data]
                    ax.semilogy(iterations, l1_norms, 
                               marker=markers.get(method, 'o'),
                               label=method, 
                               linewidth=3, 
                               markersize=8,
                               linestyle=line_styles.get(method, '-'),
                               color=colors_map.get(method, '#95a5a6'),
                               markerfacecolor='white',
                               markeredgewidth=2)
            
            if has_conv_data:
                ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
                ax.set_ylabel('L1 Norm (log scale)', fontsize=14, fontweight='bold')
                ax.set_title('Convergence Comparison - L1 Norm', fontsize=16, fontweight='bold', pad=20)
                ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
                ax.grid(True, alpha=0.3, linestyle='--')
                plt.tight_layout()
                st.pyplot(convergence_fig)
            else:
                st.info("⏸️ No convergence data available")
        
        with col2:
            # Max Error convergence
            error_fig, ax = plt.subplots(figsize=(12, 7))
            
            has_error_data = False
            for method, result in successful.items():
                metrics = result.get('metrics', {})
                conv_data = metrics.get('convergence_data', [])
                if conv_data:
                    has_error_data = True
                    iterations = [d['iteration'] for d in conv_data]
                    max_errors = [d['max_error'] for d in conv_data]
                    ax.semilogy(iterations, max_errors, 
                               marker=markers.get(method, 'o'),
                               label=method, 
                               linewidth=3, 
                               markersize=8,
                               linestyle=line_styles.get(method, '-'),
                               color=colors_map.get(method, '#95a5a6'),
                               markerfacecolor='white',
                               markeredgewidth=2)
            
            if has_error_data:
                ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
                ax.set_ylabel('Max Error (log scale)', fontsize=14, fontweight='bold')
                ax.set_title('Convergence Comparison - Max Error', fontsize=16, fontweight='bold', pad=20)
                ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
                ax.grid(True, alpha=0.3, linestyle='--')
                plt.tight_layout()
                st.pyplot(error_fig)
            else:
                st.info("⏸️ No error data available")
    
    # Enhanced PageRank values comparison (only if we have successful results)
    if successful:
        pagerank_comparison = {}
        for method, result in successful.items():
            metrics = result.get('metrics', {})
            if metrics.get('pagerank_values'):
                pagerank_comparison[method] = metrics['pagerank_values']
        
        if len(pagerank_comparison) > 1:
            st.markdown("---")
            st.subheader("🎯 PageRank Values Comparison")
            st.markdown("Compare the final PageRank values computed by each method")
        
        # Get all nodes
        all_nodes = set()
        for pr_values in pagerank_comparison.values():
            all_nodes.update(pr_values.keys())
        all_nodes = sorted(all_nodes)
        
        if len(all_nodes) <= 20:  # Only show if reasonable number of nodes
            comparison_df = pd.DataFrame({
                method: [pagerank_comparison[method].get(node, 0) for node in all_nodes]
                for method in pagerank_comparison.keys()
            }, index=[f'Node {n}' for n in all_nodes])
            
            # Format the dataframe
            styled_pr_df = comparison_df.style.format("{:.8f}").set_properties(**{
                'text-align': 'center',
                'font-size': '12px'
            }).set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#3498db'), ('color', 'white'), ('font-weight', 'bold')]},
                {'selector': 'td', 'props': [('padding', '8px')]}
            ])
            
            st.dataframe(styled_pr_df, width='stretch', height=300)
            
            # Enhanced visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart comparison
                fig, ax = plt.subplots(figsize=(14, 7))
                x = np.arange(len(all_nodes))
                width = 0.25
                
                for i, method in enumerate(pagerank_comparison.keys()):
                    values = [pagerank_comparison[method].get(node, 0) for node in all_nodes]
                    offset = (i - len(pagerank_comparison)/2 + 0.5) * width
                    ax.bar(x + offset, values, width, label=method, 
                          color=colors_map.get(method, '#95a5a6'), alpha=0.8, edgecolor='black', linewidth=1.5)
                
                ax.set_xlabel('Node', fontsize=14, fontweight='bold')
                ax.set_ylabel('PageRank Value', fontsize=14, fontweight='bold')
                ax.set_title('PageRank Values Comparison by Node', fontsize=16, fontweight='bold', pad=20)
                ax.set_xticks(x)
                ax.set_xticklabels([f'Node {n}' for n in all_nodes], rotation=45, ha='right')
                ax.legend(fontsize=12, loc='upper right')
                ax.grid(True, alpha=0.3, axis='y', linestyle='--')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                # Top nodes comparison
                st.markdown("**Top 5 Ranked Nodes**")
                top_nodes_data = []
                for method in pagerank_comparison.keys():
                    sorted_nodes = sorted(pagerank_comparison[method].items(), key=lambda x: x[1], reverse=True)
                    top_nodes_data.append({
                        'Method': method,
                        'Top Nodes': ', '.join([f"Node {n}" for n, _ in sorted_nodes[:5]])
                    })
                
                top_df = pd.DataFrame(top_nodes_data)
                st.dataframe(top_df, width='stretch', hide_index=True)
                
                # Heatmap-style visualization
                if len(all_nodes) <= 10 and len(all_nodes) > 0 and len(pagerank_comparison) > 0:
                    try:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        heatmap_data = np.array([
                            [pagerank_comparison[method].get(node, 0) for node in all_nodes]
                            for method in pagerank_comparison.keys()
                        ])
                        
                        # Check if heatmap_data has valid shape
                        if heatmap_data.size > 0 and len(heatmap_data.shape) == 2:
                            im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
                            ax.set_xticks(np.arange(len(all_nodes)))
                            ax.set_yticks(np.arange(len(pagerank_comparison.keys())))
                            ax.set_xticklabels([f'Node {n}' for n in all_nodes])
                            ax.set_yticklabels(list(pagerank_comparison.keys()))
                            ax.set_xlabel('Node', fontsize=12, fontweight='bold')
                            ax.set_ylabel('Method', fontsize=12, fontweight='bold')
                            ax.set_title('PageRank Values Heatmap', fontsize=14, fontweight='bold', pad=15)
                            
                            # Add text annotations
                            for i in range(len(pagerank_comparison.keys())):
                                for j in range(len(all_nodes)):
                                    text = ax.text(j, i, f'{heatmap_data[i, j]:.4f}',
                                                 ha="center", va="center", color="black", fontsize=9, fontweight='bold')
                            
                            plt.colorbar(im, ax=ax, label='PageRank Value')
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.info("⏸️ Heatmap data is empty or invalid")
                    except Exception as e:
                        st.warning(f"Could not create heatmap: {e}")
    
    # Method Comparison: Technical Differences
    st.markdown("---")
    st.subheader("🔍 Method Comparison: Technical Differences")
    
    comparison_tabs = st.tabs(["📊 Performance", "💾 Memory & Scalability", "🎯 Use Cases", "⚙️ Implementation Details"])
    
    with comparison_tabs[0]:
        st.markdown("### Performance Characteristics")
        perf_data = []
        for method, result in successful.items():
            metrics = result.get('metrics', {})
            time_val = metrics.get('time', 0)
            iterations = metrics.get('iterations', 0)
            perf_data.append({
                'Method': method,
                'Execution Time (s)': f"{time_val:.6f}" if time_val else "N/A",
                'Iterations': iterations,
                'Time per Iteration (ms)': f"{(time_val * 1000 / iterations):.4f}" if time_val and iterations > 0 else "N/A",
                'Convergence Rate': 'Fast' if iterations < 20 else 'Medium' if iterations < 50 else 'Slow'
            })
        
        df_perf = pd.DataFrame(perf_data)
        st.dataframe(df_perf, width='stretch', hide_index=True)
        
        # Performance insights
        if 'Serial' in successful:
            serial_time = successful['Serial'].get('metrics', {}).get('time')
            if serial_time is not None and serial_time > 0:
                st.markdown("#### 🚀 Performance Insights")
                for method in ['Pthreads', 'MPI']:
                    if method in successful:
                        method_time = successful[method].get('metrics', {}).get('time')
                        if method_time is not None and method_time > 0:
                            speedup = serial_time / method_time
                            if speedup > 1.5:
                                st.success(f"✅ **{method}** shows **{speedup:.2f}x speedup** - Excellent parallelization!")
                            elif speedup > 1.1:
                                st.info(f"ℹ️ **{method}** shows **{speedup:.2f}x speedup** - Good parallelization")
                            else:
                                st.warning(f"⚠️ **{method}** shows **{speedup:.2f}x speedup** - Overhead may be limiting performance")
                        elif method_time is None:
                            st.info(f"ℹ️ **{method}** completed successfully but execution time was not reported")
            elif serial_time is None:
                st.info("ℹ️ Serial implementation completed but execution time was not reported - cannot calculate speedup")
    
    with comparison_tabs[1]:
        st.markdown("### Memory Usage & Scalability")
        
        if successful:
            # Estimate memory usage (rough estimates)
            memory_estimates = []
            for method, result in successful.items():
                metrics = result.get('metrics', {})
                # Rough estimate: nodes * sizeof(double) for PageRank vector
                # Serial: 1 vector, Pthreads: 1 vector + thread overhead, MPI: distributed vectors
                graph_nodes = num_nodes if num_nodes else 4
            base_memory = graph_nodes * 8  # 8 bytes per double
            if method == 'Serial':
                estimated_mb = (base_memory * 2) / (1024 * 1024)  # Current + previous vector
                memory_type = "Single Process"
            elif method == 'Pthreads':
                estimated_mb = (base_memory * 2.5) / (1024 * 1024)  # Slightly more for thread overhead
                memory_type = "Shared Memory"
            else:  # MPI
                estimated_mb = (base_memory * 2) / (1024 * 1024)  # Distributed, but same per process
                memory_type = "Distributed Memory"
            
            memory_estimates.append({
                'Method': method,
                'Memory Type': memory_type,
                'Estimated Memory (MB)': f"{estimated_mb:.4f}",
                'Scalability': 'Single Machine' if method != 'MPI' else 'Cluster',
                'Max Nodes (est.)': '~1M (single machine)' if method != 'MPI' else 'Unlimited (cluster)'
            })
        
        df_memory = pd.DataFrame(memory_estimates)
        st.dataframe(df_memory, width='stretch', hide_index=True)
        
        st.markdown("#### 📈 Scalability Insights")
        st.info("""
        - **Serial**: Best for small graphs (< 10K nodes). Simple, no overhead.
        - **Pthreads**: Best for medium graphs (10K-1M nodes) on multi-core machines. Shared memory allows efficient communication.
        - **MPI**: Best for large graphs (> 1M nodes) or distributed systems. Can scale across multiple machines.
        """)
    
    with comparison_tabs[2]:
        st.markdown("### Recommended Use Cases")
        
        use_cases = {
            'Serial': {
                'Best For': ['Small graphs (< 10K nodes)', 'Development & testing', 'Simple deployments'],
                'Advantages': ['No dependencies', 'Simple to run', 'Low memory overhead', 'Easy to debug'],
                'Limitations': ['Single core only', 'Does not scale', 'Slow for large graphs']
            },
            'Pthreads': {
                'Best For': ['Medium graphs (10K-1M nodes)', 'Multi-core machines', 'Shared memory systems'],
                'Advantages': ['Utilizes multiple cores', 'Shared memory efficiency', 'Good speedup on modern CPUs', 'No network overhead'],
                'Limitations': ['Limited to single machine', 'Requires pthreads library', 'Memory shared (can be bottleneck)']
            },
            'MPI': {
                'Best For': ['Large graphs (> 1M nodes)', 'Distributed systems', 'Clusters', 'Cloud computing'],
                'Advantages': ['Scales across machines', 'Handles very large graphs', 'Distributed memory', 'Industry standard'],
                'Limitations': ['Network communication overhead', 'Requires MPI installation', 'More complex setup', 'Slower for small graphs']
            }
        }
        
        for method, details in use_cases.items():
            if method in successful:
                with st.expander(f"📌 {method} Use Cases", expanded=(method == 'Serial')):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**✅ Best For:**")
                        for use_case in details['Best For']:
                            st.markdown(f"- {use_case}")
                        st.markdown("**✨ Advantages:**")
                        for advantage in details['Advantages']:
                            st.markdown(f"- {advantage}")
                    with col2:
                        st.markdown("**⚠️ Limitations:**")
                        for limitation in details['Limitations']:
                            st.markdown(f"- {limitation}")
    
    with comparison_tabs[3]:
        st.markdown("### Implementation Details")
        
        impl_details = {
            'Serial': {
                'Parallelization': 'None (Sequential)',
                'Communication': 'N/A',
                'Synchronization': 'N/A',
                'Data Structure': 'CSR (Compressed Sparse Row)',
                'Algorithm': 'Iterative Power Method',
                'Complexity': 'O(E × I) where E=edges, I=iterations'
            },
            'Pthreads': {
                'Parallelization': 'Multi-threading (Shared Memory)',
                'Communication': 'Shared Memory (Lock-free where possible)',
                'Synchronization': 'Barriers between iterations',
                'Data Structure': 'CSR (Shared across threads)',
                'Algorithm': 'Parallel Iterative Power Method',
                'Complexity': 'O(E × I / P) where P=threads (theoretical)'
            },
            'MPI': {
                'Parallelization': 'Multi-processing (Distributed Memory)',
                'Communication': 'MPI AllReduce for synchronization',
                'Synchronization': 'MPI barriers and reductions',
                'Data Structure': 'Distributed CSR',
                'Algorithm': 'Distributed Iterative Power Method',
                'Complexity': 'O(E × I / P + communication overhead)'
            }
        }
        
        impl_data = []
        for method, details in impl_details.items():
            if method in successful:
                row = {'Method': method}
                row.update(details)
                impl_data.append(row)
        
        df_impl = pd.DataFrame(impl_data)
        st.dataframe(df_impl, width='stretch', hide_index=True)
        
        st.markdown("#### 🔬 Algorithmic Differences")
        st.markdown("""
        **Serial Implementation:**
        - Processes nodes sequentially
        - Single PageRank vector update per iteration
        - No parallelization overhead
        
        **Pthreads Implementation:**
        - Divides nodes among threads
        - Each thread updates its portion of the PageRank vector
        - Uses barriers to synchronize between iterations
        - Memory is shared, so no data copying needed
        
        **MPI Implementation:**
        - Distributes nodes across processes
        - Each process computes PageRank for its portion
        - Uses MPI_AllReduce to combine results each iteration
        - Requires network communication (adds latency)
        """)
    
    # Summary insights
    st.markdown("---")
    st.subheader("💡 Key Insights & Recommendations")
    
    if len(successful) >= 2:
        insights = []
        recommendations = []
        
        # Find fastest method (only if time is available)
        methods_with_time = {k: v for k, v in successful.items() 
                            if v.get('metrics', {}).get('time') is not None}
        if methods_with_time:
            fastest = min(methods_with_time.items(), 
                         key=lambda x: x[1].get('metrics', {}).get('time', float('inf')))
            fastest_time = fastest[1].get('metrics', {}).get('time')
            if fastest_time is not None and fastest_time > 0:
                insights.append(f"🏆 **Fastest Method:** {fastest[0]} ({fastest_time:.6f}s)")
        
        # Speedup insights
        if 'Serial' in successful:
            serial_time = successful['Serial'].get('metrics', {}).get('time')
            if serial_time is not None and serial_time > 0:
                for method in ['Pthreads', 'MPI']:
                    if method in successful:
                        method_time = successful[method].get('metrics', {}).get('time')
                        if method_time is not None and method_time > 0:
                            speedup = serial_time / method_time
                            insights.append(f"⚡ **{method} Speedup:** {speedup:.2f}x faster than Serial")
                            
                            # Recommendations based on speedup and graph characteristics
                            if graph_analysis:
                                if speedup > 2.0:
                                    recommendations.append(f"✅ **Use {method}** for this graph - excellent {speedup:.2f}x speedup!")
                                elif speedup > 1.2:
                                    recommendations.append(f"✅ **{method}** shows good {speedup:.2f}x speedup for this graph")
                                elif speedup < 1.0:
                                    recommendations.append(f"⚠️ **Stick with Serial** - {method} overhead ({1/speedup:.2f}x slower) is too high for this graph size")
                            else:
                                if speedup > 2.0:
                                    recommendations.append(f"✅ **Use {method}** for this graph size - excellent speedup!")
                                elif speedup < 1.0:
                                    recommendations.append(f"⚠️ **Stick with Serial** - {method} overhead is too high for this graph")
        
        # Convergence insights
        convergence_diff = {}
        for method, result in successful.items():
            metrics = result.get('metrics', {})
            if metrics.get('iterations'):
                convergence_diff[method] = metrics['iterations']
        
        if len(set(convergence_diff.values())) == 1:
            insights.append("🔄 **All methods** converged in the same number of iterations (algorithm correctness verified)")
        else:
            min_iter = min(convergence_diff.values())
            min_method = [k for k, v in convergence_diff.items() if v == min_iter][0]
            insights.append(f"🔄 **{min_method}** converged fastest ({min_iter} iterations)")
        
        # Graph size recommendations
        graph_nodes = num_nodes if num_nodes else 4
        if graph_nodes < 100:
            recommendations.append("💡 **Recommendation:** For small graphs like this, Serial is usually sufficient")
        elif graph_nodes < 10000:
            recommendations.append("💡 **Recommendation:** For medium graphs, Pthreads typically provides the best balance")
        else:
            recommendations.append("💡 **Recommendation:** For large graphs, consider MPI for distributed processing")
        
        if insights:
            st.markdown("#### 📊 Performance Insights")
            for insight in insights:
                st.markdown(f"- {insight}")
        
        if recommendations:
            st.markdown("#### 🎯 Recommendations")
            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            st.info("Run comparisons to see insights here!")
    
    # Detailed results expander
    with st.expander("📄 View Detailed Outputs"):
        for method, result in successful.items():
            st.subheader(f"{method} Output")
            st.code(result.get('output', ''), language='text')

st.title("🚀 PageRank Algorithm Dashboard")
st.markdown("Interactive dashboard for running and analyzing PageRank implementations with comparison visualizations")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Mode selection
mode = st.sidebar.radio(
    "Mode",
    ["Single Implementation", "Compare All Methods"]
)

# Graph input
graph_input_method = st.sidebar.radio(
    "Graph Input Method",
    ["Generate Graph", "Upload File", "Use Sample Graph"]
)

graph_file = None
nodes = 4  # Default number of nodes

if graph_input_method == "Generate Graph":
    st.sidebar.subheader("Graph Generation")
    nodes = st.sidebar.number_input("Number of Nodes", min_value=2, max_value=1000, value=10, step=1, key="gen_nodes")
    graph_type = st.sidebar.selectbox(
        "Graph Type",
        ["Random", "Scale-Free (Barabási–Albert)", "Complete", "Ring", "Star"]
    )
    density = 0.3
    if graph_type == "Random":
        density = st.sidebar.slider("Edge Density", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
    elif graph_type == "Scale-Free (Barabási–Albert)":
        edges_per_node = st.sidebar.number_input("Edges per Node", min_value=1, max_value=min(5, nodes-1), value=2, step=1)
    
    if st.sidebar.button("🔧 Generate Graph", type="primary"):
        # Generate graph based on type
        G = nx.DiGraph()
        if graph_type == "Random":
            # Generate random directed graph
            np.random.seed(42)  # For reproducibility
            for i in range(nodes):
                for j in range(nodes):
                    if i != j and np.random.random() < density:
                        G.add_edge(i, j)
        elif graph_type == "Scale-Free (Barabási–Albert)":
            # Create scale-free network using preferential attachment
            if nodes > edges_per_node:
                G_undirected = nx.barabasi_albert_graph(nodes, edges_per_node, seed=42)
                # Convert to directed graph (both directions)
                for edge in G_undirected.edges():
                    G.add_edge(edge[0], edge[1])
                    G.add_edge(edge[1], edge[0])
            else:
                # Fallback for small graphs
                for i in range(nodes):
                    for j in range(i+1, min(i+edges_per_node+1, nodes)):
                        G.add_edge(i, j)
                        G.add_edge(j, i)
        elif graph_type == "Complete":
            # Complete directed graph (every node connects to every other node)
            for i in range(nodes):
                for j in range(nodes):
                    if i != j:
                        G.add_edge(i, j)
        elif graph_type == "Ring":
            # Ring graph (each node connects to next)
            for i in range(nodes):
                G.add_edge(i, (i + 1) % nodes)
        elif graph_type == "Star":
            # Star graph (center node connects to all others)
            center = 0
            for i in range(1, nodes):
                G.add_edge(center, i)
                G.add_edge(i, center)
        
        # Ensure at least one edge
        if len(G.edges()) == 0:
            G.add_edge(0, 1)
            if nodes > 2:
                G.add_edge(1, 2)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', dir=tempfile.gettempdir()) as f:
            for edge in G.edges():
                f.write(f"{edge[0]}\t{edge[1]}\n")
            generated_file = os.path.abspath(f.name)
            st.session_state['generated_graph_file'] = generated_file
            st.session_state['generated_graph_nodes'] = len(G.nodes())
            st.session_state['generated_graph_edges'] = len(G.edges())
            st.session_state['generated_graph_type'] = graph_type
        
        st.sidebar.success(f"✅ Generated {graph_type} graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    # Use generated graph if available
    if 'generated_graph_file' in st.session_state and os.path.exists(st.session_state['generated_graph_file']):
        graph_file = st.session_state['generated_graph_file']
        st.sidebar.info(f"📊 Using generated {st.session_state.get('generated_graph_type', 'graph')} graph")
        
elif graph_input_method == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload Graph File", type=['txt'])
    if uploaded_file:
        # Save to temporary file with absolute path
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', dir=tempfile.gettempdir()) as f:
            content = uploaded_file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            f.write(content)
            graph_file = os.path.abspath(f.name)
    
    # Sample files download section
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Download Sample Graphs")
    sample_files = {
        "Small Graph (4 nodes)": "sample_graphs/small_graph.txt",
        "Medium Graph (10 nodes)": "sample_graphs/medium_graph.txt",
        "Large Graph (20 nodes)": "sample_graphs/large_graph.txt"
    }
    for name, path in sample_files.items():
        sample_path = Path(__file__).parent.parent / path
        if sample_path.exists():
            with open(sample_path, 'r') as f:
                st.sidebar.download_button(
                    label=f"⬇️ {name}",
                    data=f.read(),
                    file_name=os.path.basename(path),
                    mime="text/plain"
                )
else:
    # Use sample graph
    sample_graph = """0	1
0	2
1	2
2	0
3	0
3	1
3	2"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', dir=tempfile.gettempdir()) as f:
        f.write(sample_graph)
        graph_file = os.path.abspath(f.name)
    st.sidebar.info("Using sample graph with 4 nodes")

if graph_file:
    # Count actual nodes in graph file
    try:
        G_check = nx.DiGraph()
        with open(graph_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        from_node = int(parts[0])
                        to_node = int(parts[1])
                        G_check.add_edge(from_node, to_node)
                    except ValueError:
                        continue
        actual_nodes = max(G_check.nodes()) + 1 if len(G_check.nodes()) > 0 else 4
        nodes = actual_nodes
    except:
        actual_nodes = 4
        nodes = 4
    
    # Algorithm parameters
    st.sidebar.subheader("Algorithm Parameters")
    nodes = st.sidebar.number_input("Number of Nodes", min_value=1, value=actual_nodes, step=1, key="alg_nodes")
    threshold = st.sidebar.number_input("Convergence Threshold", min_value=1e-6, value=0.0001, format="%.6f")
    d = st.sidebar.slider("Damping Factor (d)", min_value=0.0, max_value=1.0, value=0.85, step=0.05)
    
    if mode == "Single Implementation":
        # Implementation selection
        st.sidebar.subheader("Implementation")
        implementation = st.sidebar.selectbox(
            "Select Implementation",
            ["Serial", "Pthreads", "MPI"]
        )
        
        num_threads = 1
        if implementation == "Pthreads":
            num_threads = st.sidebar.number_input("Number of Threads", min_value=1, max_value=64, value=4, step=1)
        elif implementation == "MPI":
            num_threads = st.sidebar.number_input("Number of Processes", min_value=1, max_value=64, value=4, step=1)
        
        # Run button
        if st.sidebar.button("🚀 Run PageRank", type="primary"):
            with st.spinner("Running PageRank algorithm..."):
                # Determine command based on implementation
                project_root = get_project_root()
                
                if implementation == "Serial":
                    exe_path = project_root / "pagerank_serial" / f"pagerank_serial{exe_ext}"
                    if not exe_path.exists():
                        st.error(f"❌ Executable not found: {exe_path}")
                        st.info("💡 Run `.\build_all.bat` in the project root to build all executables.")
                        st.stop()
                    cmd = [str(exe_path), graph_file, str(nodes), str(threshold), str(d)]
                elif implementation == "Pthreads":
                    exe_path = project_root / "pagerank_pthreads" / f"pagerank_pthreads{exe_ext}"
                    if not exe_path.exists():
                        st.error(f"❌ Executable not found: {exe_path}")
                        st.info("💡 Run `.\build_all.bat` in the project root to build all executables.")
                        st.stop()
                    cmd = [str(exe_path), graph_file, str(nodes), str(threshold), str(d), str(num_threads)]
                else:  # MPI
                    exe_path = project_root / "pagerank_mpi" / f"pagerank_mpi{exe_ext}"
                    if not exe_path.exists():
                        st.error(f"❌ Executable not found: {exe_path}")
                        st.info("💡 Run `.\build_all.bat` in the project root to build all executables.")
                        st.stop()
                    cmd = ["mpiexec", "-n", str(num_threads), str(exe_path),
                           graph_file, str(nodes), str(threshold), str(d)]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, 
                                          cwd=project_root)
                    
                    if result.returncode == 0:
                        st.success("✅ PageRank completed successfully!")
                        st.session_state['output'] = result.stdout
                        st.session_state['implementation'] = implementation
                        st.session_state['num_threads'] = num_threads
                        st.session_state['graph_file'] = graph_file
                    else:
                        st.error(f"❌ Error: {result.stderr}")
                        if result.stdout:
                            st.text("Output:")
                            st.code(result.stdout)
                except subprocess.TimeoutExpired:
                    st.error("⏱️ Execution timed out")
                except FileNotFoundError as e:
                    st.error(f"❌ Executable not found: {e}")
                    st.info(r"""
                    **To fix this:**
                    1. Open PowerShell in the project root directory
                    2. Run: `.\build_all.bat`
                    3. Wait for all builds to complete
                    4. Refresh this page and try again
                    
                    The executables should be in:
                    - `pagerank_serial\pagerank_serial.exe`
                    - `pagerank_pthreads\pagerank_pthreads.exe`
                    - `pagerank_mpi\pagerank_mpi.exe`
                    """)
        
        # Display single implementation results
        if 'output' in st.session_state:
            display_single_results(st.session_state['output'], 
                                 st.session_state['implementation'],
                                 st.session_state.get('num_threads', 1))
    
    else:  # Compare All Methods
        st.sidebar.subheader("Parallel Configuration")
        num_threads = st.sidebar.number_input("Number of Threads/Processes", min_value=1, max_value=64, value=4, step=1)
        
        if st.sidebar.button("🔬 Compare All Methods", type="primary"):
            with st.spinner("Running all PageRank implementations for comparison..."):
                results = run_all_implementations(graph_file, nodes, threshold, d, num_threads)
                st.session_state['comparison_results'] = results
                st.session_state['graph_file'] = graph_file
        
        # Display comparison results
        if 'comparison_results' in st.session_state:
            display_comparison_results(st.session_state['comparison_results'], nodes, graph_file)

    # Graph visualization - Always show if graph_file is set
    if graph_file:
        st.markdown("---")
        st.header("📊 Graph Visualization")
        
        # Check if file exists
        if not os.path.exists(graph_file):
            st.error(f"❌ Graph file not found: {graph_file}")
            st.info("💡 Please generate, upload, or select a sample graph.")
        else:
            try:
                G = nx.DiGraph()
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
                
                if len(G.nodes()) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        try:
                            pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
                            fig, ax = plt.subplots(figsize=(10, 8))
                            nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.8, ax=ax, node_color='lightblue', edgecolors='black', linewidths=1)
                            nx.draw_networkx_edges(G, pos, alpha=0.6, arrows=True, arrowsize=20, ax=ax, edge_color='gray', width=1.5)
                            nx.draw_networkx_labels(G, pos, font_size=10, ax=ax, font_weight='bold')
                            ax.set_title("Graph Structure", fontsize=16, fontweight='bold', pad=20)
                            ax.axis('off')
                            plt.tight_layout()
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error drawing graph: {e}")
                            # Fallback: show text representation
                            st.text("Graph edges:")
                            for edge in G.edges():
                                st.text(f"  {edge[0]} -> {edge[1]}")
                    
                    with col2:
                        st.subheader("Graph Statistics")
                        st.metric("Nodes", len(G.nodes()))
                        st.metric("Edges", len(G.edges()))
                        st.metric("Density", f"{nx.density(G):.4f}")
                        
                        # Additional stats
                        if len(G.nodes()) > 0:
                            in_degrees = [G.in_degree(n) for n in G.nodes()]
                            out_degrees = [G.out_degree(n) for n in G.nodes()]
                            st.metric("Avg In-Degree", f"{sum(in_degrees) / len(G.nodes()):.2f}")
                            st.metric("Avg Out-Degree", f"{sum(out_degrees) / len(G.nodes()):.2f}")
                        
                        if len(G.nodes()) <= 20:
                            st.subheader("Adjacency List")
                            adj_list = {}
                            for node in sorted(G.nodes()):
                                adj_list[node] = list(G.successors(node))
                            st.json(adj_list)
                        else:
                            st.info(f"Graph has {len(G.nodes())} nodes. Adjacency list not shown for large graphs.")
                else:
                    st.warning("⚠️ Graph file is empty or contains no valid edges.")
            except Exception as e:
                st.error(f"❌ Could not visualize graph: {e}")
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    # Change to project root
    os.chdir(Path(__file__).parent.parent)
