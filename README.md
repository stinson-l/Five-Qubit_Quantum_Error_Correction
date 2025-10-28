# 5-Qubit Quantum Error Correction 

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-0.45%2B-purple)](https://qiskit.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Quantum](https://img.shields.io/badge/Quantum-Error%20Correction-orange)](https://en.wikipedia.org/wiki/Quantum_error_correction)

A comprehensive implementation of the 5-qubit quantum error correction code, demonstrating fault-tolerant quantum computing principles through simulation of random Pauli errors and logical state preservation.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Mathematical Background](#mathematical-background)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [References](#references)
- [License](#license)

## Overview

This project implements a **Five-Qubit_Quantum_Error_Correction** that can correct arbitrary single-qubit errors. The code encodes one logical qubit into five physical qubits, demonstrating how quantum information can be protected from decoherence and errors in quantum systems.

### Key Objectives
-  Encode logical quantum states |ψ_L⟩ = α|0_L⟩ + β|1_L⟩
-  Simulate random Pauli errors (X, Y, Z) on physical qubits
-  Measure logical error rates and success probabilities
-  Visualize error correction performance vs. physical error rates
-  Demonstrate the error threshold theorem

## Features

- **Pure State Vector Simulation**: No classical bits or simulator backends required
- **Arbitrary Logical States**: Prepare any superposition α|0_L⟩ + β|1_L⟩
- **Random Pauli Error Channel**: Implements realistic quantum noise model
- **Performance Metrics**: Track both success rates and logical error rates
- **Visualization Tools**: Automatic plotting of results and trends
- **Threshold Analysis**: Identify the error correction threshold
- **Modular Design**: Easy to extend and modify for research purposes

## Mathematical Background

### The 5-Qubit Code

The 5-qubit code is the smallest quantum error-correcting code capable of correcting arbitrary single-qubit errors. It encodes:

```
|0_L⟩ = 1/4 Σ |c_i⟩  (sum over 16 specific basis states)
|1_L⟩ = X⊗5 |0_L⟩ (apply X gate to all 5 qubits)
```

#### Logical Codewords

The logical zero state consists of an equal superposition of 16 basis states:
```
|0_L⟩ = 1/4(|00000⟩ + |10010⟩ + |01001⟩ + |10100⟩ + 
         |01010⟩ + |00101⟩ + |11011⟩ + |00110⟩ +
         |11000⟩ + |11101⟩ + |11110⟩ + |01111⟩ +
         |10011⟩ + |01100⟩ + |10101⟩ + |00111⟩)
```

The logical one state is obtained by applying X gates to all qubits:
```
|1_L⟩ = X₁X₂X₃X₄X₅|0_L⟩
```

### Stabilizer Generators

The code is defined by four stabilizer generators:
- **S₁** = XZZXI
- **S₂** = IXZZX  
- **S₃** = XIXZZ
- **S₄** = ZXIXZ

These stabilizers satisfy:
- S_i|0_L⟩ = |0_L⟩ for all i
- S_i|1_L⟩ = |1_L⟩ for all i
- [S_i, S_j] = 0 (all stabilizers commute)

### Error Model

Random Pauli errors applied independently to each qubit:
```
ε(ρ) = (1-p)ρ + (p/3)(XρX† + YρY† + ZρZ†)
```

Where:
- **p**: Probability of error per qubit
- **X, Y, Z**: Pauli matrices
- Equal probability (1/3) for each Pauli error type

### Code Properties
- **Code distance**: d = 3
- **Number of logical qubits**: k = 1
- **Number of physical qubits**: n = 5
- **Error correction capability**: t = 1 qubit
- **Encoding rate**: R = k/n = 1/5

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/stinson-l/Five-Qubit_Quantum_Error_Correction.git
cd Five-Qubit_Quantum_Error_Correction
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv qec_env
source qec_env/bin/activate  # On Windows: qec_env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install qiskit qiskit-aer numpy matplotlib jupyter
```

### Requirements File (requirements.txt)
```txt
qiskit>=0.45.0
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
```

### Step 4: Launch Program

   ## Method 1: Using Jupyter Notebook 
   ```bash
   jupyter notebook Five-Qubit_Quantum_Error_Correction.ipynb
   ```

   ## Method 2: Using Jupyter Lab
   ```bash
   jupyter lab Five-Qubit_Quantum_Error_Correction.ipynb
   ```

   ## Method 3: Run like a Python script
   ```bash
   jupyter nbconvert --to script Five-Qubit_Quantum_Error_Correction.ipynb
   python Five-Qubit_Quantum_Error_Correction.py
   ```


## Project Structure

```
Five-Qubit_Quantum_Error_Correction/
│
├── Five-Qubit_Quantum_Error_Correction.ipynb    # Main simulation code
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└──  LICENSE                     # MIT License

```

## Implementation Details

### Core Functions

| Function | Purpose | Key Parameters | 
|----------|---------|----------------|
| `create_5qubit_encoder()` | Builds encoding circuit | None |
| `create_5qubit_decoder()` | Generates the decoder | None |
| `prepare_logical_state_from_bits(qc, data, x_bits)` | Prepares arbitrary logical state | 2-bit logical selector | 
| `sample_pauli_errors(p, num_qubits=5, rng=None)` | Applies noise model | Probability | 
| `apply_sampled_errors(qc, qr_data, paulis)` | Apply gates to the 5 data qubits | Qubit register, list of errors | 
| `append_syndrome_measurement(qc, data, anc, c_synd)` | Measures the four stabilizer generators | Data qubits, ancillas, classical syndrome bits |
| `append_conditional_correction(qc, data, c_synd)` | Recover a quantum error correction | Classical syndrome register |
| `build_qec_trial_circuit(x_bits, p, rng=None)` | Assemble a full circuit | x_bits, p | 
| `estimate_success_probability(x_bits, p, num_trials, shots)` | Execute the circuit | p, number of trials |
| `plot_results(x_bits, p_values, num_trials, shots)` | Visualize performance | Results dictionary | 

### Workflow Diagram

```
┌─────────────────────┐
│ Generate Random     │
│ State Parameters    │
│ (α, β)             │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ Encode Logical      │
│ State               │
│ |ψ_L⟩ = α|0_L⟩+β|1_L⟩│
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ Apply Random        │
│ Pauli Errors        │
│ ε(ρ) with prob. p   │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ Measure Logical     │
│ State               │
│ Project onto {0,1}_L│
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ Calculate Metrics   │
│ - Success Rate      │
│ - Logical Error Rate│
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ Repeat N Trials     │
│ Statistical Average │
└─────────────────────┘
```

### Algorithm Complexity

- **Encoding**: O(1) - Fixed number of gates
- **Error Application**: O(n) - Linear in number of qubits
- **Measurement**: O(2^n) - Exponential in worst case
- **Overall Simulation**: O(T × 2^n) where T = number of trials


### Performance Plots

The simulator generates two key visualizations:

1. **Success Rate vs. Physical Error Rate**
   - Shows probability of successful error correction
   - Demonstrates threshold behavior
   - Indicates regime where QEC is beneficial

2. **Logical Error Rate vs. Physical Error Rate** 
   - Log-scale plot showing quadratic suppression
   - Confirms P_logical ∝ p² for small p
   - Identifies break-even point


## References

### Primary Literature

1. **Original 5-Qubit Code Paper**
   - Laflamme, R., Miquel, C., Paz, J. P., & Zurek, W. H. (1996). "Perfect Quantum Error Correcting Code". *Physical Review Letters*, 77(1), 198.
   - DOI: [10.1103/PhysRevLett.77.198](https://doi.org/10.1103/PhysRevLett.77.198)

2. **Quantum Error Correction Theory**
   - Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information* (10th Anniversary Edition). Cambridge University Press.
   - ISBN: 978-1107002173

3. **Stabilizer Codes**
   - Gottesman, D. (1997). "Stabilizer Codes and Quantum Error Correction". *arXiv:quant-ph/9705052*.
   - arXiv: [quant-ph/9705052](https://arxiv.org/abs/quant-ph/9705052)

### Additional Resources

4. **Fault-Tolerant Quantum Computation**
   - Preskill, J. (1998). "Reliable quantum computers". *Proceedings of the Royal Society A*, 454(1969), 385-410.

5. **Qiskit Documentation**
   - [Qiskit Textbook: Quantum Error Correction](https://qiskit.org/textbook/ch-quantum-hardware/error-correction-repetition-code.html)
   - [Qiskit API Reference](https://qiskit.org/documentation/)

6. **Review Articles**
   - Terhal, B. M. (2015). "Quantum error correction for quantum memories". *Reviews of Modern Physics*, 87(2), 307.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 [Stinson Lee]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Author

**[Stinson Lee]**
- GitHub: [@stinson-l](https://github.com/stinson-l)



## Project Status


### Current Version
- **Version**: 1.0.0
- **Release Date**: October 2024
- **Python Support**: 3.8+
- **Qiskit Compatibility**: 0.45+


### Known Issues

- Memory usage scales exponentially with qubit number
- Simplified measurement model (no syndrome extraction)
- Gates assumed to be perfect (no gate errors)



</p>
