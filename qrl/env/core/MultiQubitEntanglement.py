'''
Implementation of MultiQubitEntanglementV1 environment
Author: Tanishq
License: Apache-2.0
'''

from gymnasium import spaces
from pennylane import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .base__ import QuantumEnv
from .utils import RX, RY, RZ

class MultiQubitEntanglementV0(QuantumEnv):
    """
    ## Description

    `MultiQubitEntanglementV0` is a multi-qubit quantum environment designed for teaching,
    visualization, and reinforcement learning (RL) experiments targeting **entangled states**
    such as GHZ or Bell states. The agent applies a discrete set of **single- and two-qubit gates**
    to steer the system from the fixed initial state |0...0⟩ to a **target entangled state**
    within a limited number of steps.

    The environment is built on top of a PennyLane quantum device and inherits from a
    `QuantumEnv` base class, providing a familiar RL-style interface (`reset()`, `step()`, `render()`).
    It supports **episodic training**, reward calculation based on fidelity with the target,
    and optional visualization of the computational basis amplitudes.

    ---
    
    ## Action Space

    The action space is **discrete**, where each integer corresponds to a quantum gate
    applied to one or two qubits. The set of available actions includes:

    | Num | Action        | Description |
    | --- | ------------- | ----------- |
    | 0   | `H_i`         | Hadamard gate on qubit i |
    | 1   | `X_i`         | Pauli-X gate on qubit i (NOT) |
    | 2   | `Y_i`         | Pauli-Y gate on qubit i |
    | 3   | `Z_i`         | Pauli-Z gate on qubit i |
    | 4   | `RX_i_pi_2`   | Rotation around X-axis by π/2 on qubit i |
    | 5   | `RY_i_pi_2`   | Rotation around Y-axis by π/2 on qubit i |
    | 6   | `RZ_i_pi_2`   | Rotation around Z-axis by π/2 on qubit i |
    | 7   | `CNOT_i_j`    | Controlled-NOT gate with control qubit i and target qubit j |

    - `i` and `j` are qubit indices ranging from 0 to `n_qubits - 1`.
    - The `action_space` is `spaces.Discrete(len(actions))`.

    ---
    
    ## Observation Space

    Observations are **concatenated real and imaginary parts** of the full statevector:

    ```python
    obs = np.concatenate([np.real(state), np.imag(state)])
    ```

    - Shape: `(2**n_qubits * 2,)`
    - Dtype: `float32`
    - Each element corresponds to the real or imaginary part of a computational basis amplitude.
    - This representation allows RL agents to process the quantum state as a vector of continuous values.

    ---
    
    ## Rewards

    Rewards are based on the **fidelity** between the current quantum state and the target state:

    ```python
    reward = |⟨target|state⟩|^2
    ```

    - Continuous reward in `[0, 1]`.
    - Higher reward means the current state is closer to the target entangled state.
    - An episode terminates when:
        1. **Success:** `reward >= reward_tolerance` (default 0.99)
        2. **Truncation:** `steps >= max_steps` (default 20)

    ---
    
    ## Reset Behavior

    On `reset()`:

    - `steps` is set to 0
    - The quantum state is initialized to |0...0⟩
    - History is initialized with the current observation
    - Returns the first observation and an empty `info` dict

    Optionally, the target state can be customized by passing a `target_state` argument
    to the constructor; otherwise, the default is the **GHZ-like superposition**:

    ```text
    |target⟩ = (|0...0⟩ + |1...1⟩) / √2
    ```

    ---
    
    ## Step Behavior

    `step(action)`:

    1. Applies the specified gate to the current quantum state.
    2. Updates the observation (`obs`) with the new statevector.
    3. Calculates the reward (fidelity with the target state).
    4. Increments the step counter.
    5. Checks for `done` conditions (reward threshold or max steps).
    6. Appends the observation to `history`.
    7. Returns `(obs, reward, done, info)` as per Gymnasium convention.

    ---
    
    ## Render

    `render()` visualizes **amplitudes of computational basis states**:

    - Creates a bar plot showing `|α_k|^2` for each basis state |k⟩
    - Updates step count in the plot title
    - Can be extended to include animations or Bloch sphere visualizations for individual qubits
    - Optional: save the plot using `save_path`

    ---
    
    ## Notes & Extensions

    - Currently supports **pure states** represented as statevectors.
    - Could be extended to **mixed states** or include **noise channels**.
    - Reward shaping or step penalties can encourage faster entanglement.
    - Observation augmentation (step number, recent gate, fidelity) can improve RL training.
    """

    def __init__(self, n_qubits=2, target_state=None, max_steps=20, reward_tolerance=0.99):
        super().__init__(n_qubits=n_qubits)
        self.n_qubits = n_qubits
        self.max_steps = max_steps
        self.reward_tolerance = reward_tolerance
        self.steps = 0
        self.history = []

        # PennyLane device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # Default target state: GHZ for n_qubits
        if target_state is None:
            self.target_state = np.zeros(2**n_qubits, dtype=complex)
            self.target_state[0] = 1/np.sqrt(2)
            self.target_state[-1] = 1/np.sqrt(2)
        else:
            self.target_state = target_state

        # Define action list
        self.actions = []
        for i in range(n_qubits):
            self.actions += [f"H_{i}", f"X_{i}", f"Y_{i}", f"Z_{i}", f"RX_{i}_pi_2", f"RY_{i}_pi_2", f"RZ_{i}_pi_2"]
        for i in range(n_qubits):
            for j in range(n_qubits):
                if i != j:
                    self.actions.append(f"CNOT_{i}_{j}")

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(2**n_qubits*2,), dtype=np.float32
        )

        # Initialize state
        self.state = np.zeros(2**n_qubits, dtype=complex)
        self.state[0] = 1.0

    def _state_to_obs(self, state):
        """Convert statevector to concatenated real+imag observation."""
        return np.concatenate([np.real(state), np.imag(state)]).astype(np.float32)

    def _apply_gate(self, gate):
        """Use PennyLane QNode to apply the gate and return new statevector."""
        @qml.qnode(self.dev)
        def circuit():
            # Initialize current state
            qml.StatePrep(self.state, wires=range(self.n_qubits))

            # Parse gate
            if "H" in gate:
                i = int(gate.split("_")[1])
                qml.Hadamard(wires=i)
            elif "X" in gate:
                i = int(gate.split("_")[1])
                qml.PauliX(wires=i)
            elif "Y" in gate:
                i = int(gate.split("_")[1])
                qml.PauliY(wires=i)
            elif "Z" in gate:
                i = int(gate.split("_")[1])
                qml.PauliZ(wires=i)
            elif "RX" in gate:
                i = int(gate.split("_")[1])
                qml.RX(np.pi/2, wires=i)
            elif "RY" in gate:
                i = int(gate.split("_")[1])
                qml.RY(np.pi/2, wires=i)
            elif "RZ" in gate:
                i = int(gate.split("_")[1])
                qml.RZ(np.pi/2, wires=i)
            elif "CNOT" in gate:
                parts = gate.split("_")
                control, target = int(parts[1]), int(parts[2])
                qml.CNOT(wires=[control, target])
            return qml.state()
        
        return circuit()

    def reset(self):
        self.steps = 0
        self.state = np.zeros(2**self.n_qubits, dtype=complex)
        self.state[0] = 1.0
        self.history = [self._state_to_obs(self.state)]
        return self._state_to_obs(self.state), {}

    def step(self, action):
        gate = self.actions[action]
        self.state = self._apply_gate(gate)
        obs = self._state_to_obs(self.state)
        reward = np.abs(np.vdot(self.target_state, self.state))**2
        done = reward > self.reward_tolerance or self.steps >= self.max_steps
        self.steps += 1
        self.history.append(obs)
        return obs, reward, done, {}

    def render(self, save_path=None, interval=800):
        """Visualize amplitudes of computational basis states."""
        fig, ax = plt.subplots(figsize=(6,6))
        amplitudes = np.abs(self.state)**2
        ax.bar(range(len(amplitudes)), amplitudes)
        ax.set_xlabel("Computational basis state")
        ax.set_ylabel("Probability")
        ax.set_title(f"Step {self.steps}")
        plt.show()
