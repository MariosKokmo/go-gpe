package main

import (
	"context"
	"sync"
)

// PlotData is sent FROM the simulation goroutine TO the GUI goroutine
// via the updateChan. It contains the necessary information for the GUI
// to update the plots and labels.
type PlotData struct {
	// Current simulation time.
	Time float64

	// 2D slice representing the density (|psi|^2) for the current plot view.
	// The slice is typically transposed for standard plot orientation (Y vs X).
	DensitySlice [][]float64

	// 2D slice representing the phase (angle(psi)) for the current plot view.
	// The slice is typically transposed.
	PhaseSlice [][]float64

	// Spatial extent of the slices for plot axes labeling: [xmin, xmax, ymin, ymax].
	// Corresponds to the axes after transposition (usually X vs Y).
	Extent [4]float64

	// Indicates which axis was sliced ('x', 'y', or 'z').
	// Useful if the GUI were to allow changing the slice view.
	SliceAxis rune

	// If not nil, indicates an error occurred during simulation step or data prep.
	// The GUI should display this error instead of updating plots.
	Error error
}

// ControlMsg is sent FROM the GUI goroutine TO the simulation goroutine
// via the controlChan. It contains commands or parameter updates initiated
// by the user through the GUI.
type ControlMsg struct {
	// Command specifies the action to be taken (e.g., "start", "stop", "reset").
	Command string

	// GValue holds the new interaction strength 'g' when the command is "update_g".
	// It's also used to pass the current slider value during a "reset" command.
	GValue float64 // Use float64 to match GUI slider, simulation might convert to float32 if needed (e.g., GPU version)

	// Vortex holds information for the "imprint" command.
	Vortex VortexInfo // VortexInfo struct defined in simulation.go (or here)
}

// GPE3DSimulation holds the state and parameters of the simulation.
type GPE3DSimulation struct {
	// Mutex for thread-safe access to simulation state (psi, time, gVal, running)
	// Use RWMutex to allow multiple readers (e.g., for plotting) but exclusive writer (for step).
	mu sync.RWMutex

	// Configuration parameters received from the user.
	params SimParams

	// Physical constants (using float64 for CPU precision).
	hbar float64
	m    float64

	// Grid properties calculated from params.
	dx, dy, dz float64 // Grid spacing
	dv         float64 // Volume element (dx*dy*dz)
	xEdges     [2]float64 // Min/max x extent for plotting
	yEdges     [2]float64 // Min/max y extent for plotting
	zEdges     [2]float64 // Min/max z extent for plotting

	// Grid arrays (using Go slices).
	// Use complex128 for wavefunction, float64 for real-valued grids.
	x, y, z    [][][]float64      // Real space coordinates (Nx x Ny x Nz)
	kx, ky, kz [][][]float64      // Momentum space coordinates (Nx x Ny x Nz)
	kSq        [][][]float64      // kx^2 + ky^2 + kz^2 (Nx x Ny x Nz)
	potentialV [][][]float64      // Time-independent part of potential (Nx x Ny x Nz)
	psi        [][][]complex128 // The wavefunction (Nx x Ny x Nz)
	initialPsi [][][]complex128 // Stored initial state for reset (Nx x Ny x Nz)
	expK       [][][]complex128 // Precomputed kinetic operator exp( K*dt/2 ) (Nx x Ny x Nz)

	// Simulation time and step size.
	time float64 // Current simulation time
	dt   float64 // Time step size

	// Simulation parameters that can be changed dynamically.
	gVal float64 // Interaction strength (can be updated via GUI)

	// Vortex tracking (simple list of vortex properties).
	vortices []VortexInfo

	// Control state.
	running bool // Flag indicating if the simulation loop is active

	// Communication channels with the GUI.
	updateChan  chan<- PlotData    // Sends plot data updates TO the GUI
	controlChan <-chan ControlMsg // Receives control commands FROM the GUI

	vortexChan chan VortexInfo // New channel for vortex imprint requests
	
	// Context for graceful shutdown.
	ctx    context.Context    // Parent context
	cancel context.CancelFunc // Function to cancel the context

	// WaitGroup to synchronize shutdown.
	wg sync.WaitGroup
}

// VortexInfo stores information about imprinted vortices.
type VortexInfo struct {
	X0, Y0 float64 // Position (usually in the XY plane for this setup)
	Charge int     // Topological charge
}