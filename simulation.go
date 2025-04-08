package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"math/cmplx"
	"runtime" // For SetFinalizer, though explicit Close is better
	"time"

	// For FFT: Use Go's standard library's lack of built-in multi-D FFT
	// or choose a third-party library. We'll use mjibson/go-dsp/fft
	// which requires flattening.
	"github.com/mjibson/go-dsp/fft"

	// For linspace-like functionality (optional, can write manually)
	"gonum.org/v1/gonum/floats"
)

// NewSimulation creates and initializes a new GPE3DSimulation instance.
// It sets up grids, potentials, initial state, and precomputes operators.
// Corresponds roughly to the GPE3DSimulation.__init__ method in Python.
func NewSimulation(params SimParams, updateChan chan<- PlotData, controlChan <-chan ControlMsg) (*GPE3DSimulation, error) {
	// --- Input Validation ---
	if params.Nx <= 0 || params.Ny <= 0 || params.Nz <= 0 {
		return nil, errors.New("grid dimensions (Nx, Ny, Nz) must be positive")
	}
	if params.L <= 0 {
		return nil, errors.New("axis Length (L) must be positive")
	}
	// Basic validation for omegas (allow >= 0, negatives were warned about in dialog)
	// No error here, but could add one if needed.

	// --- Context for Shutdown ---
	// Create a cancellable context for managing the simulation goroutine's lifecycle.
	ctx, cancel := context.WithCancel(context.Background())

	// --- Initialize Struct ---
	sim := &GPE3DSimulation{
		params:      params,
		hbar:        1.0,    // Set physical constants
		m:           1.0,
		dt:          0.001,  // Set default time step (consider making configurable later)
		gVal:        500.0,  // Default interaction strength (will be updated by GUI slider)
		updateChan:  updateChan,  // Store communication channels
		controlChan: controlChan,
		ctx:         ctx,    // Store context and cancel function
		cancel:      cancel,
		// mu, wg are zero-initialized correctly
		// slices (x,y,z, psi, etc.) will be allocated in helper methods
		vortexChan: make(chan VortexInfo, 10), // Buffered channel for vortex requests
	}

	// Set a finalizer as a fallback for resource cleanup, although explicit
	// signaling and waiting via context/waitgroup is preferred.
	// The `Close` method isn't strictly necessary for this CPU version
	// unless file handles or other OS resources were used.
	runtime.SetFinalizer(sim, func(s *GPE3DSimulation) {
		log.Println("Warning: GPE3DSimulation finalizer called (indicates potential lack of explicit shutdown)")
		s.cancel() // Ensure context is cancelled if object is GC'd
	})

	// --- Setup Grids ---
	log.Println("Creating simulation grids...")
	err := sim.createGrids() // Calculate dx, dy, dz, dv and allocate/fill grid arrays
	if err != nil {
		sim.cancel() // Cancel context if setup fails
		return nil, fmt.Errorf("failed to create grids: %w", err)
	}
	log.Printf("Grid spacing: dx=%.4f, dy=%.4f, dz=%.4f", sim.dx, sim.dy, sim.dz)

	// --- Calculate Potential ---
	log.Println("Calculating potential...")
	sim.potentialV = sim.calculatePotential() // Calculate V(x,y,z)

	// --- Calculate Initial State ---
	log.Println("Calculating initial wavefunction...")
	sim.psi = sim.initialState() // Calculate psi(x,y,z, t=0)
	// Deep copy the initial state for the reset function.
	sim.initialPsi = deepCopyPsi(sim.psi)
	if sim.initialPsi == nil {
		sim.cancel()
		return nil, errors.New("failed to copy initial state")
	}

	// --- Precompute Operators ---
	log.Println("Precomputing kinetic operator...")
	sim.expK = sim.precomputeExpK() // Calculate exp(-i*K*dt/2/hbar) in k-space

	log.Println("Simulation engine initialized successfully.")
	return sim, nil
}

// createGrids calculates grid parameters (dx, dy, dz, etc.) and allocates
// and fills the coordinate (x, y, z) and momentum (kx, ky, kz, kSq) grid arrays.
func (sim *GPE3DSimulation) createGrids() error {
	nx, ny, nz := sim.params.Nx, sim.params.Ny, sim.params.Nz
	L := sim.params.L // Use float64 for precision

	// 1D coordinate arrays (centers of grid cells)
	xLin := make([]float64, nx)
	yLin := make([]float64, ny)
	zLin := make([]float64, nz)

	// Calculate grid spacings (handle dimensions of size 1)
	if nx > 1 {
		// floats.Span(dst, start, end) generates len(dst) points from start to end inclusive.
		// We want endpoint=False like numpy.linspace, so the effective length is L.
		// The spacing is L/nx. The last point is L/2 - L/nx.
		endX := L/2.0 - L/float64(nx)
		floats.Span(xLin, -L/2.0, endX)
		sim.dx = L / float64(nx) // Or xLin[1] - xLin[0]
	} else {
		xLin[0] = 0.0 // Center at 0 if only one point
		sim.dx = L
	}
	// Repeat for Y and Z dimensions
	if ny > 1 {
		endY := L/2.0 - L/float64(ny)
		floats.Span(yLin, -L/2.0, endY)
		sim.dy = L / float64(ny)
	} else {
		yLin[0] = 0.0
		sim.dy = L
	}
	if nz > 1 {
		endZ := L/2.0 - L/float64(nz)
		floats.Span(zLin, -L/2.0, endZ)
		sim.dz = L / float64(nz)
	} else {
		zLin[0] = 0.0
		sim.dz = L
	}

	// Calculate volume element
	sim.dv = sim.dx * sim.dy * sim.dz

	// Store grid edges for plotting extent: [left, right, bottom, top]
	// Edges are halfway between the first/last point and the next virtual point.
	sim.xEdges = [2]float64{xLin[0] - sim.dx/2.0, xLin[nx-1] + sim.dx/2.0}
	sim.yEdges = [2]float64{yLin[0] - sim.dy/2.0, yLin[ny-1] + sim.dy/2.0}
	sim.zEdges = [2]float64{zLin[0] - sim.dz/2.0, zLin[nz-1] + sim.dz/2.0}

	// --- Allocate 3D Grid Arrays ---
	sim.x = make([][][]float64, nx)
	sim.y = make([][][]float64, nx)
	sim.z = make([][][]float64, nx)
	sim.kx = make([][][]float64, nx)
	sim.ky = make([][][]float64, nx)
	sim.kz = make([][][]float64, nx)
	sim.kSq = make([][][]float64, nx)
	for i := 0; i < nx; i++ {
		sim.x[i] = make([][]float64, ny)
		sim.y[i] = make([][]float64, ny)
		sim.z[i] = make([][]float64, ny)
		sim.kx[i] = make([][]float64, ny)
		sim.ky[i] = make([][]float64, ny)
		sim.kz[i] = make([][]float64, ny)
		sim.kSq[i] = make([][]float64, ny)
		for j := 0; j < ny; j++ {
			sim.x[i][j] = make([]float64, nz)
			sim.y[i][j] = make([]float64, nz)
			sim.z[i][j] = make([]float64, nz)
			sim.kx[i][j] = make([]float64, nz)
			sim.ky[i][j] = make([]float64, nz)
			sim.kz[i][j] = make([]float64, nz)
			sim.kSq[i][j] = make([]float64, nz)
		}
	}

	// --- Fill Coordinate Grids ---
	for i := 0; i < nx; i++ {
		for j := 0; j < ny; j++ {
			for k := 0; k < nz; k++ {
				sim.x[i][j][k] = xLin[i]
				sim.y[i][j][k] = yLin[j]
				sim.z[i][j][k] = zLin[k]
			}
		}
	}

	// --- Calculate Momentum Space Grids (kx, ky, kz, kSq) ---
	// Use the convention matching `fft.FFTFreq` which is:
	// freqs = [0, 1, ..., n/2-1, -n/2, ..., -1] / (d*n) where n=dimension, d=spacing
	// k = 2 * pi * freqs
	kxVec := make([]float64, nx)
	kyVec := make([]float64, ny)
	kzVec := make([]float64, nz)

	// Helper to calculate FFT frequencies scaled for k-vector
	calcKFreqs := func(n int, d float64) []float64 {
		kVec := make([]float64, n)
		scale := 2.0 * math.Pi / (float64(n) * d) // 2*pi / L_total
		for i := 0; i < n; i++ {
			var freq float64
			if i < (n+1)/2 { // 0 and positive frequencies
				freq = float64(i)
			} else { // negative frequencies
				freq = float64(i - n)
			}
			kVec[i] = freq * scale
		}
		return kVec
	}

	kxVec = calcKFreqs(nx, sim.dx)
	kyVec = calcKFreqs(ny, sim.dy)
	kzVec = calcKFreqs(nz, sim.dz)

	// Fill momentum grids and kSq
	for i := 0; i < nx; i++ {
		for j := 0; j < ny; j++ {
			for k := 0; k < nz; k++ {
				sim.kx[i][j][k] = kxVec[i]
				sim.ky[i][j][k] = kyVec[j]
				sim.kz[i][j][k] = kzVec[k]
				sim.kSq[i][j][k] = kxVec[i]*kxVec[i] + kyVec[j]*kyVec[j] + kzVec[k]*kzVec[k]
			}
		}
	}
	return nil // Success
}

// calculatePotential computes the time-independent harmonic potential V(x,y,z).
func (sim *GPE3DSimulation) calculatePotential() [][][]float64 {
	nx, ny, nz := sim.params.Nx, sim.params.Ny, sim.params.Nz
	ox := sim.params.OmegaX // Use parameters stored in the struct
	oy := sim.params.OmegaY
	oz := sim.params.OmegaZ
	m := sim.m

	potential := make([][][]float64, nx)
	for i := 0; i < nx; i++ {
		potential[i] = make([][]float64, ny)
		for j := 0; j < ny; j++ {
			potential[i][j] = make([]float64, nz)
			for k := 0; k < nz; k++ {
				// Get coordinates for this point (already calculated in sim.x, sim.y, sim.z)
				x := sim.x[i][j][k]
				y := sim.y[i][j][k]
				z := sim.z[i][j][k]
				// Calculate potential: 0.5 * m * (omega_x^2 * x^2 + ...)
				potential[i][j][k] = 0.5 * m * (ox*ox*x*x + oy*oy*y*y + oz*oz*z*z)
			}
		}
	}
	return potential
}

// initialState calculates the initial state (wavefunction) psi at t=0.
// Uses a Gaussian profile based on harmonic oscillator ground state widths.
// Handles anisotropic potentials and normalizes the state.
func (sim *GPE3DSimulation) initialState() [][][]complex128 {
	nx, ny, nz := sim.params.Nx, sim.params.Ny, sim.params.Nz
	L := sim.params.L
	ox := sim.params.OmegaX
	oy := sim.params.OmegaY
	oz := sim.params.OmegaZ
	m := sim.m
	hbar := sim.hbar

	psi0 := make([][][]complex128, nx) // Allocate space for the wavefunction

	// --- Calculate Gaussian widths ---
	// Use harmonic oscillator ground state width: sigma_i = sqrt(hbar / (m * omega_i))
	// Handle omega = 0 case to avoid division by zero, use L/scale instead.
	widthScale := 4.0 // Arbitrary scale if omega is zero
	var sigmaX, sigmaY, sigmaZ float64

	if ox > 1e-9 { // Use threshold to check for non-zero omega
		sigmaX = math.Sqrt(hbar / (m * ox))
	} else {
		sigmaX = L / widthScale
		log.Printf("OmegaX is near zero, using width L/%.1f = %.3f", widthScale, sigmaX)
	}
	if oy > 1e-9 {
		sigmaY = math.Sqrt(hbar / (m * oy))
	} else {
		sigmaY = L / widthScale
		log.Printf("OmegaY is near zero, using width L/%.1f = %.3f", widthScale, sigmaY)
	}
	if oz > 1e-9 {
		sigmaZ = math.Sqrt(hbar / (m * oz))
	} else {
		sigmaZ = L / widthScale
		log.Printf("OmegaZ is near zero, using width L/%.1f = %.3f", widthScale, sigmaZ)
	}

	// Clamp widths to prevent them being too large relative to the box size.
	// This avoids issues with normalization or periodic boundaries if they were used.
	sigmaX = math.Min(sigmaX, L/2.1)
	sigmaY = math.Min(sigmaY, L/2.1)
	sigmaZ = math.Min(sigmaZ, L/2.1)

	log.Printf("Initial state Gaussian widths: σx=%.3f, σy=%.3f, σz=%.3f", sigmaX, sigmaY, sigmaZ)

	// --- Fill Wavefunction with Gaussian ---
	var totalNormSq float64 // Accumulator for normalization integral (|psi|^2 * dv)
	for i := 0; i < nx; i++ {
		psi0[i] = make([][]complex128, ny)
		for j := 0; j < ny; j++ {
			psi0[i][j] = make([]complex128, nz)
			for k := 0; k < nz; k++ {
				x := sim.x[i][j][k]
				y := sim.y[i][j][k]
				z := sim.z[i][j][k]

				// Calculate exponent: -(x^2/(2*sx^2) + y^2/(2*sy^2) + z^2/(2*sz^2))
				expArg := -((x*x)/(2*sigmaX*sigmaX) + (y*y)/(2*sigmaY*sigmaY) + (z*z)/(2*sigmaZ*sigmaZ))

				// Value is exp(expArg), which is real for a standard Gaussian initial state.
				valReal := math.Exp(expArg)
				psi0[i][j][k] = complex(valReal, 0)

				// Accumulate norm squared: |psi|^2 = valReal^2
				totalNormSq += valReal * valReal
			}
		}
	}

	// --- Normalize ---
	// norm = sqrt( integral(|psi|^2 dV) ) = sqrt( sum(|psi_ijk|^2) * dv )
	norm := math.Sqrt(totalNormSq * sim.dv)
	log.Printf("Initial state norm before normalization: %.5f", norm)

	if norm < 1e-15 { // Check for near-zero norm (e.g., if widths were tiny)
		log.Println("Warning: Initial state norm is near zero. Cannot normalize.")
		// Return the zero-ish state. The simulation might diverge or be trivial.
		return psi0
	}

	// Divide every element by the norm.
	normInv := 1.0 / norm
	for i := 0; i < nx; i++ {
		for j := 0; j < ny; j++ {
			for k := 0; k < nz; k++ {
				psi0[i][j][k] *= complex(normInv, 0)
			}
		}
	}

	return psi0
}

// precomputeExpK calculates the kinetic energy evolution operator in k-space.
// expK = exp(-0.5i * hbar / m * k_sq * dt)
func (sim *GPE3DSimulation) precomputeExpK() [][][]complex128 {
	nx, ny, nz := sim.params.Nx, sim.params.Ny, sim.params.Nz
	hbar := sim.hbar
	m := sim.m
	dt := sim.dt

	expK := make([][][]complex128, nx) // Allocate result array

	// Calculate the factor in the exponent's argument: -0.5 * hbar / m * dt
	// The 'i' will be handled by using cmplx.Exp(complex(0, arg))
	factor := -0.5 * hbar / m * dt

	for i := 0; i < nx; i++ {
		expK[i] = make([][]complex128, ny)
		for j := 0; j < ny; j++ {
			expK[i][j] = make([]complex128, nz)
			for k := 0; k < nz; k++ {
				kSq := sim.kSq[i][j][k]        // Get precalculated k^2 for this point
				arg := factor * kSq            // Argument for exp(i * arg)
				expK[i][j][k] = cmplx.Exp(complex(0, arg)) // Calculate exp(i*arg)
			}
		}
	}
	return expK
}

// deepCopyPsi creates a deep copy of a 3D complex wavefunction array.
// Necessary for storing the initial state correctly for reset.
func deepCopyPsi(psiIn [][][]complex128) [][][]complex128 {
	if psiIn == nil {
		return nil
	}
	nx := len(psiIn)
	if nx == 0 {
		return [][][]complex128{}
	}
	ny := len(psiIn[0])
	if ny == 0 {
		return make([][][]complex128, nx)
	}
	nz := len(psiIn[0][0])
	if nz == 0 {
		// Handle case where inner slices might be empty
		psiOut := make([][][]complex128, nx)
		for i := range psiOut {
			psiOut[i] = make([][]complex128, ny)
			// No need to make inner slice if nz is 0
		}
		return psiOut
	}

	// Allocate the new 3D slice
	psiOut := make([][][]complex128, nx)
	for i := range psiOut {
		psiOut[i] = make([][]complex128, ny)
		for j := range psiOut[i] {
			psiOut[i][j] = make([]complex128, nz)
			// Copy the innermost slice
			copy(psiOut[i][j], psiIn[i][j])
		}
	}
	return psiOut
}

// Close is a placeholder for resource cleanup. Not strictly needed for CPU version
// unless files or other OS resources are opened, but good practice for consistency
// with potential GPU versions or future extensions. It ensures the context is cancelled.
func (sim *GPE3DSimulation) Close() {
	log.Println("Closing simulation object (calling cancel context).")
	sim.cancel() // Signal any running goroutines (like sim.Run) to stop.
}


// potentialStep performs the potential+interaction part of the evolution for half a time step.
// psi_out = exp(-0.5i * V_eff * dt / hbar) * psi_in
// V_eff = V_harmonic + g * |psi_in|^2
// This function creates a *new* array for the output to avoid modifying the input directly.
func (sim *GPE3DSimulation) potentialStep(psiIn [][][]complex128) [][][]complex128 {
	nx, ny, nz := sim.params.Nx, sim.params.Ny, sim.params.Nz
	dt := sim.dt
	hbar := sim.hbar
	// Read gVal under RLock, as it might be changed concurrently by the GUI thread.
	sim.mu.RLock()
	g := sim.gVal
	sim.mu.RUnlock()

	// Allocate output array
	psiOut := make([][][]complex128, nx)

	// Calculate the exponent pre-factor: -0.5 * dt / hbar
	// The 'i' will be handled by cmplx.Exp(complex(0, arg))
	expFactor := -0.5 * dt / hbar

	for i := 0; i < nx; i++ {
		psiOut[i] = make([][]complex128, ny)
		for j := 0; j < ny; j++ {
			psiOut[i][j] = make([]complex128, nz)
			for k := 0; k < nz; k++ {
				psiIn_ijk := psiIn[i][j][k]

				// Calculate |psi|^2 = real^2 + imag^2
				psiAbsSq := cmplx.Abs(psiIn_ijk) * cmplx.Abs(psiIn_ijk)

				// Calculate effective potential V_eff = V_harmonic + g * |psi|^2
				vEff := sim.potentialV[i][j][k] + g*psiAbsSq

				// Calculate the argument for exp(i * arg)
				arg := expFactor * vEff

				// Calculate exp(i * arg)
				expTerm := cmplx.Exp(complex(0, arg))

				// Multiply input wavefunction by the exponential factor
				psiOut[i][j][k] = expTerm * psiIn_ijk
			}
		}
	}
	return psiOut
}


// kineticStep performs the kinetic part of the evolution using FFTs.
// Operates for a full time step dt.
// Steps: FFT -> Multiply by precomputed expK -> Inverse FFT
// Uses mjibson/go-dsp/fft, which requires flattening the 3D array.
// Returns a *new* array for the output.
// kineticStep performs the kinetic part of the evolution using sequential 1D FFTs.
// Operates for a full time step dt.
// Steps: FFT(Z) -> FFT(Y) -> FFT(X) -> Multiply by expK -> IFFT(X) -> IFFT(Y) -> IFFT(Z) -> Normalize
// Returns a *new* array for the output.
func (sim *GPE3DSimulation) kineticStep(psiIn [][][]complex128) [][][]complex128 {
	nx, ny, nz := sim.params.Nx, sim.params.Ny, sim.params.Nz
	//N := nx * ny * nz // Total number of points

	// --- Create intermediate and output arrays ---
	// It's often clearer to work with temporary arrays for each FFT direction.
	// Deep copy input to avoid modifying it if psiIn is sim.psi directly
	tempPsi := deepCopyPsi(psiIn)
	if tempPsi == nil {
        log.Println("Error: Failed to copy input psi in kineticStep")
        return nil
    }
    psiK := make([][][]complex128, nx) // To store result after all FFTs
    for i := range psiK {
        psiK[i] = make([][]complex128, ny)
        for j := range psiK[i] {
            psiK[i][j] = make([]complex128, nz)
        }
    }


	// --- Forward FFT (Dimension by Dimension) ---

	// 1. FFT along Z (for each X, Y pair)
	sliceZ := make([]complex128, nz) // Reusable buffer for Z slices
	for i := 0; i < nx; i++ {
		for j := 0; j < ny; j++ {
			// Extract the Z-slice
			for k := 0; k < nz; k++ {
				sliceZ[k] = tempPsi[i][j][k]
			}
			// Perform 1D FFT
			fftSliceZ := fft.FFT(sliceZ)
			// Store the result back (or in psiK if doing in-place concept)
            // Storing back into tempPsi for the next dimension's FFT
			for k := 0; k < nz; k++ {
				tempPsi[i][j][k] = fftSliceZ[k]
			}
		}
	}

	// 2. FFT along Y (for each X, Z pair)
	sliceY := make([]complex128, ny) // Reusable buffer for Y slices
	for i := 0; i < nx; i++ {
		for k := 0; k < nz; k++ {
			// Extract the Y-slice
			for j := 0; j < ny; j++ {
				sliceY[j] = tempPsi[i][j][k] // Read from the result of Z-FFT
			}
			// Perform 1D FFT
			fftSliceY := fft.FFT(sliceY)
			// Store the result back
			for j := 0; j < ny; j++ {
				tempPsi[i][j][k] = fftSliceY[j]
			}
		}
	}

	// 3. FFT along X (for each Y, Z pair)
	sliceX := make([]complex128, nx) // Reusable buffer for X slices
	for j := 0; j < ny; j++ {
		for k := 0; k < nz; k++ {
			// Extract the X-slice
			for i := 0; i < nx; i++ {
				sliceX[i] = tempPsi[i][j][k] // Read from the result of Y-FFT
			}
			// Perform 1D FFT
			fftSliceX := fft.FFT(sliceX)
			// Store the final k-space result in psiK
			for i := 0; i < nx; i++ {
				psiK[i][j][k] = fftSliceX[i] // Store in dedicated k-space array
			}
		}
	}
    // tempPsi is no longer needed

	// --- Apply Kinetic Operator in k-space ---
	// Multiply element-wise: psiK[i][j][k] *= expK[i][j][k]
	for i := 0; i < nx; i++ {
		for j := 0; j < ny; j++ {
			for k := 0; k < nz; k++ {
				psiK[i][j][k] *= sim.expK[i][j][k] // Use the precomputed 3D expK
			}
		}
	}

	// --- Inverse FFT (Dimension by Dimension) ---
    // We can reuse tempPsi as the output buffer now, starting with psiK data
    tempPsi = psiK // Point tempPsi to the k-space data to start IFFTs


	// 4. IFFT along X (for each Y, Z pair)
    // sliceX is already allocated
	for j := 0; j < ny; j++ {
		for k := 0; k < nz; k++ {
			// Extract X-slice from k-space data
			for i := 0; i < nx; i++ {
				sliceX[i] = tempPsi[i][j][k]
			}
			// Perform 1D Inverse FFT
			ifftSliceX := fft.IFFT(sliceX) // Note: fft.IFFT is unnormalized
			// Store result back
			for i := 0; i < nx; i++ {
				tempPsi[i][j][k] = ifftSliceX[i]
			}
		}
	}

	// 5. IFFT along Y (for each X, Z pair)
    // sliceY is already allocated
	for i := 0; i < nx; i++ {
		for k := 0; k < nz; k++ {
			// Extract Y-slice
			for j := 0; j < ny; j++ {
				sliceY[j] = tempPsi[i][j][k]
			}
			// Perform 1D Inverse FFT
			ifftSliceY := fft.IFFT(sliceY)
			// Store result back
			for j := 0; j < ny; j++ {
				tempPsi[i][j][k] = ifftSliceY[j]
			}
		}
	}

	// 6. IFFT along Z (for each X, Y pair)
    // sliceZ is already allocated
    psiOut := tempPsi // The final result will be in tempPsi after this loop
	for i := 0; i < nx; i++ {
		for j := 0; j < ny; j++ {
			// Extract Z-slice
			for k := 0; k < nz; k++ {
				sliceZ[k] = tempPsi[i][j][k]
			}
			// Perform 1D Inverse FFT
			ifftSliceZ := fft.IFFT(sliceZ)
			// Store final real-space result
			for k := 0; k < nz; k++ {
				psiOut[i][j][k] = ifftSliceZ[k]
			}
		}
	}

	// --- Normalize ---
	// Since fft.IFFT applies a 1/n factor for each dimension's inverse transform,
	// the total normalization factor applied by the sequential IFFTs is (1/nx)*(1/ny)*(1/nz) = 1/N.
	// Therefore, no additional normalization is needed here if using fft.IFFT.
	// If fft.IFFT was unnormalized, you would multiply by 1/N here.
    // Let's double-check mjibson/fft doc: `IFFT` *does* apply the 1/N scaling. So we are good.

	return psiOut // Return the final result
}


// step performs a single simulation time step using the split-operator method.
// psi(t+dt) = exp(-iVdt/2) * exp(-iKdt) * exp(-iVdt/2) * psi(t)
// It modifies the internal simulation state (sim.psi, sim.time).
// Assumes the caller handles locking if necessary (e.g., Run method does).
func (sim *GPE3DSimulation) step() error {
    //log.Println("Starting simulation step...")
    // Perform the first potential half-step
    psiMid1 := sim.potentialStep(sim.psi)
    if psiMid1 == nil {
        log.Println("Error: First potential step failed.")
        return errors.New("potentialStep (1st half) failed")
    }

    // Perform the full kinetic step
    psiMid2 := sim.kineticStep(psiMid1)
    if psiMid2 == nil {
        log.Println("Error: Kinetic step failed.")
        return errors.New("kineticStep failed")
    }

    // Perform the second potential half-step
    psiFinal := sim.potentialStep(psiMid2)
    if psiFinal == nil {
        log.Println("Error: Second potential step failed.")
        return errors.New("potentialStep (2nd half) failed")
    }

    // Update the simulation state
    sim.psi = psiFinal
    sim.time += sim.dt
    //log.Printf("Simulation step completed. Time: %.3f", sim.time)
    return nil
}


// Run is the main loop for the simulation goroutine.
// It listens for control messages, performs simulation steps when running,
// and sends plot data updates to the GUI.
func (sim *GPE3DSimulation) Run() {
    defer sim.wg.Done()
    defer close(sim.updateChan)
    log.Println("Simulation Run() goroutine started.")

    ticker := time.NewTicker(20 * time.Millisecond)
    defer ticker.Stop()

    for {
        select {
        case <-sim.ctx.Done():
            log.Println("Simulation context cancelled. Exiting Run().")
            return
        case msg := <-sim.controlChan: // Process control messages
            log.Printf("Received control message: %v", msg)
            sim.handleControlMessage(msg)
			case vortex := <-sim.vortexChan: // Process vortex imprint requests
            log.Printf("Processing vortex imprint: (%.2f, %.2f) with charge %d.", vortex.X0, vortex.Y0, vortex.Charge)
            sim.applyVortex(vortex)
        case <-ticker.C:
            if sim.running {
                //log.Println("Performing simulation step...")
                err := sim.step()
                if err != nil {
                    log.Printf("Error during simulation step: %v", err)
                    sim.updateChan <- PlotData{Error: err}
                    continue
                }

                //log.Println("Sending plot data to update channel...")
                sim.mu.RLock() // Acquire read lock only for accessing sim.psi
                densitySlice, extent := sim.getDensitySliceFromData(sim.psi, 'z', sim.params.Nz/2)
                phaseSlice, _ := sim.getPhaseSliceFromData(sim.psi, 'z', sim.params.Nz/2)
                sim.mu.RUnlock() // Release read lock immediately
				
				select {
				case sim.updateChan <- PlotData{
                    Time:         sim.time,
                    DensitySlice: densitySlice,
                    PhaseSlice:   phaseSlice,
                    Extent:       extent,
                }:
					//log.Println("Plot data sent to GUI")
				default:
					log.Println("Update channel full, skipping plot update.")
				}
            }
        }
    }
}

// handleControlMessage processes commands received from the GUI.
// It modifies the simulation state based on the command.
// Assumes the caller (Run method) holds the Write Lock if state modification is needed.
func (sim *GPE3DSimulation) handleControlMessage(msg ControlMsg) {
    // Acquire write lock to modify simulation state
    sim.mu.Lock()
    defer sim.mu.Unlock()

    switch msg.Command {
    case "start":
        if !sim.running {
            sim.running = true
            log.Println("Simulation started via control message.")
        } else {
            log.Println("Simulation is already running.")
        }
    case "stop":
        if sim.running {
            sim.running = false
            log.Println("Simulation stopped via control message.")
        }
    case "reset":
        log.Println("Resetting simulation state...")
        sim.psi = deepCopyPsi(sim.initialPsi)
        sim.time = 0.0
        sim.vortices = nil
        log.Println("Simulation state reset.")
    case "update_g":
        log.Printf("Updating interaction strength (g) to %.2f.", msg.GValue)
        sim.gVal = msg.GValue
    case "imprint":
        log.Printf("Imprinting vortex at (%.2f, %.2f) with charge %d.",
            msg.Vortex.X0, msg.Vortex.Y0, msg.Vortex.Charge)
        sim.imprintVortex(msg.Vortex.X0, msg.Vortex.Y0, msg.Vortex.Charge)
    default:
        log.Printf("Unknown control command: %s", msg.Command)
    }
}

// imprintVortex applies a phase vortex to the wavefunction sim.psi.
// psi(x,y,z) *= exp(i * charge * atan2(x - x0, -(y - y0)))
// This method directly modifies sim.psi. The caller (handleControlMessage)
// should ensure the simulation is stopped and hold the necessary lock.
func (sim *GPE3DSimulation) imprintVortex(x0, y0 float64, charge int) {
    // If charge is zero, there's nothing to do.
	if charge == 0 {
        log.Println("Charge is zero, no vortex to imprint.")
        return
    }

    vortexInfo := VortexInfo{X0: x0, Y0: y0, Charge: charge}
    select {
    case sim.vortexChan <- vortexInfo:
        log.Printf("Vortex imprint request queued: (%.2f, %.2f) with charge %d.", x0, y0, charge)
    default:
        log.Println("Vortex channel full, cannot queue imprint request.")
    }
}

func (sim *GPE3DSimulation) applyVortex(vortex VortexInfo) {
    sim.mu.Lock()
    defer sim.mu.Unlock()

    nx, ny, nz := sim.params.Nx, sim.params.Ny, sim.params.Nz
    chargeF := float64(vortex.Charge)

    // Perform the vortex imprint operation
    for i := 0; i < nx; i++ {
        for j := 0; j < ny; j++ {
            for k := 0; k < nz; k++ {
                dx := sim.x[i][j][k] - vortex.X0
                dy := sim.y[i][j][k] - vortex.Y0
                phaseShift := chargeF * math.Atan2(dy, dx)
                sim.psi[i][j][k] *= cmplx.Exp(complex(0, phaseShift))
            }
        }
    }

    log.Printf("Vortex imprinted at (%.2f, %.2f) with charge %d.", vortex.X0, vortex.Y0, vortex.Charge)
}
